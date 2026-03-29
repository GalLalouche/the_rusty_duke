//! Supervised training binary: train a pure NN on labeled positions.
//!
//! Supports arbitrary hidden layer depths, e.g. 1106->256->128->64->1.
//!
//! Usage:
//!   supervised_train --input D:/temp/labeled_positions.bin \
//!                    --hidden 256,128,64 \
//!                    --lr 0.001 \
//!                    --epochs 10 \
//!                    --batch-size 256 \
//!                    --label-scale 100 \
//!                    --eval-interval 50000 \
//!                    --eval-games 500 \
//!                    --benchmark base,random \
//!                    --checkpoint-dir D:/temp/supervised_nn

use std::time::Instant;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use duke_training::cli::parse_flag;
use duke_training::encoding::{BOARD_FEATURES, BAG_FEATURES, TOTAL_FEATURES};
use duke_training::game_setup::{create_bag, create_initial_state};
use duke_training::generic_mlp::{GenericMlp, GenericEvaluator, MAX_HIDDEN};
use duke_training::loaded_model::LoadedModel;
use duke_training::match_runner::{run_matches, win_rate, Player};

// ── Labeled position data ──────────────────────────────────────────────────

/// A single labeled position loaded from the LPOS file.
struct LabeledPosition {
    /// Active board feature indices (each < 1080).
    active_indices: Vec<u16>,
    /// Dense bag features (26 f32 values).
    bag_features: [f32; BAG_FEATURES],
    /// LR-Cheap depth-2 minimax label (roughly -1000 to +1000).
    label: f32,
    /// Frequency weight (how many times this position appeared).
    count: u32,
}

/// Load labeled positions from an LPOS binary file.
/// Returns (positions, min_label, max_label).
fn load_lpos(path: &str) -> (Vec<LabeledPosition>, f32, f32) {
    let t0 = Instant::now();
    eprintln!("Loading labeled positions from {} ...", path);

    let data = std::fs::read(path).expect("Failed to read LPOS file");
    let mut cursor = 0usize;

    // Helper: read N bytes from the data buffer.
    macro_rules! read_bytes {
        ($n:expr) => {{
            let end = cursor + $n;
            assert!(end <= data.len(), "Unexpected EOF at offset {}", cursor);
            let slice = &data[cursor..end];
            cursor = end;
            slice
        }};
    }
    macro_rules! read_u16 {
        () => {{
            let b = read_bytes!(2);
            u16::from_le_bytes([b[0], b[1]])
        }};
    }
    macro_rules! read_u32 {
        () => {{
            let b = read_bytes!(4);
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        }};
    }
    macro_rules! read_f32 {
        () => {{
            let b = read_bytes!(4);
            f32::from_le_bytes([b[0], b[1], b[2], b[3]])
        }};
    }

    // Header
    let magic = read_bytes!(4);
    assert_eq!(magic, b"LPOS", "Not an LPOS file (bad magic)");
    let version = read_u32!();
    assert_eq!(version, 1, "Unsupported LPOS version {}", version);
    let num_positions = read_u32!() as usize;

    eprintln!("  File header: {} positions, version {}", num_positions, version);

    let mut positions = Vec::with_capacity(num_positions);
    for _ in 0..num_positions {
        let num_active = read_u16!() as usize;
        let mut active_indices = Vec::with_capacity(num_active);
        for _ in 0..num_active {
            active_indices.push(read_u16!());
        }
        let mut bag_features = [0.0f32; BAG_FEATURES];
        for i in 0..BAG_FEATURES {
            bag_features[i] = read_f32!();
        }
        let label = read_f32!();
        let count = read_u32!();
        positions.push(LabeledPosition {
            active_indices,
            bag_features,
            label,
            count,
        });
    }

    let elapsed = t0.elapsed();
    let file_mb = data.len() as f64 / (1024.0 * 1024.0);
    eprintln!(
        "  Loaded {} positions ({:.1} MB) in {:.1}s",
        positions.len(),
        file_mb,
        elapsed.as_secs_f64()
    );

    // Print label statistics
    let labels: Vec<f32> = positions.iter().map(|p| p.label).collect();
    let min = labels.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = labels.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = labels.iter().map(|l| *l as f64).sum::<f64>() / labels.len() as f64;
    let total_count: u64 = positions.iter().map(|p| p.count as u64).sum();
    eprintln!(
        "  Label stats: min={:.2}, max={:.2}, mean={:.4}, total_count={}",
        min, max, mean, total_count
    );

    (positions, min, max)
}

// ── Helpers ────────────────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Clamp label to [-10, +10] then map linearly to [0, 1].
/// Normal positions (±6) get good spread (0.2..0.8).
/// Terminals (±30) clamp to 0/1.
const LABEL_CLAMP: f32 = 10.0;

#[inline]
fn label_to_target(label: f32) -> f32 {
    let clamped = label.clamp(-LABEL_CLAMP, LABEL_CLAMP);
    (clamped + LABEL_CLAMP) / (2.0 * LABEL_CLAMP)
}

// ── Forward pass with intermediates ────────────────────────────────────────

/// Pre-allocated scratch buffers for forward/backward passes.
/// Eliminates per-position heap allocations in the training loop.
struct FcScratch {
    /// Pre-ReLU activations per hidden layer: pre_relu[layer_idx][..layer_size].
    pre_relu: Vec<Vec<f32>>,
    /// Post-ReLU activations per hidden layer: post_relu[layer_idx][..layer_size].
    post_relu: Vec<Vec<f32>>,
    /// Scratch buffer for backward pass gradient w.r.t. current layer activations.
    /// Sized to the maximum hidden layer size.
    d_h: Vec<f32>,
    /// Scratch buffer for backward pass gradient propagated to previous layer.
    /// Sized to the maximum hidden layer size.
    d_prev: Vec<f32>,
    /// Gradient accumulator, sized to num_params. Reused across batches.
    grad: Vec<f32>,
    /// Cached output from the most recent forward pass.
    output: f32,
}

impl FcScratch {
    fn new(hidden_layers: &[usize], num_params: usize) -> Self {
        let max_hidden = *hidden_layers.iter().max().unwrap();
        let pre_relu = hidden_layers.iter().map(|&h| vec![0.0f32; h]).collect();
        let post_relu = hidden_layers.iter().map(|&h| vec![0.0f32; h]).collect();
        Self {
            pre_relu,
            post_relu,
            d_h: vec![0.0f32; max_hidden],
            d_prev: vec![0.0f32; max_hidden],
            grad: vec![0.0f32; num_params],
            output: 0.0,
        }
    }

    /// Zero the gradient buffer (call once per batch).
    #[inline]
    fn zero_grad(&mut self) {
        self.grad.iter_mut().for_each(|g| *g = 0.0);
    }
}

/// Perform forward pass through an arbitrary-depth network, saving intermediates
/// into pre-allocated scratch buffers.
///
/// Weight layout (same as GenericMlp):
///   For each hidden layer i:
///     [W_i: prev_size * cur_size] [b_i: cur_size]
///   Output layer:
///     [W_out: last_hidden] [b_out: 1]
fn forward_with_intermediates(
    weights: &[f32],
    input_size: usize,
    hidden_layers: &[usize],
    active_board: &[u16],
    bag: &[f32; BAG_FEATURES],
    scratch: &mut FcScratch,
) {
    let num_layers = hidden_layers.len();
    let mut offset = 0usize;

    // ── First hidden layer: sparse input accumulation ──
    let h1 = hidden_layers[0];
    let l1_w = &weights[offset..offset + input_size * h1];
    offset += input_size * h1;
    let l1_b = &weights[offset..offset + h1];
    offset += h1;

    let h_pre = &mut scratch.pre_relu[0];
    h_pre.copy_from_slice(l1_b);

    // Sparse board features (binary, value = 1.0)
    for &idx in active_board {
        let feat = idx as usize;
        debug_assert!(feat < BOARD_FEATURES, "board feature index {} >= {}", feat, BOARD_FEATURES);
        let col = &l1_w[feat * h1..(feat + 1) * h1];
        for j in 0..h1 {
            h_pre[j] += col[j];
        }
    }

    // Dense bag features (26 values starting at index 1080)
    for (i, &val) in bag.iter().enumerate() {
        if val != 0.0 {
            let feat = BOARD_FEATURES + i;
            let col = &l1_w[feat * h1..(feat + 1) * h1];
            for j in 0..h1 {
                h_pre[j] += col[j] * val;
            }
        }
    }

    // ReLU
    let h_post = &mut scratch.post_relu[0];
    for j in 0..h1 {
        h_post[j] = scratch.pre_relu[0][j].max(0.0);
    }

    // ── Subsequent hidden layers: dense ──
    for layer_idx in 1..num_layers {
        let prev_size = hidden_layers[layer_idx - 1];
        let cur_size = hidden_layers[layer_idx];
        let lw = &weights[offset..offset + prev_size * cur_size];
        offset += prev_size * cur_size;
        let lb = &weights[offset..offset + cur_size];
        offset += cur_size;

        let cur_pre = &mut scratch.pre_relu[layer_idx];
        cur_pre.copy_from_slice(lb);
        for i in 0..prev_size {
            let w_row = &lw[i * cur_size..(i + 1) * cur_size];
            let s = scratch.post_relu[layer_idx - 1][i];
            for j in 0..cur_size {
                cur_pre[j] += w_row[j] * s;
            }
        }

        // ReLU into post_relu
        let cur_post = &mut scratch.post_relu[layer_idx];
        for j in 0..cur_size {
            cur_post[j] = scratch.pre_relu[layer_idx][j].max(0.0);
        }
    }

    // ── Output layer: dot product + sigmoid ──
    let last_h = hidden_layers[num_layers - 1];
    let out_w = &weights[offset..offset + last_h];
    offset += last_h;
    let out_b = weights[offset];

    let last_post = &scratch.post_relu[num_layers - 1];
    let mut logit = out_b;
    for j in 0..last_h {
        logit += out_w[j] * last_post[j];
    }

    scratch.output = sigmoid(logit);
}

// ── Backward pass ──────────────────────────────────────────────────────────

/// Accumulate gradients for one sample into `scratch.grad`.
/// Uses pre-allocated d_h / d_prev buffers from scratch to avoid per-position allocations.
///
/// Layout of `grad` matches `weights` (same as GenericMlp):
///   For each hidden layer i:
///     [W_i: prev_size * cur_size] [b_i: cur_size]
///   Output layer:
///     [W_out: last_hidden] [b_out: 1]
fn backward(
    weights: &[f32],
    input_size: usize,
    hidden_layers: &[usize],
    active_board: &[u16],
    bag: &[f32; BAG_FEATURES],
    target: f32,
    sample_weight: f32,
    scratch: &mut FcScratch,
) {
    let num_layers = hidden_layers.len();

    // Precompute weight offsets for each layer
    // layer_offsets[i] = start of hidden layer i's weights in the flat vector
    let mut layer_offsets = Vec::with_capacity(num_layers + 1);
    {
        let mut off = 0usize;
        let mut prev = input_size;
        for &h in hidden_layers {
            layer_offsets.push(off);
            off += prev * h + h; // weights + bias
            prev = h;
        }
        layer_offsets.push(off); // output layer offset
    }

    let output = scratch.output;

    // MSE loss: L = (output - target)^2
    // dL/d_output = 2 * (output - target)
    // d_output/d_logit = output * (1 - output)  [sigmoid derivative]
    let d_logit = 2.0 * (output - target) * output * (1.0 - output) * sample_weight;

    // ── Output layer gradients ──
    let last_h = hidden_layers[num_layers - 1];
    let out_offset = layer_offsets[num_layers];

    for j in 0..last_h {
        scratch.grad[out_offset + j] += d_logit * scratch.post_relu[num_layers - 1][j];
    }
    scratch.grad[out_offset + last_h] += d_logit; // output bias

    // ── Backprop through hidden layers (last to first) ──
    // Bootstrap d_h with d_logit * W_out for the last hidden layer.
    let out_w = &weights[out_offset..out_offset + last_h];
    for j in 0..last_h {
        scratch.d_h[j] = d_logit * out_w[j];
    }

    // Track which buffer holds the current d_next via a flag.
    // d_h starts as d_next for the last hidden layer.
    // We apply ReLU derivative in-place, then propagate into the other buffer.
    let mut d_next_is_d_h = true;

    for layer_idx in (0..num_layers).rev() {
        let cur_size = hidden_layers[layer_idx];
        let prev_size = if layer_idx == 0 { input_size } else { hidden_layers[layer_idx - 1] };
        let off = layer_offsets[layer_idx];
        let bias_offset = off + prev_size * cur_size;

        // Apply ReLU derivative in-place on the d_next buffer: d[j] *= (pre_relu > 0 ? 1 : 0)
        {
            let d_cur = if d_next_is_d_h { &mut scratch.d_h } else { &mut scratch.d_prev };
            for j in 0..cur_size {
                if scratch.pre_relu[layer_idx][j] <= 0.0 {
                    d_cur[j] = 0.0;
                }
            }
        }

        if layer_idx == 0 {
            // First hidden layer: SPARSE weight update, no propagation needed
            let h1 = cur_size;
            let d_cur = if d_next_is_d_h { &scratch.d_h } else { &scratch.d_prev };
            for j in 0..h1 {
                let dj = d_cur[j];
                if dj == 0.0 { continue; }

                scratch.grad[bias_offset + j] += dj;

                for &idx in active_board {
                    let feat = idx as usize;
                    scratch.grad[off + feat * h1 + j] += dj;
                }

                for (i, &val) in bag.iter().enumerate() {
                    if val != 0.0 {
                        let feat = BOARD_FEATURES + i;
                        scratch.grad[off + feat * h1 + j] += dj * val;
                    }
                }
            }
        } else {
            // Dense hidden layers: full weight update + propagate gradient
            let lw = &weights[off..off + prev_size * cur_size];

            // Bias gradients
            {
                let d_cur = if d_next_is_d_h { &scratch.d_h } else { &scratch.d_prev };
                for j in 0..cur_size {
                    scratch.grad[bias_offset + j] += d_cur[j];
                }
            }

            // Propagate gradient to previous layer into the OTHER buffer
            {
                let (d_cur, d_out) = if d_next_is_d_h {
                    (&scratch.d_h as &Vec<f32>, &mut scratch.d_prev)
                } else {
                    (&scratch.d_prev as &Vec<f32>, &mut scratch.d_h)
                };
                for i in 0..prev_size {
                    let w_row = &lw[i * cur_size..(i + 1) * cur_size];
                    let mut sum = 0.0f32;
                    for j in 0..cur_size {
                        sum += w_row[j] * d_cur[j];
                    }
                    d_out[i] = sum;
                }
            }

            // Weight gradients: d_W[i,j] = d_h[j] * prev_post[i]
            {
                let d_cur = if d_next_is_d_h { &scratch.d_h } else { &scratch.d_prev };
                let prev_post = &scratch.post_relu[layer_idx - 1];
                for i in 0..prev_size {
                    let s = prev_post[i];
                    for j in 0..cur_size {
                        scratch.grad[off + i * cur_size + j] += d_cur[j] * s;
                    }
                }
            }

            // Swap: the "other" buffer now holds d_next for the next (earlier) layer
            d_next_is_d_h = !d_next_is_d_h;
        }
    }
}

// ── Adam optimizer ─────────────────────────────────────────────────────────

struct AdamState {
    m: Vec<f32>,  // first moment
    v: Vec<f32>,  // second moment
    t: u64,       // time step
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
}

impl AdamState {
    fn new(num_params: usize, lr: f32) -> Self {
        Self {
            m: vec![0.0; num_params],
            v: vec![0.0; num_params],
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// Perform one Adam update step. `grad` is the mean gradient over the batch.
    fn step(&mut self, weights: &mut [f32], grad: &[f32]) {
        self.t += 1;
        let t = self.t as f32;
        let lr_t = self.lr * (1.0 - self.beta2.powf(t)).sqrt() / (1.0 - self.beta1.powf(t));

        for i in 0..weights.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
            weights[i] -= lr_t * self.m[i] / (self.v[i].sqrt() + self.eps);
        }
    }
}

// ── Evaluation ─────────────────────────────────────────────────────────────

/// Evaluate the current network against benchmark opponents.
fn evaluate_model(
    net: &GenericMlp,
    benchmark_specs: &[String],
    eval_games: u32,
) {
    let bag = create_bag();
    let gs = create_initial_state(&bag);

    let model_eval = GenericEvaluator { net: GenericMlp::from_flat(
        net.weights.clone(),
        net.input_size,
        net.hidden_layers.clone(),
    )};
    let model_player = Player::Evaluator(&model_eval);

    for spec in benchmark_specs {
        let opponent = LoadedModel::from_spec(spec, false);
        let opp_player = opponent.as_player();
        let label = &opponent.label;
        let result = run_matches(
            &gs,
            &model_player,
            &opp_player,
            eval_games,
            &format!("vs {}", label),
        );
        let wr = win_rate(result.player_a_wins, result.ties, eval_games);
        eprintln!(
            "  vs {}: {:.1}% win rate ({} W / {} L / {} T)",
            label,
            wr * 100.0,
            result.player_a_wins,
            result.player_b_wins,
            result.ties,
        );
    }
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let input_path: String = parse_flag(&args, "--input")
        .unwrap_or_else(|| {
            eprintln!(
                "Usage: supervised_train --input <path> [--hidden 256,128,64] [--lr 0.001] \
                 [--epochs 10] [--batch-size 256] \
                 [--eval-interval 50000] [--eval-games 500] [--benchmark base,random] \
                 [--checkpoint-dir D:/temp/supervised_nn] [--seed 42]"
            );
            std::process::exit(1);
        });

    let hidden_str: String = parse_flag(&args, "--hidden").unwrap_or_else(|| "128".to_string());
    let hidden_layers: Vec<usize> = hidden_str
        .split(',')
        .map(|s| s.trim().parse::<usize>().expect("--hidden values must be comma-separated integers"))
        .collect();
    assert!(!hidden_layers.is_empty(), "--hidden must specify at least one layer size");
    for &h in &hidden_layers {
        assert!(h > 0 && h <= MAX_HIDDEN, "hidden layer size {} must be in 1..={}", h, MAX_HIDDEN);
    }
    let lr: f32 = parse_flag(&args, "--lr").unwrap_or(0.001);
    let epochs: usize = parse_flag(&args, "--epochs").unwrap_or(10);
    let batch_size: usize = parse_flag(&args, "--batch-size").unwrap_or(256);
    let eval_interval: usize = parse_flag(&args, "--eval-interval").unwrap_or(50000);
    let eval_games: u32 = parse_flag(&args, "--eval-games").unwrap_or(500);
    let checkpoint_dir: String = parse_flag(&args, "--checkpoint-dir")
        .unwrap_or_else(|| "D:/temp/supervised_nn".to_string());
    let seed: u64 = parse_flag(&args, "--seed").unwrap_or(42);

    let benchmark_str: String = parse_flag(&args, "--benchmark")
        .unwrap_or_else(|| "base,random".to_string());
    let benchmark_specs: Vec<String> = benchmark_str.split(',').map(|s| s.trim().to_string()).collect();

    // Print configuration
    eprintln!("=== Supervised Training ===");
    // Load data first so we know label range
    let (positions, _label_min, _label_max) = load_lpos(&input_path);

    eprintln!("  Input:          {}", input_path);
    let arch_str = std::iter::once(TOTAL_FEATURES.to_string())
        .chain(hidden_layers.iter().map(|h| h.to_string()))
        .chain(std::iter::once("1".to_string()))
        .collect::<Vec<_>>()
        .join(" -> ");
    eprintln!("  Architecture:   {}", arch_str);
    eprintln!("  Learning rate:  {}", lr);
    eprintln!("  Epochs:         {}", epochs);
    eprintln!("  Batch size:     {}", batch_size);
    eprintln!("  Label mapping:  clamp to +-{}, linear to (0..1)", LABEL_CLAMP);
    eprintln!("  Eval interval:  {} positions", eval_interval);
    eprintln!("  Eval games:     {}", eval_games);
    eprintln!("  Benchmark:      {:?}", benchmark_specs);
    eprintln!("  Checkpoint dir: {}", checkpoint_dir);
    eprintln!("  Seed:           {}", seed);
    eprintln!();

    // Create checkpoint directory
    std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoint directory");
    let num_positions = positions.len();

    // Build expanded index array weighted by count.
    // Each position index is repeated `count` times, then we shuffle this array.
    eprintln!("Building weighted index array ...");
    let t_idx = Instant::now();
    let total_weighted: usize = positions.iter().map(|p| p.count as usize).sum();

    // If total_weighted is huge (e.g. 50M+), cap counts to avoid excessive memory.
    // Use the index array approach only if it fits in reasonable memory (~200M entries).
    let (indices, effective_total) = if total_weighted <= 200_000_000 {
        let mut indices: Vec<u32> = Vec::with_capacity(total_weighted);
        for (i, pos) in positions.iter().enumerate() {
            for _ in 0..pos.count {
                indices.push(i as u32);
            }
        }
        let len = indices.len();
        (indices, len)
    } else {
        // Too many -- just use each position once, weighted by sqrt(count) to
        // down-weight extremely frequent positions while still sampling more from common ones.
        eprintln!(
            "  Total weighted count {} exceeds 200M, using sqrt-weighted sampling",
            total_weighted
        );
        let mut indices: Vec<u32> = Vec::new();
        for (i, pos) in positions.iter().enumerate() {
            let repeats = (pos.count as f64).sqrt().ceil() as u32;
            for _ in 0..repeats {
                indices.push(i as u32);
            }
        }
        let len = indices.len();
        (indices, len)
    };
    eprintln!(
        "  {} training samples (from {} unique positions) in {:.1}s",
        effective_total,
        num_positions,
        t_idx.elapsed().as_secs_f64()
    );

    // Initialize network
    let mut rng = StdRng::seed_from_u64(seed);
    let input_size = TOTAL_FEATURES;
    let num_params = GenericMlp::param_count(input_size, &hidden_layers);
    eprintln!("Network: {} parameters", num_params);

    let mut net = GenericMlp::random(input_size, hidden_layers.clone(), &mut rng);
    let mut adam = AdamState::new(num_params, lr);
    let mut scratch = FcScratch::new(&hidden_layers, num_params);

    // Initial evaluation
    eprintln!("\n--- Initial evaluation ---");
    evaluate_model(&net, &benchmark_specs, eval_games);

    // Training loop
    let mut total_samples = 0usize;
    let mut shuffled_indices = indices.clone();

    // Adaptive learning rate state
    let mut best_loss: f64 = f64::INFINITY;
    let mut batches_since_improvement: usize = 0;
    let mut recent_loss_sum: f64 = 0.0;
    let mut recent_loss_count: usize = 0;
    let lr_check_interval: usize = 1000;
    let lr_stall_threshold: usize = 5000;
    let lr_min: f32 = 1e-6;
    let lr_max: f32 = 0.1;

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        eprintln!("\n=== Epoch {}/{} ===", epoch + 1, epochs);

        // Shuffle
        shuffled_indices.shuffle(&mut rng);

        let num_batches = (effective_total + batch_size - 1) / batch_size;
        let mut epoch_loss = 0.0f64;
        let mut epoch_samples = 0usize;
        let mut last_eval_at = 0usize;

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;
            let batch_end = (batch_start + batch_size).min(effective_total);
            let actual_batch_size = batch_end - batch_start;
            let inv_batch = 1.0f32 / actual_batch_size as f32;

            // Zero gradient accumulator (reused across batches)
            scratch.zero_grad();
            let mut batch_loss = 0.0f64;

            for si in batch_start..batch_end {
                let pos_idx = shuffled_indices[si] as usize;
                let pos = &positions[pos_idx];
                let target = label_to_target(pos.label);

                forward_with_intermediates(
                    &net.weights,
                    input_size,
                    &hidden_layers,
                    &pos.active_indices,
                    &pos.bag_features,
                    &mut scratch,
                );

                let error = scratch.output - target;
                batch_loss += (error * error) as f64;

                backward(
                    &net.weights,
                    input_size,
                    &hidden_layers,
                    &pos.active_indices,
                    &pos.bag_features,
                    target,
                    inv_batch,
                    &mut scratch,
                );
            }

            // Adam update
            adam.step(&mut net.weights, &scratch.grad);

            epoch_loss += batch_loss;
            epoch_samples += actual_batch_size;
            total_samples += actual_batch_size;

            // Adaptive learning rate
            let avg_batch_loss = batch_loss / actual_batch_size as f64;
            recent_loss_sum += avg_batch_loss;
            recent_loss_count += 1;

            if recent_loss_count >= lr_check_interval {
                let recent_avg = recent_loss_sum / recent_loss_count as f64;
                if recent_avg < best_loss {
                    // Improved
                    best_loss = recent_avg;
                    batches_since_improvement = 0;
                } else {
                    batches_since_improvement += recent_loss_count;

                    if recent_avg > best_loss * 1.05 {
                        // Loss increased significantly: slow down
                        let old_lr = adam.lr;
                        adam.lr = (adam.lr * 0.5).max(lr_min);
                        if adam.lr != old_lr {
                            eprintln!(
                                "  LR adjusted: {} -> {} (loss increased)",
                                old_lr, adam.lr
                            );
                        }
                        batches_since_improvement = 0;
                    } else if batches_since_improvement >= lr_stall_threshold {
                        // Loss stalled: speed up
                        let old_lr = adam.lr;
                        adam.lr = (adam.lr * 1.5).min(lr_max);
                        if adam.lr != old_lr {
                            eprintln!(
                                "  LR adjusted: {} -> {} (loss stalled)",
                                old_lr, adam.lr
                            );
                        }
                        batches_since_improvement = 0;
                    }
                }
                recent_loss_sum = 0.0;
                recent_loss_count = 0;
            }

            // Progress logging
            if (batch_idx + 1) % 1000 == 0 || batch_idx + 1 == num_batches {
                let avg_loss = epoch_loss / epoch_samples as f64;
                let elapsed = epoch_start.elapsed().as_secs_f64();
                let rate = epoch_samples as f64 / elapsed;
                eprintln!(
                    "  Batch {}/{}: loss={:.6}, samples={}, rate={:.0} pos/s",
                    batch_idx + 1,
                    num_batches,
                    avg_loss,
                    epoch_samples,
                    rate,
                );
            }

            // Periodic evaluation
            if eval_interval > 0 && (epoch_samples - last_eval_at) >= eval_interval {
                last_eval_at = epoch_samples;
                eprintln!("\n  --- Evaluation at {} samples (epoch {}) ---", epoch_samples, epoch + 1);
                evaluate_model(&net, &benchmark_specs, eval_games);
                eprintln!();
            }
        }

        let epoch_avg_loss = epoch_loss / epoch_samples as f64;
        let epoch_elapsed = epoch_start.elapsed();
        eprintln!(
            "Epoch {} complete: avg_loss={:.6}, {:.1}s ({:.0} pos/s)",
            epoch + 1,
            epoch_avg_loss,
            epoch_elapsed.as_secs_f64(),
            epoch_samples as f64 / epoch_elapsed.as_secs_f64(),
        );

        // Save checkpoint
        let ckpt_path = format!("{}/epoch_{}.gmlp", checkpoint_dir, epoch + 1);
        net.save(&ckpt_path).expect("Failed to save checkpoint");
        eprintln!("  Saved checkpoint: {}", ckpt_path);

        // End-of-epoch evaluation
        eprintln!("  --- End-of-epoch evaluation ---");
        evaluate_model(&net, &benchmark_specs, eval_games);
    }

    // Save final model
    let final_path = format!("{}/final.gmlp", checkpoint_dir);
    net.save(&final_path).expect("Failed to save final model");
    eprintln!("\n=== Training complete ===");
    eprintln!("Final model saved to: {}", final_path);
    eprintln!("Total samples processed: {}", total_samples);

    // Final evaluation
    eprintln!("\n--- Final evaluation ---");
    evaluate_model(&net, &benchmark_specs, eval_games);
}
