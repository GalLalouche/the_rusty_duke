//! Supervised training binary: train a pure NN (1106->H->1) on labeled positions.
//!
//! Usage:
//!   supervised_train --input D:/temp/labeled_positions.bin \
//!                    --hidden 128 \
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
use duke_training::game_setup::{create_bag, create_initial_state, StaticHeuristicEvaluator};
use duke_training::generic_mlp::{GenericMlp, GenericNnueEvaluator, MAX_HIDDEN};
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
fn load_lpos(path: &str) -> Vec<LabeledPosition> {
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

    positions
}

// ── Sigmoid / target mapping ───────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Map a raw label to a [0, 1] training target via sigmoid(label / scale).
#[inline]
fn label_to_target(label: f32, scale: f32) -> f32 {
    sigmoid(label / scale)
}

// ── Forward pass with intermediates ────────────────────────────────────────

/// Forward pass result containing intermediate values needed for backpropagation.
/// Only supports single-hidden-layer networks (1106 -> H -> 1).
struct ForwardResult {
    /// Pre-ReLU hidden activations.
    h_pre: [f32; MAX_HIDDEN],
    /// Post-ReLU hidden activations.
    h: [f32; MAX_HIDDEN],
    /// Sigmoid output.
    output: f32,
}

/// Perform forward pass on a single-hidden-layer network, saving intermediates.
///
/// The network layout in `weights` is:
///   [L1 weights: input_size * h1] [L1 bias: h1] [L2 weights: h1] [L2 bias: 1]
fn forward_with_intermediates(
    weights: &[f32],
    input_size: usize,
    h1: usize,
    active_board: &[u16],
    bag: &[f32; BAG_FEATURES],
) -> ForwardResult {
    let l1_w = &weights[0..input_size * h1];
    let l1_b = &weights[input_size * h1..input_size * h1 + h1];

    let mut h_pre = [0.0f32; MAX_HIDDEN];
    // Initialize with bias
    h_pre[..h1].copy_from_slice(l1_b);

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
    let mut h = [0.0f32; MAX_HIDDEN];
    for j in 0..h1 {
        h[j] = h_pre[j].max(0.0);
    }

    // Output layer: logit = W2 * h + b2
    let w2_offset = input_size * h1 + h1;
    let w2 = &weights[w2_offset..w2_offset + h1];
    let b2 = weights[w2_offset + h1];

    let mut logit = b2;
    for j in 0..h1 {
        logit += w2[j] * h[j];
    }

    let output = sigmoid(logit);

    ForwardResult { h_pre, h, output }
}

// ── Backward pass ──────────────────────────────────────────────────────────

/// Accumulate gradients for one sample into `grad`.
///
/// Layout of `grad` matches `weights`:
///   [L1 weights: input_size * h1] [L1 bias: h1] [L2 weights: h1] [L2 bias: 1]
fn backward(
    weights: &[f32],
    input_size: usize,
    h1: usize,
    active_board: &[u16],
    bag: &[f32; BAG_FEATURES],
    fwd: &ForwardResult,
    target: f32,
    sample_weight: f32,
    grad: &mut [f32],
) {
    // MSE loss: L = (output - target)^2
    // dL/d_output = 2 * (output - target)
    // d_output/d_logit = output * (1 - output)  [sigmoid derivative]
    let d_logit = 2.0 * (fwd.output - target) * fwd.output * (1.0 - fwd.output) * sample_weight;

    // Output layer: logit = W2 * h + b2
    let w2_offset = input_size * h1 + h1;
    let w2 = &weights[w2_offset..w2_offset + h1];

    // Gradients for W2 and b2
    for j in 0..h1 {
        grad[w2_offset + j] += d_logit * fwd.h[j];
    }
    grad[w2_offset + h1] += d_logit; // bias

    // Backprop through ReLU to L1
    // d_logit/d_h[j] = W2[j]
    // d_h/d_h_pre[j] = 1 if h_pre[j] > 0, else 0 (ReLU derivative)
    let l1_bias_offset = input_size * h1;

    for j in 0..h1 {
        if fwd.h_pre[j] <= 0.0 {
            continue; // ReLU killed this neuron
        }
        let d_h_j = d_logit * w2[j];

        // L1 bias gradient
        grad[l1_bias_offset + j] += d_h_j;

        // L1 weight gradients for active board features (sparse, value = 1.0)
        for &idx in active_board {
            let feat = idx as usize;
            grad[feat * h1 + j] += d_h_j;
        }

        // L1 weight gradients for bag features (dense)
        for (i, &val) in bag.iter().enumerate() {
            if val != 0.0 {
                let feat = BOARD_FEATURES + i;
                grad[feat * h1 + j] += d_h_j * val;
            }
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

    let model_eval = GenericNnueEvaluator { net: GenericMlp::from_flat(
        net.weights.clone(),
        net.input_size,
        net.hidden_layers.clone(),
    )};
    let model_player = Player::Evaluator(&model_eval);

    for spec in benchmark_specs {
        match spec.as_str() {
            "random" => {
                let result = run_matches(
                    &gs,
                    &model_player,
                    &Player::Random,
                    eval_games,
                    "vs Random",
                );
                let wr = win_rate(result.player_a_wins, result.ties, eval_games);
                eprintln!(
                    "  vs Random: {:.1}% win rate ({} W / {} L / {} T)",
                    wr * 100.0,
                    result.player_a_wins,
                    result.player_b_wins,
                    result.ties,
                );
            }
            "base" => {
                let base_eval = StaticHeuristicEvaluator::new();
                let base_player = Player::Evaluator(&base_eval);
                let result = run_matches(
                    &gs,
                    &model_player,
                    &base_player,
                    eval_games,
                    "vs Base",
                );
                let wr = win_rate(result.player_a_wins, result.ties, eval_games);
                eprintln!(
                    "  vs Base: {:.1}% win rate ({} W / {} L / {} T)",
                    wr * 100.0,
                    result.player_a_wins,
                    result.player_b_wins,
                    result.ties,
                );
            }
            _ => {
                eprintln!("  Unknown benchmark spec '{}', skipping", spec);
            }
        }
    }
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let input_path: String = parse_flag(&args, "--input")
        .unwrap_or_else(|| {
            eprintln!(
                "Usage: supervised_train --input <path> [--hidden 128] [--lr 0.001] \
                 [--epochs 10] [--batch-size 256] [--label-scale 100] \
                 [--eval-interval 50000] [--eval-games 500] [--benchmark base,random] \
                 [--checkpoint-dir D:/temp/supervised_nn] [--seed 42]"
            );
            std::process::exit(1);
        });

    let hidden_size: usize = parse_flag(&args, "--hidden").unwrap_or(128);
    let lr: f32 = parse_flag(&args, "--lr").unwrap_or(0.001);
    let epochs: usize = parse_flag(&args, "--epochs").unwrap_or(10);
    let batch_size: usize = parse_flag(&args, "--batch-size").unwrap_or(256);
    let label_scale: f32 = parse_flag(&args, "--label-scale").unwrap_or(100.0);
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
    eprintln!("  Input:          {}", input_path);
    eprintln!("  Architecture:   {} -> {} -> 1", TOTAL_FEATURES, hidden_size);
    eprintln!("  Learning rate:  {}", lr);
    eprintln!("  Epochs:         {}", epochs);
    eprintln!("  Batch size:     {}", batch_size);
    eprintln!("  Label scale:    {} (target = sigmoid(label / scale))", label_scale);
    eprintln!("  Eval interval:  {} positions", eval_interval);
    eprintln!("  Eval games:     {}", eval_games);
    eprintln!("  Benchmark:      {:?}", benchmark_specs);
    eprintln!("  Checkpoint dir: {}", checkpoint_dir);
    eprintln!("  Seed:           {}", seed);
    eprintln!();

    // Create checkpoint directory
    std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoint directory");

    // Load data
    let positions = load_lpos(&input_path);
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
    let hidden_layers = vec![hidden_size];
    let num_params = GenericMlp::param_count(input_size, &hidden_layers);
    eprintln!("Network: {} parameters", num_params);

    let mut net = GenericMlp::random(input_size, hidden_layers.clone(), &mut rng);
    let mut adam = AdamState::new(num_params, lr);

    // Initial evaluation
    eprintln!("\n--- Initial evaluation ---");
    evaluate_model(&net, &benchmark_specs, eval_games);

    // Training loop
    let mut total_samples = 0usize;
    let mut shuffled_indices = indices.clone();

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

            // Accumulate gradients
            let mut grad = vec![0.0f32; num_params];
            let mut batch_loss = 0.0f64;

            for si in batch_start..batch_end {
                let pos_idx = shuffled_indices[si] as usize;
                let pos = &positions[pos_idx];
                let target = label_to_target(pos.label, label_scale);

                let fwd = forward_with_intermediates(
                    &net.weights,
                    input_size,
                    hidden_size,
                    &pos.active_indices,
                    &pos.bag_features,
                );

                let error = fwd.output - target;
                batch_loss += (error * error) as f64;

                backward(
                    &net.weights,
                    input_size,
                    hidden_size,
                    &pos.active_indices,
                    &pos.bag_features,
                    &fwd,
                    target,
                    inv_batch,
                    &mut grad,
                );
            }

            // Adam update
            adam.step(&mut net.weights, &grad);

            epoch_loss += batch_loss;
            epoch_samples += actual_batch_size;
            total_samples += actual_batch_size;

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
