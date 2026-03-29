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

use rand::Rng;
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

/// Label names for the 41 combined features (for display when loading FLPS).
const COMBINED_FEATURE_NAMES: [&str; 41] = [
    "near_my_duke_friendly", "near_my_duke_enemy",
    "near_enemy_duke_friendly", "near_enemy_duke_enemy",
    "my_moves", "opp_moves", "my_reachable", "opp_reachable", "contested",
    "my_defended", "my_threatened", "opp_defended", "opp_threatened",
    "my_duke_mob", "opp_duke_mob",
    "my_duke_disc", "my_footman_disc", "my_pikeman_disc", "my_knight_disc",
    "my_sergeant_disc", "my_ranger_disc", "my_champion_disc", "my_wizard_disc",
    "my_general_disc", "my_marshall_disc", "my_assassin_disc", "my_longbowman_disc",
    "my_dragoon_disc",
    "opp_duke_disc", "opp_footman_disc", "opp_pikeman_disc", "opp_knight_disc",
    "opp_sergeant_disc", "opp_ranger_disc", "opp_champion_disc", "opp_wizard_disc",
    "opp_general_disc", "opp_marshall_disc", "opp_assassin_disc", "opp_longbowman_disc",
    "opp_dragoon_disc",
];

/// Load labeled positions from an LPOS or FLPS binary file.
/// For FLPS files, `label_index` selects which of the N labels to use.
/// Returns (positions, min_label, max_label).
fn load_lpos(path: &str, label_index: usize) -> (Vec<LabeledPosition>, f32, f32) {
    let t0 = Instant::now();
    eprintln!("Loading labeled positions from {} ...", path);

    let data = std::fs::read(path).expect("Failed to read labeled positions file");
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

    // Detect format by magic bytes
    let magic = read_bytes!(4);
    let is_flps = magic == b"FLPS";
    let is_lpos = magic == b"LPOS";
    assert!(is_lpos || is_flps,
        "Unknown file format (magic: {:?}), expected LPOS or FLPS", magic);

    let version = read_u32!();
    assert_eq!(version, 1, "Unsupported version {}", version);
    let num_positions = read_u32!() as usize;

    let num_labels = if is_flps {
        let nl = read_u32!() as usize;
        assert!(label_index < nl,
            "--label-index {} out of range (file has {} labels)", label_index, nl);
        let label_name = if nl == 41 && label_index < COMBINED_FEATURE_NAMES.len() {
            COMBINED_FEATURE_NAMES[label_index]
        } else {
            "unknown"
        };
        eprintln!("  FLPS format: {} positions, {} labels, using label index {} ({})",
            num_positions, nl, label_index, label_name);
        nl
    } else {
        if label_index != 0 {
            eprintln!("  Warning: --label-index {} ignored for LPOS format (single label)", label_index);
        }
        eprintln!("  LPOS format: {} positions, version {}", num_positions, version);
        1
    };

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

        let label = if is_flps {
            // Read all labels, pick the one at label_index
            let mut selected = 0.0f32;
            for li in 0..num_labels {
                let val = read_f32!();
                if li == label_index {
                    selected = val;
                }
            }
            selected
        } else {
            read_f32!()
        };

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

/// Format a loss value: scientific notation if < 1e-5, otherwise 6 decimal places.
fn fmt_loss(v: f64) -> String {
    if v.abs() < 1e-5 { format!("{:.3e}", v) } else { format!("{:.6}", v) }
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

// ── Batched forward + backward with sgemm ─────────────────────────────────

use matrixmultiply::sgemm;

/// Pre-allocated scratch buffers for batched forward/backward passes.
/// All activation/gradient matrices are row-major: [batch_size × neurons].
struct FcScratch {
    /// Batch pre-ReLU activations per hidden layer: [batch_size × layer_size].
    pre_act: Vec<Vec<f32>>,
    /// Batch post-ReLU activations per hidden layer: [batch_size × layer_size].
    act: Vec<Vec<f32>>,
    /// Batch gradient matrices per hidden layer: [batch_size × layer_size].
    d_act: Vec<Vec<f32>>,
    /// Gradient accumulator, sized to num_params. Reused across batches.
    grad: Vec<f32>,
    /// Batch outputs (sigmoid): [batch_size].
    outputs: Vec<f32>,
    /// Batch output logits (pre-sigmoid): [batch_size].
    logits: Vec<f32>,
    /// Batch d_logit: [batch_size × 1] used as matrix for output layer backward.
    d_logits: Vec<f32>,
    /// Precomputed weight offsets for each layer (including output layer at the end).
    layer_offsets: Vec<usize>,
}

impl FcScratch {
    fn new(hidden_layers: &[usize], num_params: usize, max_batch: usize, input_size: usize) -> Self {
        let pre_act = hidden_layers.iter().map(|&h| vec![0.0f32; max_batch * h]).collect();
        let act = hidden_layers.iter().map(|&h| vec![0.0f32; max_batch * h]).collect();
        let d_act = hidden_layers.iter().map(|&h| vec![0.0f32; max_batch * h]).collect();

        // Precompute weight offsets
        let mut layer_offsets = Vec::with_capacity(hidden_layers.len() + 1);
        let mut off = 0usize;
        let mut prev = input_size;
        for &h in hidden_layers {
            layer_offsets.push(off);
            off += prev * h + h;
            prev = h;
        }
        layer_offsets.push(off); // output layer offset

        Self {
            pre_act,
            act,
            d_act,
            grad: vec![0.0f32; num_params],
            outputs: vec![0.0f32; max_batch],
            logits: vec![0.0f32; max_batch],
            d_logits: vec![0.0f32; max_batch],
            layer_offsets,
        }
    }

    /// Zero the gradient buffer (call once per batch).
    #[inline]
    fn zero_grad(&mut self) {
        self.grad.iter_mut().for_each(|g| *g = 0.0);
    }
}

/// Batched forward pass: process `batch_size` positions at once.
///
/// L1 (sparse): per-position accumulation into act[0] rows.
/// Subsequent layers: single sgemm call per layer.
/// Output: sgemm for dot product, then per-element sigmoid.
///
/// Weight layout (same as GenericMlp):
///   For each hidden layer i:
///     [W_i: prev_size * cur_size] [b_i: cur_size]
///   Output layer:
///     [W_out: last_hidden] [b_out: 1]
fn batch_forward(
    weights: &[f32],
    input_size: usize,
    hidden_layers: &[usize],
    positions: &[&LabeledPosition],
    scratch: &mut FcScratch,
    residual_interval: usize,
) {
    let batch_size = positions.len();
    let num_layers = hidden_layers.len();

    // ── L1 (sparse): per-position accumulation ──
    let h1 = hidden_layers[0];
    let l1_off = scratch.layer_offsets[0];
    let l1_w = &weights[l1_off..l1_off + input_size * h1];
    let l1_b = &weights[l1_off + input_size * h1..l1_off + input_size * h1 + h1];

    for (bi, pos) in positions.iter().enumerate() {
        let row = &mut scratch.pre_act[0][bi * h1..(bi + 1) * h1];
        row.copy_from_slice(l1_b);

        // Sparse board features (binary, value = 1.0)
        for &idx in &pos.active_indices {
            let feat = idx as usize;
            let col_start = feat * h1;
            let col = &l1_w[col_start..col_start + h1];
            // Ensure the compiler sees matching lengths for auto-vectorization
            let row_slice = &mut row[..h1];
            for j in 0..h1 {
                row_slice[j] += col[j];
            }
        }

        // Dense bag features
        for (i, &val) in pos.bag_features.iter().enumerate() {
            if val != 0.0 {
                let feat = BOARD_FEATURES + i;
                let col_start = feat * h1;
                let col = &l1_w[col_start..col_start + h1];
                let row_slice = &mut row[..h1];
                for j in 0..h1 {
                    row_slice[j] += col[j] * val;
                }
            }
        }
    }

    // ReLU for L1
    let pre0 = &scratch.pre_act[0];
    let act0 = &mut scratch.act[0];
    for i in 0..batch_size * h1 {
        act0[i] = pre0[i].max(0.0);
    }

    // ── Subsequent hidden layers: batched sgemm ──
    for layer_idx in 1..num_layers {
        let prev_size = hidden_layers[layer_idx - 1];
        let cur_size = hidden_layers[layer_idx];
        let off = scratch.layer_offsets[layer_idx];
        let lw = &weights[off..off + prev_size * cur_size];
        let lb = &weights[off + prev_size * cur_size..off + prev_size * cur_size + cur_size];

        // pre_act[layer] = act[layer-1] × W + bias (broadcast)
        // act[layer-1]: [batch_size × prev_size], W: [prev_size × cur_size]
        // result: [batch_size × cur_size]
        let pre = &mut scratch.pre_act[layer_idx];

        unsafe {
            sgemm(
                batch_size,                         // m
                prev_size,                          // k
                cur_size,                           // n
                1.0,                                // alpha
                scratch.act[layer_idx - 1].as_ptr(), // A
                prev_size as isize,                 // rsa
                1,                                  // csa
                lw.as_ptr(),                        // B
                cur_size as isize,                  // rsb
                1,                                  // csb
                0.0,                                // beta
                pre.as_mut_ptr(),                   // C
                cur_size as isize,                  // rsc
                1,                                  // csc
            );
        }

        // Bias
        for bi in 0..batch_size {
            let row_pre = &mut pre[bi * cur_size..(bi + 1) * cur_size];
            for j in 0..cur_size {
                row_pre[j] += lb[j];
            }
        }

        // Residual (skip) connection: add activation from N layers back
        if residual_interval > 0
            && layer_idx >= residual_interval
            && hidden_layers[layer_idx] == hidden_layers[layer_idx - residual_interval]
        {
            let src_layer = layer_idx - residual_interval;
            let src_act = &scratch.act[src_layer];
            for i in 0..batch_size * cur_size {
                pre[i] += src_act[i];
            }
        }

        // ReLU
        let act_l = &mut scratch.act[layer_idx];
        for i in 0..batch_size * cur_size {
            act_l[i] = pre[i].max(0.0);
        }
    }

    // ── Output layer: [batch_size × last_hidden] × [last_hidden × 1] ──
    let last_h = hidden_layers[num_layers - 1];
    let out_off = scratch.layer_offsets[num_layers];
    let out_w = &weights[out_off..out_off + last_h];
    let out_b = weights[out_off + last_h];

    // Manual dot product per position (last_hidden × 1 -- sgemm overhead not worth it)
    let last_act = &scratch.act[num_layers - 1];
    for bi in 0..batch_size {
        let row = &last_act[bi * last_h..(bi + 1) * last_h];
        let mut logit = out_b;
        for j in 0..last_h {
            logit += row[j] * out_w[j];
        }
        scratch.logits[bi] = logit;
        scratch.outputs[bi] = sigmoid(logit);
    }
}

/// Batched backward pass: compute gradients for the entire batch at once.
///
/// Dense FC layers use sgemm for weight gradients and input gradients.
/// L1 sparse layer uses per-position sparse accumulation.
fn batch_backward(
    weights: &[f32],
    input_size: usize,
    hidden_layers: &[usize],
    positions: &[&LabeledPosition],
    targets: &[f32],
    inv_batch: f32,
    scratch: &mut FcScratch,
    residual_interval: usize,
) {
    let batch_size = positions.len();
    let num_layers = hidden_layers.len();
    let last_h = hidden_layers[num_layers - 1];
    let out_off = scratch.layer_offsets[num_layers];

    // ── Output gradient: d_logit for all positions ──
    for bi in 0..batch_size {
        let o = scratch.outputs[bi];
        let d = 2.0 * (o - targets[bi]) * o * (1.0 - o) * inv_batch;
        scratch.d_logits[bi] = d;
    }

    // ── Output layer weight gradients ──
    // d_W_out[j] = sum_i(d_logit[i] * act_last[i, j])
    // d_b_out = sum_i(d_logit[i])
    let last_act = &scratch.act[num_layers - 1];
    let grad_out = &mut scratch.grad[out_off..out_off + last_h + 1];
    for bi in 0..batch_size {
        let d = scratch.d_logits[bi];
        let row = &last_act[bi * last_h..(bi + 1) * last_h];
        for j in 0..last_h {
            grad_out[j] += d * row[j];
        }
        grad_out[last_h] += d; // bias
    }

    // ── Backprop d_logit to last hidden layer ──
    // d_act_last[i, j] = d_logit[i] * W_out[j]
    let out_w = &weights[out_off..out_off + last_h];
    let d_last = &mut scratch.d_act[num_layers - 1];
    for bi in 0..batch_size {
        let d = scratch.d_logits[bi];
        let row = &mut d_last[bi * last_h..(bi + 1) * last_h];
        for j in 0..last_h {
            row[j] = d * out_w[j];
        }
    }

    // Apply ReLU mask for last hidden layer
    let pre_last = &scratch.pre_act[num_layers - 1];
    for i in 0..batch_size * last_h {
        if pre_last[i] <= 0.0 {
            d_last[i] = 0.0;
        }
    }

    // ── FC hidden layers (last to first) ──
    for layer_idx in (1..num_layers).rev() {
        let prev_size = hidden_layers[layer_idx - 1];
        let cur_size = hidden_layers[layer_idx];
        let off = scratch.layer_offsets[layer_idx];
        let bias_offset = off + prev_size * cur_size;
        let lw = &weights[off..off + prev_size * cur_size];

        // Split d_act to get both d_cur and d_prev without borrow conflict
        let (d_lower, d_upper) = scratch.d_act.split_at_mut(layer_idx);
        let d_cur = &d_upper[0]; // d_act[layer_idx]
        let d_prev = &mut d_lower[layer_idx - 1]; // d_act[layer_idx - 1]

        // Weight gradient: d_W = act[layer-1]^T × d_act[layer]
        // [prev_size × batch_size] × [batch_size × cur_size] = [prev_size × cur_size]
        let prev_act = &scratch.act[layer_idx - 1];
        unsafe {
            sgemm(
                prev_size,                          // m
                batch_size,                         // k
                cur_size,                           // n
                1.0,                                // alpha
                prev_act.as_ptr(),                  // A (treated as transposed)
                1,                                  // rsa (col stride of original = row stride of transpose)
                prev_size as isize,                 // csa (row stride of original = col stride of transpose)
                d_cur.as_ptr(),                     // B
                cur_size as isize,                  // rsb
                1,                                  // csb
                1.0,                                // beta: ACCUMULATE into grad
                scratch.grad[off..].as_mut_ptr(),   // C
                cur_size as isize,                  // rsc
                1,                                  // csc
            );
        }

        // Bias gradient: column sum of d_act[layer]
        for bi in 0..batch_size {
            let row = &d_cur[bi * cur_size..(bi + 1) * cur_size];
            for j in 0..cur_size {
                scratch.grad[bias_offset + j] += row[j];
            }
        }

        // Input gradient: d_act[layer-1] = d_act[layer] × W^T
        // [batch_size × cur_size] × [cur_size × prev_size] = [batch_size × prev_size]
        unsafe {
            sgemm(
                batch_size,                         // m
                cur_size,                           // k
                prev_size,                          // n
                1.0,                                // alpha
                d_cur.as_ptr(),                     // A
                cur_size as isize,                  // rsa
                1,                                  // csa
                lw.as_ptr(),                        // B (treated as transposed)
                1,                                  // rsb (col stride of original = row stride of transpose)
                cur_size as isize,                  // csb (row stride of original = col stride of transpose)
                0.0,                                // beta
                d_prev.as_mut_ptr(),                // C
                prev_size as isize,                 // rsc
                1,                                  // csc
            );
        }

        // Apply ReLU mask for previous layer
        let pre_prev = &scratch.pre_act[layer_idx - 1];
        for i in 0..batch_size * prev_size {
            if pre_prev[i] <= 0.0 {
                d_prev[i] = 0.0;
            }
        }

        // Residual gradient: if layer (layer_idx-1) was the source for a residual
        // connection to some later layer, add that layer's gradient.
        // The destination layer is (layer_idx-1) + residual_interval.
        if residual_interval > 0 {
            let src = layer_idx - 1;
            let dst = src + residual_interval;
            if dst < num_layers
                && hidden_layers[dst] == hidden_layers[src]
            {
                // d_act[dst] already has ReLU mask applied; add to d_prev (= d_act[src])
                // We need to re-borrow since d_prev is d_act[layer_idx-1]
                let (d_lower2, d_upper2) = scratch.d_act.split_at_mut(dst);
                let d_dst = &d_upper2[0];
                let d_src2 = &mut d_lower2[src];
                let size = hidden_layers[src];
                for i in 0..batch_size * size {
                    d_src2[i] += d_dst[i];
                }
            }
        }
    }

    // ── L1 gradient (sparse): per-position ──
    if num_layers >= 1 {
        let h1 = hidden_layers[0];
        let off = scratch.layer_offsets[0];
        let bias_offset = off + input_size * h1;

        // d_act[0] already has ReLU mask applied (if num_layers > 1, it was done above;
        // if num_layers == 1, we need to handle the case where layer_idx==0 is the last hidden)

        // If num_layers == 1, d_act[0] was set from the output layer backprop and already
        // has ReLU mask applied. For num_layers > 1, the loop above handled it.

        let d_l1 = &scratch.d_act[0];
        for bi in 0..batch_size {
            let d_row = &d_l1[bi * h1..(bi + 1) * h1];
            let pos = positions[bi];

            // Bias gradient: vectorizable contiguous add
            let bias_grad = &mut scratch.grad[bias_offset..bias_offset + h1];
            for j in 0..h1 {
                bias_grad[j] += d_row[j];
            }

            // Sparse board features: iterate features in outer loop for contiguous grad writes
            for &idx in &pos.active_indices {
                let feat = idx as usize;
                let grad_col = &mut scratch.grad[off + feat * h1..off + feat * h1 + h1];
                for j in 0..h1 {
                    grad_col[j] += d_row[j];
                }
            }

            // Dense bag features
            for (i, &val) in pos.bag_features.iter().enumerate() {
                if val != 0.0 {
                    let feat = BOARD_FEATURES + i;
                    let grad_col = &mut scratch.grad[off + feat * h1..off + feat * h1 + h1];
                    for j in 0..h1 {
                        grad_col[j] += d_row[j] * val;
                    }
                }
            }
        }
    }
}

/// Compute batch MSE loss (sum of squared errors, not averaged).
#[inline]
fn batch_loss(scratch: &FcScratch, targets: &[f32], batch_size: usize) -> f64 {
    let mut loss = 0.0f64;
    for bi in 0..batch_size {
        let e = scratch.outputs[bi] - targets[bi];
        loss += (e * e) as f64;
    }
    loss
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
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let one_minus_beta1 = 1.0 - beta1;
        let one_minus_beta2 = 1.0 - beta2;
        let eps = self.eps;
        let n = weights.len();

        // Pass 1: update first moment
        for i in 0..n {
            self.m[i] = beta1 * self.m[i] + one_minus_beta1 * grad[i];
        }
        // Pass 2: update second moment
        for i in 0..n {
            self.v[i] = beta2 * self.v[i] + one_minus_beta2 * grad[i] * grad[i];
        }
        // Pass 3: update weights
        for i in 0..n {
            weights[i] -= lr_t * self.m[i] / (self.v[i].sqrt() + eps);
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
                 [--epochs 10] [--batch-size 256] [--label-index 0] \
                 [--eval-interval 50000] [--eval-games 500] [--benchmark base,random] \
                 [--checkpoint-dir D:/temp/supervised_nn] [--seed 42] \
                 [--residual 0]"
            );
            std::process::exit(1);
        });
    let label_index: usize = parse_flag(&args, "--label-index").unwrap_or(0);

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
    let batch_size: usize = parse_flag(&args, "--batch-size").unwrap_or(512);
    let eval_interval: usize = parse_flag(&args, "--eval-interval").unwrap_or(50000);
    let eval_games: u32 = parse_flag(&args, "--eval-games").unwrap_or(500);
    let checkpoint_dir: String = parse_flag(&args, "--checkpoint-dir")
        .unwrap_or_else(|| "D:/temp/supervised_nn".to_string());
    let seed: u64 = parse_flag(&args, "--seed").unwrap_or(42);

    let residual_interval: usize = parse_flag(&args, "--residual").unwrap_or(0);
    let sparse_init = args.contains(&"--sparse-init".to_string());
    let benchmark_str: String = parse_flag(&args, "--benchmark")
        .unwrap_or_else(|| "base,random".to_string());
    let benchmark_specs: Vec<String> = benchmark_str.split(',').map(|s| s.trim().to_string()).collect();

    // Print configuration
    eprintln!("=== Supervised Training ===");
    // Load data first so we know label range
    let (mut positions, _label_min, _label_max) = load_lpos(&input_path, label_index);

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
    if residual_interval > 0 {
        // Report which layers will actually get residual connections
        let mut residual_pairs = Vec::new();
        for l in 1..hidden_layers.len() {
            if l >= residual_interval
                && hidden_layers[l] == hidden_layers[l - residual_interval]
            {
                residual_pairs.push(format!("L{}->L{}", l - residual_interval, l));
            }
        }
        eprintln!("  Residual:       every {} layers ({})", residual_interval,
            if residual_pairs.is_empty() { "none active".to_string() }
            else { residual_pairs.join(", ") });
    } else {
        eprintln!("  Residual:       off");
    }
    if sparse_init {
        eprintln!("  Sparse init:    enabled (L1 uses effective fan_in)");
    }
    eprintln!();

    let max_positions: Option<usize> = parse_flag(&args, "--max-positions");
    if let Some(max) = max_positions {
        eprintln!("  Max positions:  {}", max);
    }

    // Create checkpoint directory
    std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoint directory");

    // Optionally truncate to --max-positions
    if let Some(max) = max_positions {
        if positions.len() > max {
            eprintln!("Truncating {} positions to {} ...", positions.len(), max);
            positions.truncate(max);
        }
    }
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

    // --sparse-init: re-initialize L1 weights using effective fan_in (avg active features)
    // instead of full input_size. With ~24 active out of 1106, default Kaiming makes L1
    // weights ~7x too small, weakening initial activations.
    if sparse_init {
        let avg_active: f64 = positions.iter()
            .take(10000)
            .map(|p| p.active_indices.len() as f64 + p.bag_features.iter().filter(|&&v| v != 0.0).count() as f64)
            .sum::<f64>() / positions.len().min(10000) as f64;
        let h1 = hidden_layers[0];
        let scale = (6.0 / avg_active).sqrt() as f32;
        let l1_weights = &mut net.weights[0..input_size * h1];
        for w in l1_weights.iter_mut() {
            *w = rng.gen::<f32>() * 2.0 * scale - scale;
        }
        eprintln!("  Sparse init:    L1 fan_in={:.0} (was {}), scale={:.4}", avg_active, input_size, scale);
    }

    let mut adam = AdamState::new(num_params, lr);
    let mut scratch = FcScratch::new(&hidden_layers, num_params, batch_size, input_size);
    let mut batch_targets = vec![0.0f32; batch_size];

    // Initial evaluation
    eprintln!("\n--- Initial evaluation ---");
    evaluate_model(&net, &benchmark_specs, eval_games);

    // Training loop
    let mut total_samples = 0usize;
    let mut shuffled_indices = indices.clone();

    // Adaptive learning rate: halve on loss spike, increase on stall, with bounds.
    let mut best_loss: f64 = f64::INFINITY;
    let mut batches_since_improvement: usize = 0;
    let mut recent_loss_sum: f64 = 0.0;
    let mut recent_loss_count: usize = 0;
    let lr_check_interval: usize = 1000;
    let lr_stall_threshold: usize = 5000;
    let lr_min: f32 = lr / 10.0;  // Never drop below 10% of initial LR
    let lr_max: f32 = lr * 3.0;   // Never exceed 3x initial LR (was 10x, too aggressive)

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

            // Gather batch position refs and targets
            let batch_positions: Vec<&LabeledPosition> = (batch_start..batch_end)
                .map(|si| &positions[shuffled_indices[si] as usize])
                .collect();
            for (i, pos) in batch_positions.iter().enumerate() {
                batch_targets[i] = label_to_target(pos.label);
            }

            // Zero gradient accumulator (reused across batches)
            scratch.zero_grad();

            // Batched forward pass
            batch_forward(
                &net.weights,
                input_size,
                &hidden_layers,
                &batch_positions,
                &mut scratch,
                residual_interval,
            );

            let batch_loss_val = batch_loss(&scratch, &batch_targets[..actual_batch_size], actual_batch_size);

            // Batched backward pass
            batch_backward(
                &net.weights,
                input_size,
                &hidden_layers,
                &batch_positions,
                &batch_targets[..actual_batch_size],
                inv_batch,
                &mut scratch,
                residual_interval,
            );

            // Adam update
            adam.step(&mut net.weights, &scratch.grad);

            epoch_loss += batch_loss_val;
            epoch_samples += actual_batch_size;
            total_samples += actual_batch_size;

            // Adaptive learning rate
            let avg_batch_loss = batch_loss_val / actual_batch_size as f64;
            recent_loss_sum += avg_batch_loss;
            recent_loss_count += 1;

            if recent_loss_count >= lr_check_interval {
                let recent_avg = recent_loss_sum / recent_loss_count as f64;
                if recent_avg < best_loss {
                    best_loss = recent_avg;
                    batches_since_improvement = 0;
                } else {
                    batches_since_improvement += recent_loss_count;

                    if recent_avg > best_loss * 1.05 {
                        // Loss spiked: halve LR (with floor)
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
                        // Stuck: bump LR to escape local minimum (with ceiling)
                        let old_lr = adam.lr;
                        adam.lr = (adam.lr * 1.5).min(lr_max);
                        if adam.lr != old_lr {
                            eprintln!(
                                "  LR adjusted: {} -> {} (loss stalled)",
                                old_lr, adam.lr
                            );
                            // Reset best_loss so higher LR gets a fair chance
                            best_loss = recent_avg;
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
                    "  Batch {}/{}: loss={}, samples={}, rate={:.0} pos/s",
                    batch_idx + 1,
                    num_batches,
                    fmt_loss(avg_loss),
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
            "Epoch {} complete: avg_loss={}, {:.1}s ({:.0} pos/s)",
            epoch + 1,
            fmt_loss(epoch_avg_loss),
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

#[cfg(test)]
mod tests {
    use super::*;
    use duke_training::encoding::TOTAL_FEATURES;
    use duke_training::generic_mlp::GenericMlp;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Create a dummy batch of labeled positions for testing.
    fn make_dummy_positions(n: usize) -> Vec<LabeledPosition> {
        (0..n)
            .map(|i| LabeledPosition {
                active_indices: vec![(i % 100) as u16, ((i * 7 + 3) % 200) as u16],
                bag_features: {
                    let mut b = [0.0f32; BAG_FEATURES];
                    b[0] = 1.0;
                    b[1] = 0.5;
                    b
                },
                label: (i as f32 - n as f32 / 2.0) * 0.1,
                count: 1,
            })
            .collect()
    }

    /// Run a forward pass and return outputs for the batch.
    fn run_forward(
        weights: &[f32],
        input_size: usize,
        hidden_layers: &[usize],
        positions: &[LabeledPosition],
        residual_interval: usize,
    ) -> Vec<f32> {
        let batch_size = positions.len();
        let num_params = GenericMlp::param_count(input_size, hidden_layers);
        let mut scratch = FcScratch::new(hidden_layers, num_params, batch_size, input_size);
        let pos_refs: Vec<&LabeledPosition> = positions.iter().collect();
        batch_forward(weights, input_size, hidden_layers, &pos_refs, &mut scratch, residual_interval);
        scratch.outputs[..batch_size].to_vec()
    }

    /// Run forward + backward and return (outputs, gradients).
    fn run_forward_backward(
        weights: &[f32],
        input_size: usize,
        hidden_layers: &[usize],
        positions: &[LabeledPosition],
        residual_interval: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let batch_size = positions.len();
        let num_params = GenericMlp::param_count(input_size, hidden_layers);
        let mut scratch = FcScratch::new(hidden_layers, num_params, batch_size, input_size);
        let pos_refs: Vec<&LabeledPosition> = positions.iter().collect();
        let targets: Vec<f32> = positions.iter().map(|p| label_to_target(p.label)).collect();
        let inv_batch = 1.0 / batch_size as f32;

        batch_forward(weights, input_size, hidden_layers, &pos_refs, &mut scratch, residual_interval);
        let outputs = scratch.outputs[..batch_size].to_vec();

        scratch.zero_grad();
        batch_backward(weights, input_size, hidden_layers, &pos_refs, &targets, inv_batch, &mut scratch, residual_interval);
        let grads = scratch.grad.clone();
        (outputs, grads)
    }

    #[test]
    fn test_residual_0_matches_no_residual() {
        // residual_interval=0 should produce identical outputs to no-residual behavior
        let mut rng = StdRng::seed_from_u64(42);
        let input_size = TOTAL_FEATURES;
        let hidden = vec![64, 64, 64];
        let net = GenericMlp::random(input_size, hidden.clone(), &mut rng);
        let positions = make_dummy_positions(8);

        let out_0 = run_forward(&net.weights, input_size, &hidden, &positions, 0);
        // Run again with same weights to confirm determinism
        let out_0b = run_forward(&net.weights, input_size, &hidden, &positions, 0);

        assert_eq!(out_0, out_0b, "forward pass should be deterministic");

        // Also check backward produces same gradients
        let (_, grad_0) = run_forward_backward(&net.weights, input_size, &hidden, &positions, 0);
        let (_, grad_0b) = run_forward_backward(&net.weights, input_size, &hidden, &positions, 0);
        assert_eq!(grad_0, grad_0b, "backward pass should be deterministic");
    }

    #[test]
    fn test_residual_only_same_width() {
        // 128->64->64->64: residual=1 should only apply between layers of width 64
        // Layer 0: 128, Layer 1: 64, Layer 2: 64, Layer 3: 64
        // Residual should apply: L1->L2 (both 64), L2->L3 (both 64)
        // But NOT input->L0 (different sizes), and NOT L0->L1 (128 != 64)
        let mut rng = StdRng::seed_from_u64(99);
        let input_size = TOTAL_FEATURES;
        let hidden_mixed = vec![128, 64, 64, 64];
        let net = GenericMlp::random(input_size, hidden_mixed.clone(), &mut rng);
        let positions = make_dummy_positions(8);

        let out_no_res = run_forward(&net.weights, input_size, &hidden_mixed, &positions, 0);
        let out_res1 = run_forward(&net.weights, input_size, &hidden_mixed, &positions, 1);

        // They should differ because layers 1->2 and 2->3 get residuals (both 64)
        assert_ne!(out_no_res, out_res1,
            "residual=1 should produce different outputs when same-width layers exist");

        // Now test with all-different widths: 128->64->32 -- no residuals possible
        let hidden_diff = vec![128, 64, 32];
        let net2 = GenericMlp::random(input_size, hidden_diff.clone(), &mut rng);
        let out_diff_no = run_forward(&net2.weights, input_size, &hidden_diff, &positions, 0);
        let out_diff_r1 = run_forward(&net2.weights, input_size, &hidden_diff, &positions, 1);
        assert_eq!(out_diff_no, out_diff_r1,
            "residual=1 with all-different widths should be identical to no residual");
    }

    #[test]
    fn test_residual_forward_finite() {
        // With residuals, outputs should still be finite
        let mut rng = StdRng::seed_from_u64(123);
        let input_size = TOTAL_FEATURES;
        let hidden = vec![64, 64, 64, 64, 64, 64, 64, 64]; // 8 layers
        let net = GenericMlp::random(input_size, hidden.clone(), &mut rng);
        let positions = make_dummy_positions(16);

        for interval in &[1, 2, 4] {
            let out = run_forward(&net.weights, input_size, &hidden, &positions, *interval);
            for (i, &v) in out.iter().enumerate() {
                assert!(v.is_finite(), "output[{}] not finite with residual={}", i, interval);
                assert!(v >= 0.0 && v <= 1.0, "output[{}]={} out of sigmoid range with residual={}", i, v, interval);
            }
        }
    }

    #[test]
    fn test_residual_changes_output() {
        // Residual vs no-residual should produce different outputs (same weights, same-width layers)
        let mut rng = StdRng::seed_from_u64(77);
        let input_size = TOTAL_FEATURES;
        let hidden = vec![64, 64, 64, 64];
        let net = GenericMlp::random(input_size, hidden.clone(), &mut rng);
        let positions = make_dummy_positions(8);

        let out_0 = run_forward(&net.weights, input_size, &hidden, &positions, 0);
        let out_1 = run_forward(&net.weights, input_size, &hidden, &positions, 1);
        let out_2 = run_forward(&net.weights, input_size, &hidden, &positions, 2);

        assert_ne!(out_0, out_1, "residual=1 should differ from residual=0");
        assert_ne!(out_0, out_2, "residual=2 should differ from residual=0");
        // residual=1 and residual=2 may or may not be the same, but usually differ
    }

    #[test]
    fn test_deep_network_training_with_residuals() {
        // A deep network (256x8) with residuals should be able to reduce loss
        let mut rng = StdRng::seed_from_u64(42);
        let input_size = TOTAL_FEATURES;
        let hidden = vec![64, 64, 64, 64, 64, 64, 64, 64]; // 8 layers (use 64 for speed)
        let mut net = GenericMlp::random(input_size, hidden.clone(), &mut rng);
        let num_params = GenericMlp::param_count(input_size, &hidden);
        let batch_size = 32;
        let residual_interval = 1;

        let positions = make_dummy_positions(batch_size);
        let pos_refs: Vec<&LabeledPosition> = positions.iter().collect();
        let targets: Vec<f32> = positions.iter().map(|p| label_to_target(p.label)).collect();
        let inv_batch = 1.0 / batch_size as f32;

        let mut scratch = FcScratch::new(&hidden, num_params, batch_size, input_size);
        let mut adam = AdamState::new(num_params, 0.001);

        // Compute initial loss
        batch_forward(&net.weights, input_size, &hidden, &pos_refs, &mut scratch, residual_interval);
        let initial_loss = batch_loss(&scratch, &targets, batch_size);

        // Train for 200 steps
        for _ in 0..200 {
            scratch.zero_grad();
            batch_forward(&net.weights, input_size, &hidden, &pos_refs, &mut scratch, residual_interval);
            batch_backward(&net.weights, input_size, &hidden, &pos_refs, &targets, inv_batch, &mut scratch, residual_interval);
            adam.step(&mut net.weights, &scratch.grad);
        }

        batch_forward(&net.weights, input_size, &hidden, &pos_refs, &mut scratch, residual_interval);
        let final_loss = batch_loss(&scratch, &targets, batch_size);

        assert!(final_loss < initial_loss,
            "Loss should decrease: initial={:.6}, final={:.6}", initial_loss, final_loss);
        assert!(final_loss.is_finite(), "Final loss should be finite");
    }

    #[test]
    fn test_residual_interval_2() {
        // With interval=2 on 64x6, residuals should be at layers 2,4 (skipping from 0,2)
        let mut rng = StdRng::seed_from_u64(55);
        let input_size = TOTAL_FEATURES;
        let hidden = vec![64, 64, 64, 64, 64, 64];
        let net = GenericMlp::random(input_size, hidden.clone(), &mut rng);
        let positions = make_dummy_positions(8);

        let out_0 = run_forward(&net.weights, input_size, &hidden, &positions, 0);
        let out_2 = run_forward(&net.weights, input_size, &hidden, &positions, 2);

        // Should differ since layers 2 and 4 get residual connections
        assert_ne!(out_0, out_2, "residual=2 should produce different outputs");
    }

    #[test]
    fn test_gradient_with_residual_is_finite() {
        // Gradients should be finite with residual connections
        let mut rng = StdRng::seed_from_u64(88);
        let input_size = TOTAL_FEATURES;
        let hidden = vec![64, 64, 64, 64, 64, 64, 64, 64];
        let net = GenericMlp::random(input_size, hidden.clone(), &mut rng);
        let positions = make_dummy_positions(8);

        let (_, grads) = run_forward_backward(&net.weights, input_size, &hidden, &positions, 1);
        for (i, &g) in grads.iter().enumerate() {
            assert!(g.is_finite(), "gradient[{}] not finite with residual=1", i);
        }

        let (_, grads2) = run_forward_backward(&net.weights, input_size, &hidden, &positions, 2);
        for (i, &g) in grads2.iter().enumerate() {
            assert!(g.is_finite(), "gradient[{}] not finite with residual=2", i);
        }
    }
}
