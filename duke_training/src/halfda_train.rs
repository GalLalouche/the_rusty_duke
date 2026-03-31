//! HalfDA NNUE training binary for The Duke.
//!
//! Trains a HalfDA value network on positions from trajectory files.
//! Uses burn for GPU-accelerated training with sparse inputs.
//!
//! Architecture: 67392 sparse -> 2048 (clipped ReLU) -> 1 (sigmoid)
//!
//! The sparse L1 accumulation is done on CPU (summing embedding rows for
//! active features), then the accumulated 2048-dim vectors are sent to the
//! burn backend for the rest of the forward pass, loss, and backprop.
//!
//! Usage:
//!   halfda_train --trajectories D:/temp/ckpt_heuristic_1m/trajectories.dtrj \
//!                --max-positions 100000 \
//!                --epochs 5 \
//!                --batch-size 1024 \
//!                --lr 0.001 \
//!                --hidden 2048 \
//!                --lambda 0.5 \
//!                --checkpoint-dir D:/temp/halfda

use std::time::Instant;

use burn::backend::Autodiff;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use duke_training::cli::parse_flag;
use duke_training::game_setup::StaticHeuristicEvaluator;
use duke_training::game_setup::GameEvaluator;
use duke_training::halfda::{encode_halfda, HALFDA_FEATURES};
use duke_training::supervised_common::{game_outcome_target, label_to_target};
use duke_training::trajectory_io::load_trajectories;

use duke_rust::game::state::GameResult;

// ── Backend selection ────────────────────────────────────────────────────
//
// LibTorch backend with CUDA support.
// Requires LIBTORCH env var pointing to PyTorch installation.

type TrainBackend = Autodiff<burn::backend::LibTorch>;
#[allow(dead_code)]
type InferBackend = burn::backend::LibTorch;

// ── Model ─────────────────────────────────────────────────────────────────

/// HalfDA NNUE model: Linear(hidden->1) with sigmoid.
///
/// The L1 (sparse -> hidden) accumulation is done manually on CPU by summing
/// embedding rows, so the burn model only handles the dense part.
#[derive(Module, Debug)]
struct HalfDAModel<B: Backend> {
    /// Output layer: hidden -> 1.
    output: burn::nn::Linear<B>,
}

impl<B: Backend> HalfDAModel<B> {
    fn new(device: &B::Device, hidden_size: usize) -> Self {
        let output = burn::nn::LinearConfig::new(hidden_size, 1).init(device);
        Self { output }
    }

    /// Forward pass on pre-accumulated L1 activations.
    /// `l1_out`: [batch, hidden] -- already ReLU-clipped on CPU.
    fn forward(&self, l1_out: Tensor<B, 2>) -> Tensor<B, 2> {
        let logits = self.output.forward(l1_out); // [batch, 1]
        sigmoid(logits)
    }
}

// ── Sparse L1 (embedding table on CPU) ───────────────────────────────────

/// CPU-side embedding table for the sparse L1 layer.
///
/// Weights are stored column-major: `weights[feature_idx * hidden + neuron]`.
/// This allows efficient accumulation by summing entire rows.
struct SparseL1 {
    weights: Vec<f32>,
    bias: Vec<f32>,
    hidden: usize,
}

impl SparseL1 {
    fn new(num_features: usize, hidden: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
        // For sparse inputs with ~8 active features: effective fan_in ~ 8
        let scale = (2.0 / (8.0 + hidden as f64)).sqrt() as f32;
        let weights: Vec<f32> = (0..num_features * hidden)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        let bias = vec![0.0f32; hidden];
        Self { weights, bias, hidden }
    }

    /// Accumulate active features into a dense vector, add bias, apply clipped ReLU.
    /// Returns a vector of length `hidden`.
    #[inline]
    #[allow(dead_code)]
    fn accumulate(&self, indices: &[u32]) -> Vec<f32> {
        let h = self.hidden;
        let mut out = self.bias.clone();
        for &idx in indices {
            let offset = idx as usize * h;
            let row = &self.weights[offset..offset + h];
            for j in 0..h {
                out[j] += row[j];
            }
        }
        // Clipped ReLU: clamp to [0, 1]
        for v in out.iter_mut() {
            *v = v.clamp(0.0, 1.0);
        }
        out
    }

    /// Accumulate a batch of positions into a flat buffer [batch_size * hidden].
    #[allow(dead_code)]
    fn accumulate_batch(&self, batch_indices: &[&[u32]], out_buf: &mut Vec<f32>) {
        let h = self.hidden;
        let batch_size = batch_indices.len();
        out_buf.resize(batch_size * h, 0.0);

        for (i, indices) in batch_indices.iter().enumerate() {
            let offset = i * h;
            // Start from bias
            out_buf[offset..offset + h].copy_from_slice(&self.bias);
            // Accumulate active features
            for &idx in *indices {
                let w_offset = idx as usize * h;
                let row = &self.weights[w_offset..w_offset + h];
                for j in 0..h {
                    out_buf[offset + j] += row[j];
                }
            }
            // Clipped ReLU
            for j in 0..h {
                let v = &mut out_buf[offset + j];
                *v = v.clamp(0.0, 1.0);
            }
        }
    }

    /// Apply gradients from burn's autodiff to update the sparse L1 weights.
    ///
    /// `dl_dl1_out` is the gradient of loss w.r.t. the L1 output (after ReLU).
    /// We need to backprop through clipped ReLU and accumulate into embedding rows.
    fn backward_and_step(
        &mut self,
        batch_indices: &[&[u32]],
        l1_pre_relu: &[f32],  // [batch * hidden], before clipped ReLU
        dl_dl1_out: &[f32],   // [batch * hidden], gradient w.r.t. L1 output
        _lr: f32,
        adam: &mut SparseAdamState,
    ) {
        let h = self.hidden;
        let batch_size = batch_indices.len();
        let inv_batch = 1.0 / batch_size as f32;

        // Accumulate bias gradient
        let mut bias_grad = vec![0.0f32; h];
        for i in 0..batch_size {
            let off = i * h;
            for j in 0..h {
                let pre = l1_pre_relu[off + j];
                // Clipped ReLU derivative: 1 if 0 < pre < 1, else 0
                let mask = if pre > 0.0 && pre < 1.0 { 1.0 } else { 0.0 };
                bias_grad[j] += dl_dl1_out[off + j] * mask * inv_batch;
            }
        }

        // Update bias with Adam
        adam.step_slice(&mut self.bias, &bias_grad, 0);

        // Accumulate and apply weight gradients (only for active features)
        // Use a temporary gradient buffer for each active feature
        for (i, indices) in batch_indices.iter().enumerate() {
            let off = i * h;
            for &idx in *indices {
                let w_off = idx as usize * h;
                let adam_off = (idx as usize + 1) * h; // +1 because bias uses slot 0
                let mut grad = vec![0.0f32; h];
                for j in 0..h {
                    let pre = l1_pre_relu[off + j];
                    let mask = if pre > 0.0 && pre < 1.0 { 1.0 } else { 0.0 };
                    grad[j] = dl_dl1_out[off + j] * mask * inv_batch;
                }
                adam.step_slice(&mut self.weights[w_off..w_off + h], &grad, adam_off);
            }
        }
    }
}

/// Simplified per-parameter Adam state for sparse updates.
///
/// Stores first/second moment per-parameter. Only updates touched parameters.
struct SparseAdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u64,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
}

impl SparseAdamState {
    fn new(total_slots: usize, lr: f32) -> Self {
        Self {
            m: vec![0.0; total_slots],
            v: vec![0.0; total_slots],
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    fn begin_step(&mut self) {
        self.t += 1;
    }

    /// Update a slice of weights with their gradients at the given Adam state offset.
    fn step_slice(&mut self, weights: &mut [f32], grad: &[f32], adam_offset: usize) {
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let t = self.t as f32;
        let lr_t = self.lr * (1.0 - beta2.powf(t)).sqrt() / (1.0 - beta1.powf(t));

        for (i, (&g, w)) in grad.iter().zip(weights.iter_mut()).enumerate() {
            let mi = &mut self.m[adam_offset + i];
            let vi = &mut self.v[adam_offset + i];
            *mi = beta1 * *mi + (1.0 - beta1) * g;
            *vi = beta2 * *vi + (1.0 - beta2) * g * g;
            *w -= lr_t * *mi / (vi.sqrt() + eps);
        }
    }
}

// ── Training data ─────────────────────────────────────────────────────────

struct TrainingPosition {
    halfda_indices: Vec<u32>,
    /// Combined target: lambda * eval_target + (1 - lambda) * game_outcome
    target: f32,
}

// ── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let traj_path: String = parse_flag(&args, "--trajectories").unwrap_or_else(|| {
        eprintln!(
            "Usage: halfda_train --trajectories <path.dtrj> \
             [--max-positions 100000] [--epochs 5] [--batch-size 1024] \
             [--lr 0.001] [--hidden 2048] [--lambda 0.5] [--seed 42] \
             [--checkpoint-dir D:/temp/halfda]"
        );
        std::process::exit(1);
    });

    let max_positions: usize = parse_flag(&args, "--max-positions").unwrap_or(100_000);
    let epochs: usize = parse_flag(&args, "--epochs").unwrap_or(5);
    let batch_size: usize = parse_flag(&args, "--batch-size").unwrap_or(1024);
    let lr: f32 = parse_flag(&args, "--lr").unwrap_or(0.001);
    let hidden: usize = parse_flag(&args, "--hidden").unwrap_or(2048);
    let lambda: f32 = parse_flag(&args, "--lambda").unwrap_or(0.5);
    let scores_path: Option<String> = parse_flag(&args, "--scores");
    let seed: u64 = parse_flag(&args, "--seed").unwrap_or(42);
    let checkpoint_dir: String = parse_flag(&args, "--checkpoint-dir")
        .unwrap_or_else(|| "D:/temp/halfda".to_string());

    assert!((0.0..=1.0).contains(&lambda), "--lambda must be in [0.0, 1.0], got {}", lambda);

    let backend_name = "LibTorch (CUDA if available)";

    eprintln!("=== HalfDA NNUE Training ===");
    eprintln!("  Trajectories:    {}", traj_path);
    eprintln!("  Max positions:   {}", max_positions);
    eprintln!("  Epochs:          {}", epochs);
    eprintln!("  Batch size:      {}", batch_size);
    eprintln!("  Learning rate:   {}", lr);
    eprintln!("  Hidden size:     {}", hidden);
    eprintln!("  Lambda:          {} (eval={:.0}%, outcome={:.0}%)", lambda, lambda * 100.0, (1.0 - lambda) * 100.0);
    eprintln!("  Seed:            {}", seed);
    eprintln!("  Checkpoint dir:  {}", checkpoint_dir);
    eprintln!("  Features:        {} (HalfDA)", HALFDA_FEATURES);
    eprintln!("  Architecture:    {} -> {} (clipped ReLU) -> 1 (sigmoid)", HALFDA_FEATURES, hidden);
    eprintln!("  Backend:         {}", backend_name);

    let total_params = HALFDA_FEATURES * hidden + hidden + hidden + 1;
    eprintln!("  Parameters:      {} ({:.1}M)", total_params, total_params as f64 / 1e6);
    eprintln!();

    std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoint directory");

    // ── Load trajectories and extract positions ─────────────────────────

    let t0 = Instant::now();
    eprintln!("Loading trajectories from {} ...", traj_path);
    let trajectories = load_trajectories(&traj_path).expect("Failed to load trajectories");
    eprintln!("  Loaded {} games in {:.1}s", trajectories.len(), t0.elapsed().as_secs_f64());

    // Load precomputed depth-N scores if provided, otherwise fall back to static eval
    let precomputed_scores: Option<Vec<f32>> = if let Some(ref sp) = scores_path {
        eprintln!("Loading precomputed scores from {} ...", sp);
        let data = std::fs::read(sp).expect("Failed to read scores file");
        let n = data.len() / 4;
        let scores: Vec<f32> = data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        eprintln!("  Loaded {} scores", n);
        // Verify count matches total states
        let total_states: usize = trajectories.iter().map(|g| g.states.len()).sum();
        assert_eq!(n, total_states,
            "Scores file has {} entries but trajectories have {} states", n, total_states);

        // Quantile normalization: map each score to its rank percentile, then to [-10, +10].
        // This guarantees uniform spread across the target range regardless of the
        // original score distribution. Without this, 90% of depth-3 scores cluster
        // near zero and map to targets ~0.5, making positions indistinguishable.
        let mut indexed: Vec<(usize, f32)> = scores.iter().enumerate()
            .filter(|(_, s)| !s.is_nan() && s.abs() < 29.9)
            .map(|(i, &s)| (i, s))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let nn = indexed.len();
        let mut scores = scores;
        if nn > 0 {
            for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
                // Map rank to [-10, +10]: rank 0 -> -10, rank nn-1 -> +10
                let normalized = (rank as f32 / (nn - 1).max(1) as f32) * 20.0 - 10.0;
                scores[orig_idx] = normalized;
            }
            eprintln!("  Quantile-normalized {} non-terminal scores to [-10, +10]", nn);
            // Terminals stay as +-30 — label_to_target maps them to 0/1
        }
        Some(scores)
    } else {
        None
    };
    let evaluator = StaticHeuristicEvaluator::new();

    eprintln!("Extracting positions (max {}) ...", max_positions);
    let t1 = Instant::now();
    let mut positions: Vec<TrainingPosition> = Vec::with_capacity(max_positions);

    let mut score_idx = 0usize;
    'outer: for game in &trajectories {
        let game_result = game.result;

        for gs in &game.states {
            let current_score_idx = score_idx;
            score_idx += 1;

            // Skip terminal states
            if gs.clone().game_result() != GameResult::Ongoing {
                continue;
            }

            let halfda = encode_halfda(gs);

            // Use precomputed depth-N score if available, otherwise static eval
            let raw_label = if let Some(ref scores) = precomputed_scores {
                let s = scores[current_score_idx];
                if s.is_nan() { continue; } // skip NaN (terminal/epsilon-random)
                s
            } else {
                evaluator.evaluate(gs)
            };
            let eval_target = label_to_target(raw_label);

            let current_player = gs.current_player_turn();
            let outcome = game_outcome_target(game_result, current_player);

            let target = lambda * eval_target + (1.0 - lambda) * outcome;

            positions.push(TrainingPosition {
                halfda_indices: halfda.as_slice().to_vec(),
                target,
            });

            if positions.len() >= max_positions {
                break 'outer;
            }
        }
    }

    let num_positions = positions.len();
    eprintln!(
        "  Extracted {} positions in {:.1}s",
        num_positions,
        t1.elapsed().as_secs_f64()
    );

    // Stats
    let avg_features: f64 = positions.iter().map(|p| p.halfda_indices.len() as f64).sum::<f64>()
        / num_positions as f64;
    let avg_target: f64 = positions.iter().map(|p| p.target as f64).sum::<f64>()
        / num_positions as f64;
    eprintln!("  Avg features/pos: {:.1}", avg_features);
    eprintln!("  Avg target:       {:.4}", avg_target);
    eprintln!();

    // ── Initialize model ────────────────────────────────────────────────

    // Sparse L1 on CPU
    let mut sparse_l1 = SparseL1::new(HALFDA_FEATURES, hidden);

    // Adam for sparse L1: (1 bias slot + HALFDA_FEATURES weight slots) * hidden
    let adam_slots = (1 + HALFDA_FEATURES) * hidden;
    let mut sparse_adam = SparseAdamState::new(adam_slots, lr);

    // Dense part on burn (GPU/CPU)
    let device = <TrainBackend as Backend>::Device::default();
    let mut model = HalfDAModel::<TrainBackend>::new(&device, hidden);
    let mut optimizer: OptimizerAdaptor<Adam, HalfDAModel<TrainBackend>, TrainBackend> =
        AdamConfig::new().init();

    eprintln!("Model initialized. Backend: {}", backend_name);
    eprintln!();

    // ── Training loop ───────────────────────────────────────────────────

    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..num_positions).collect();
    let mut total_samples = 0usize;

    // Pre-allocate buffers
    let mut l1_buf: Vec<f32> = Vec::with_capacity(batch_size * hidden);
    let mut l1_pre_relu_buf: Vec<f32> = Vec::with_capacity(batch_size * hidden);

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        eprintln!("=== Epoch {}/{} ===", epoch + 1, epochs);

        indices.shuffle(&mut rng);

        let num_batches = (num_positions + batch_size - 1) / batch_size;
        let mut epoch_loss = 0.0f64;
        let mut epoch_samples = 0usize;

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;
            let batch_end = (batch_start + batch_size).min(num_positions);
            let actual_batch = batch_end - batch_start;

            // Gather batch
            let batch_positions: Vec<&TrainingPosition> = (batch_start..batch_end)
                .map(|i| &positions[indices[i]])
                .collect();

            // ── CPU: Sparse L1 accumulation ─────────────────────────

            // Collect index slices for the batch
            let batch_idx_slices: Vec<&[u32]> = batch_positions
                .iter()
                .map(|p| p.halfda_indices.as_slice())
                .collect();

            // Compute pre-ReLU activations (for backward pass)
            l1_pre_relu_buf.resize(actual_batch * hidden, 0.0);
            for (i, indices_slice) in batch_idx_slices.iter().enumerate() {
                let off = i * hidden;
                l1_pre_relu_buf[off..off + hidden].copy_from_slice(&sparse_l1.bias);
                for &idx in *indices_slice {
                    let w_off = idx as usize * hidden;
                    let row = &sparse_l1.weights[w_off..w_off + hidden];
                    for j in 0..hidden {
                        l1_pre_relu_buf[off + j] += row[j];
                    }
                }
            }

            // Apply clipped ReLU for the forward pass
            l1_buf.resize(actual_batch * hidden, 0.0);
            l1_buf.copy_from_slice(&l1_pre_relu_buf);
            for v in l1_buf.iter_mut() {
                *v = v.clamp(0.0, 1.0);
            }

            // ── GPU: Forward pass, loss, backward ───────────────────

            // Send L1 output to burn
            let l1_tensor = Tensor::<TrainBackend, 2>::from_floats(
                burn::tensor::TensorData::new(l1_buf.clone(), [actual_batch, hidden]),
                &device,
            )
            .require_grad();

            // Targets
            let target_buf: Vec<f32> = batch_positions.iter().map(|p| p.target).collect();
            let targets = Tensor::<TrainBackend, 2>::from_floats(
                burn::tensor::TensorData::new(target_buf, [actual_batch, 1]),
                &device,
            );

            // Forward
            let predictions = model.forward(l1_tensor.clone()); // [batch, 1]

            // MSE loss
            let diff = predictions.clone() - targets;
            let loss = diff.clone().mul(diff).mean();

            let loss_value: f32 = loss
                .clone()
                .into_data()
                .to_vec::<f32>()
                .expect("loss extraction")[0];

            epoch_loss += loss_value as f64 * actual_batch as f64;
            epoch_samples += actual_batch;
            total_samples += actual_batch;

            // Backward
            let grads = loss.backward();

            // Get gradient w.r.t. L1 output tensor (for sparse L1 backward)
            let dl_dl1: Vec<f32> = l1_tensor
                .grad(&grads)
                .expect("L1 gradient must exist")
                .into_data()
                .to_vec::<f32>()
                .expect("gradient extraction");

            // Update dense model (output layer) via burn optimizer
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr as f64, model, grads_params);

            // Update sparse L1 via CPU Adam
            sparse_adam.begin_step();
            sparse_l1.backward_and_step(
                &batch_idx_slices,
                &l1_pre_relu_buf,
                &dl_dl1,
                lr,
                &mut sparse_adam,
            );

            // Progress logging
            if (batch_idx + 1) % 100 == 0 || batch_idx + 1 == num_batches {
                let avg_loss = epoch_loss / epoch_samples as f64;
                let elapsed = epoch_start.elapsed().as_secs_f64();
                let rate = epoch_samples as f64 / elapsed;
                eprintln!(
                    "  Batch {}/{}: loss={:.6}, samples={}, rate={:.0} pos/s",
                    batch_idx + 1, num_batches, avg_loss, epoch_samples, rate,
                );
            }
        }

        let epoch_avg_loss = epoch_loss / epoch_samples as f64;
        let epoch_elapsed = epoch_start.elapsed();
        let epoch_rate = epoch_samples as f64 / epoch_elapsed.as_secs_f64();
        eprintln!(
            "Epoch {} complete: avg_loss={:.6}, {:.1}s, {:.0} pos/s",
            epoch + 1, epoch_avg_loss, epoch_elapsed.as_secs_f64(), epoch_rate,
        );
        eprintln!();
    }

    eprintln!("=== Training complete ===");
    eprintln!("Total samples processed: {}", total_samples);

    // Save sparse L1 weights
    let l1_path = format!("{}/halfda_l1.bin", checkpoint_dir);
    save_sparse_l1(&sparse_l1, &l1_path).expect("Failed to save sparse L1");
    eprintln!("Saved sparse L1 weights to: {}", l1_path);

    // Save burn model
    let model_path = format!("{}/halfda_dense", checkpoint_dir);
    let recorder = burn::record::NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(&model_path, &recorder)
        .expect("Failed to save dense model");
    eprintln!("Saved dense model to: {}.mpk", model_path);

    // Also save output weights as plain binary for use by HalfDAEvaluator (no burn dependency)
    let dense_bin_path = format!("{}/halfda_output.bin", checkpoint_dir);
    {
        use std::io::Write;
        let weight_tensor = model.output.weight.val();
        let bias_tensor = model.output.bias.as_ref().expect("output layer has no bias").val();
        let weight_data: Vec<f32> = weight_tensor.into_data().to_vec().expect("weight to_vec failed");
        let bias_data: Vec<f32> = bias_tensor.into_data().to_vec().expect("bias to_vec failed");
        let mut f = std::fs::File::create(&dense_bin_path).expect("Failed to create output.bin");
        f.write_all(b"HDA2").unwrap(); // magic
        f.write_all(&(weight_data.len() as u32).to_le_bytes()).unwrap();
        for &w in &weight_data {
            f.write_all(&w.to_le_bytes()).unwrap();
        }
        f.write_all(&(bias_data.len() as u32).to_le_bytes()).unwrap();
        for &b in &bias_data {
            f.write_all(&b.to_le_bytes()).unwrap();
        }
        eprintln!("Saved plain output weights to: {}", dense_bin_path);
    }
}

// ── Save/load sparse L1 ─────────────────────────────────────────────────

fn save_sparse_l1(l1: &SparseL1, path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"HDA1")?; // magic
    f.write_all(&1u32.to_le_bytes())?; // version
    f.write_all(&(HALFDA_FEATURES as u32).to_le_bytes())?;
    f.write_all(&(l1.hidden as u32).to_le_bytes())?;

    // Bias
    for &v in &l1.bias {
        f.write_all(&v.to_le_bytes())?;
    }
    // Weights
    for &v in &l1.weights {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}
