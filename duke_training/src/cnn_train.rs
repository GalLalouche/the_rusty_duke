//! CNN supervised training binary for The Duke.
//!
//! Trains a CNN value network on labeled positions (LPOS format), using
//! burn's autodiff for backpropagation. Supports box (3x3), diamond
//! (manhattan-2 masked 5x5), and cross (3x3+5x1+1x5) kernels.
//!
//! Usage:
//!   cnn_train --input D:/temp/labeled_positions_v2.bin \
//!             --conv-channels 64,64,32 \
//!             --fc-sizes 128 \
//!             --kernel box \
//!             --epochs 20 \
//!             --batch-size 256 \
//!             --lr 0.001 \
//!             --eval-interval 500000 \
//!             --eval-games 500 \
//!             --benchmark base,random \
//!             --checkpoint-dir D:/temp/cnn_box

use std::time::Instant;

use burn::backend::{Autodiff, NdArray};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use duke_training::cli::parse_flag;
use duke_training::cnn_model::{apply_diamond_mask, CnnValueNetwork, KernelType};
use duke_training::encoding::{
    active_board_features, bag_features as compute_bag_features, BAG_FEATURES, BOARD_FEATURES,
    BOARD_SIZE, NUM_BOARD_PLANES,
};
use duke_training::game_setup::{create_bag, create_initial_state, GameEvaluator};
use duke_training::loaded_model::LoadedModel;
use duke_training::match_runner::{run_matches, win_rate, Player};

use duke_rust::game::state::GameState;

// ── Type aliases ──────────────────────────────────────────────────────────

type TrainBackend = Autodiff<NdArray>;
type InferBackend = NdArray;

// ── Labeled position data ─────────────────────────────────────────────────

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
fn load_lpos(path: &str) -> (Vec<LabeledPosition>, f32, f32) {
    let t0 = Instant::now();
    eprintln!("Loading labeled positions from {} ...", path);

    let data = std::fs::read(path).expect("Failed to read LPOS file");
    let mut cursor = 0usize;

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

    let magic = read_bytes!(4);
    assert_eq!(magic, b"LPOS", "Not an LPOS file (bad magic)");
    let version = read_u32!();
    assert_eq!(version, 1, "Unsupported LPOS version {}", version);
    let num_positions = read_u32!() as usize;

    eprintln!(
        "  File header: {} positions, version {}",
        num_positions, version
    );

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

// ── Label normalization ───────────────────────────────────────────────────

/// Clamp label to [-10, +10] then map linearly to [0, 1].
const LABEL_CLAMP: f32 = 10.0;

#[inline]
fn label_to_target(label: f32) -> f32 {
    let clamped = label.clamp(-LABEL_CLAMP, LABEL_CLAMP);
    (clamped + LABEL_CLAMP) / (2.0 * LABEL_CLAMP)
}

// ── Tensor encoding ───────────────────────────────────────────────────────

/// Encode a batch of LPOS positions into board and bag tensors for the CNN.
///
/// Returns:
/// - `board`: `[batch, 30, 6, 6]` float tensor
/// - `bag`: `[batch, 26]` float tensor
/// - `targets`: `[batch, 1]` float tensor
fn encode_batch<B: Backend>(
    positions: &[&LabeledPosition],
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 2>, Tensor<B, 2>) {
    let batch_size = positions.len();
    let board_plane_size = BOARD_SIZE * BOARD_SIZE; // 36
    let board_vol = NUM_BOARD_PLANES * board_plane_size; // 30 * 36 = 1080

    // Allocate flat buffers
    let mut board_buf = vec![0.0f32; batch_size * board_vol];
    let mut bag_buf = vec![0.0f32; batch_size * BAG_FEATURES];
    let mut target_buf = vec![0.0f32; batch_size];

    for (i, pos) in positions.iter().enumerate() {
        // Board: set active indices to 1.0
        let board_offset = i * board_vol;
        for &idx in &pos.active_indices {
            let idx = idx as usize;
            debug_assert!(idx < BOARD_FEATURES, "board feature index {} >= {}", idx, BOARD_FEATURES);
            // The sparse index encodes plane * 36 + cell, which maps directly
            // to our [30, 6, 6] layout stored in row-major order.
            board_buf[board_offset + idx] = 1.0;
        }

        // Bag features
        let bag_offset = i * BAG_FEATURES;
        bag_buf[bag_offset..bag_offset + BAG_FEATURES].copy_from_slice(&pos.bag_features);

        // Target
        target_buf[i] = label_to_target(pos.label);
    }

    let board = Tensor::<B, 4>::from_floats(
        burn::tensor::TensorData::new(board_buf, [batch_size, NUM_BOARD_PLANES, BOARD_SIZE, BOARD_SIZE]),
        device,
    );
    let bag = Tensor::<B, 2>::from_floats(
        burn::tensor::TensorData::new(bag_buf, [batch_size, BAG_FEATURES]),
        device,
    );
    let targets = Tensor::<B, 2>::from_floats(
        burn::tensor::TensorData::new(target_buf, [batch_size, 1]),
        device,
    );

    (board, bag, targets)
}

/// Encode a single game state into board and bag tensors for the CNN.
fn encode_game_state<B: Backend>(
    gs: &GameState,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 2>) {
    let board_plane_size = BOARD_SIZE * BOARD_SIZE;
    let board_vol = NUM_BOARD_PLANES * board_plane_size;

    let mut board_buf = vec![0.0f32; board_vol];
    let features = active_board_features(gs);
    for &idx in features.as_slice() {
        debug_assert!(idx < BOARD_FEATURES);
        board_buf[idx] = 1.0;
    }

    let bag = compute_bag_features(gs);

    let board = Tensor::<B, 4>::from_floats(
        burn::tensor::TensorData::new(board_buf, [1, NUM_BOARD_PLANES, BOARD_SIZE, BOARD_SIZE]),
        device,
    );
    let bag_tensor = Tensor::<B, 2>::from_floats(
        burn::tensor::TensorData::new(bag.to_vec(), [1, BAG_FEATURES]),
        device,
    );

    (board, bag_tensor)
}

// ── Game evaluator wrapper ────────────────────────────────────────────────

/// Wraps a burn CnnValueNetwork for game-playing evaluation using NdArray backend.
struct BurnCnnEvaluator {
    model: CnnValueNetwork<InferBackend>,
    device: <InferBackend as Backend>::Device,
}

impl GameEvaluator for BurnCnnEvaluator {
    fn evaluate(&self, gs: &GameState) -> f32 {
        let (board, bag) = encode_game_state::<InferBackend>(gs, &self.device);
        let output = self.model.forward(board, bag); // [1, 1]
        let val: Vec<f32> = output.into_data().to_vec().expect("scalar extraction");
        val[0]
    }
}

// BurnCnnEvaluator is Send + Sync because NdArray tensors are Send + Sync.
unsafe impl Send for BurnCnnEvaluator {}
unsafe impl Sync for BurnCnnEvaluator {}

// ── Evaluation ────────────────────────────────────────────────────────────

/// Evaluate the current CNN against benchmark opponents.
fn evaluate_model(
    model: &CnnValueNetwork<TrainBackend>,
    benchmark_specs: &[String],
    eval_games: u32,
) {
    let bag = create_bag();
    let gs = create_initial_state(&bag);

    // Convert autodiff model to inner (inference) model
    use burn::module::AutodiffModule;
    let inner_model = model.clone().valid();
    let inner_device = <InferBackend as Backend>::Device::default();

    let evaluator = BurnCnnEvaluator {
        model: inner_model,
        device: inner_device,
    };
    let model_player = Player::Evaluator(&evaluator);

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

// ── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let input_path: String = parse_flag(&args, "--input").unwrap_or_else(|| {
        eprintln!(
            "Usage: cnn_train --input <path> [--conv-channels 64,64,32] [--fc-sizes 128] \
             [--kernel box] [--epochs 20] [--batch-size 256] [--lr 0.001] \
             [--eval-interval 500000] [--eval-games 500] [--benchmark base,random] \
             [--checkpoint-dir D:/temp/cnn_box] [--seed 42]"
        );
        std::process::exit(1);
    });

    let conv_str: String = parse_flag(&args, "--conv-channels").unwrap_or_else(|| "64,64,32".to_string());
    let conv_channels: Vec<usize> = conv_str
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .expect("--conv-channels values must be comma-separated integers")
        })
        .collect();
    assert!(
        !conv_channels.is_empty(),
        "--conv-channels must specify at least one channel count"
    );

    let fc_str: String = parse_flag(&args, "--fc-sizes").unwrap_or_else(|| "128".to_string());
    let fc_sizes: Vec<usize> = fc_str
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .expect("--fc-sizes values must be comma-separated integers")
        })
        .collect();

    let kernel_str: String = parse_flag(&args, "--kernel").unwrap_or_else(|| "box".to_string());
    let kernel_type = match kernel_str.as_str() {
        "box" => KernelType::Box,
        "diamond" => KernelType::Diamond,
        "cross" => KernelType::Cross,
        other => panic!(
            "Unknown kernel type '{}'. Use 'box' (3x3), 'diamond' (manhattan-2 5x5), or 'cross' (3x3+5x1+1x5).",
            other
        ),
    };
    let use_diamond = kernel_type == KernelType::Diamond;

    let lr: f64 = parse_flag(&args, "--lr").unwrap_or(0.001);
    let epochs: usize = parse_flag(&args, "--epochs").unwrap_or(20);
    let batch_size: usize = parse_flag(&args, "--batch-size").unwrap_or(256);
    let eval_interval: usize = parse_flag(&args, "--eval-interval").unwrap_or(500000);
    let eval_games: u32 = parse_flag(&args, "--eval-games").unwrap_or(500);
    let checkpoint_dir: String = parse_flag(&args, "--checkpoint-dir")
        .unwrap_or_else(|| "D:/temp/cnn_box".to_string());
    let seed: u64 = parse_flag(&args, "--seed").unwrap_or(42);

    let benchmark_str: String =
        parse_flag(&args, "--benchmark").unwrap_or_else(|| "base,random".to_string());
    let benchmark_specs: Vec<String> = benchmark_str
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    // Print configuration
    eprintln!("=== CNN Supervised Training ===");

    let (positions, _label_min, _label_max) = load_lpos(&input_path);

    eprintln!("  Input:            {}", input_path);
    let kernel_label = match kernel_type {
        KernelType::Box => "box (3x3)",
        KernelType::Diamond => "diamond (5x5 masked)",
        KernelType::Cross => "cross (3x3+5x1+1x5)",
    };
    eprintln!(
        "  Conv channels:    {:?} (kernel={})",
        conv_channels,
        kernel_label,
    );
    eprintln!("  FC sizes:         {:?}", fc_sizes);
    eprintln!("  Learning rate:    {}", lr);
    eprintln!("  Epochs:           {}", epochs);
    eprintln!("  Batch size:       {}", batch_size);
    eprintln!(
        "  Label mapping:    clamp to +-{}, linear to (0..1)",
        LABEL_CLAMP
    );
    eprintln!("  Eval interval:    {} positions", eval_interval);
    eprintln!("  Eval games:       {}", eval_games);
    eprintln!("  Benchmark:        {:?}", benchmark_specs);
    eprintln!("  Checkpoint dir:   {}", checkpoint_dir);
    eprintln!("  Seed:             {}", seed);
    eprintln!();

    // Create checkpoint directory
    std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoint directory");
    let num_positions = positions.len();

    // Build weighted index array for sampling
    eprintln!("Building weighted index array ...");
    let t_idx = Instant::now();
    let total_weighted: usize = positions.iter().map(|p| p.count as usize).sum();

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

    // Initialize model
    let device = <TrainBackend as Backend>::Device::default();
    let mut model = CnnValueNetwork::<TrainBackend>::new(
        &device,
        &conv_channels,
        &fc_sizes,
        kernel_type,
    );

    // Apply diamond mask to initial weights if needed
    if use_diamond {
        apply_diamond_mask(&mut model, &device);
    }

    // Count parameters
    let num_params = burn::module::Module::num_params(&model);
    eprintln!("CNN model: {} parameters", num_params);

    let last_conv_ch = *conv_channels.last().unwrap();
    let fc_input = last_conv_ch * BOARD_SIZE * BOARD_SIZE + BAG_FEATURES;
    eprintln!(
        "  Conv: {} -> {:?} ({}) -> flatten {}",
        NUM_BOARD_PLANES,
        conv_channels,
        kernel_label,
        last_conv_ch * BOARD_SIZE * BOARD_SIZE
    );
    eprintln!(
        "  FC: {} -> {:?} -> 1 (sigmoid)",
        fc_input, fc_sizes
    );
    eprintln!();

    // Initialize optimizer
    let mut optimizer: OptimizerAdaptor<Adam, CnnValueNetwork<TrainBackend>, TrainBackend> =
        AdamConfig::new().init();

    // Initial evaluation
    eprintln!("--- Initial evaluation ---");
    evaluate_model(&model, &benchmark_specs, eval_games);

    // Training loop
    let mut rng = StdRng::seed_from_u64(seed);
    let mut total_samples = 0usize;
    let mut shuffled_indices = indices.clone();

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        eprintln!("\n=== Epoch {}/{} ===", epoch + 1, epochs);

        shuffled_indices.shuffle(&mut rng);

        let num_batches = (effective_total + batch_size - 1) / batch_size;
        let mut epoch_loss = 0.0f64;
        let mut epoch_samples = 0usize;
        let mut last_eval_at = 0usize;

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;
            let batch_end = (batch_start + batch_size).min(effective_total);
            let actual_batch_size = batch_end - batch_start;

            // Gather batch positions
            let batch_positions: Vec<&LabeledPosition> = (batch_start..batch_end)
                .map(|si| &positions[shuffled_indices[si] as usize])
                .collect();

            // Encode batch
            let (board, bag, targets) =
                encode_batch::<TrainBackend>(&batch_positions, &device);

            // Forward pass
            let predictions = model.forward(board, bag); // [batch, 1]

            // MSE loss
            let diff = predictions.clone() - targets;
            let loss = diff.clone().mul(diff).mean();

            let loss_value: f32 = loss
                .clone()
                .into_data()
                .to_vec::<f32>()
                .expect("loss extraction")[0];

            epoch_loss += loss_value as f64 * actual_batch_size as f64;
            epoch_samples += actual_batch_size;
            total_samples += actual_batch_size;

            // Backward pass and optimizer step
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads);

            // Apply diamond mask after optimizer step
            if use_diamond {
                apply_diamond_mask(&mut model, &device);
            }

            // Progress logging
            if (batch_idx + 1) % 500 == 0 || batch_idx + 1 == num_batches {
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
                eprintln!(
                    "\n  --- Evaluation at {} samples (epoch {}) ---",
                    epoch_samples,
                    epoch + 1
                );
                evaluate_model(&model, &benchmark_specs, eval_games);
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
        let ckpt_path = format!("{}/cnn_epoch_{}", checkpoint_dir, epoch + 1);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model
            .clone()
            .save_file(&ckpt_path, &recorder)
            .expect("Failed to save checkpoint");
        eprintln!("  Saved checkpoint: {}.mpk", ckpt_path);

        // End-of-epoch evaluation
        eprintln!("  --- End-of-epoch evaluation ---");
        evaluate_model(&model, &benchmark_specs, eval_games);
    }

    // Save final model
    let final_path = format!("{}/cnn_final", checkpoint_dir);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(&final_path, &recorder)
        .expect("Failed to save final model");
    eprintln!("\n=== Training complete ===");
    eprintln!("Final model saved to: {}.mpk", final_path);
    eprintln!("Total samples processed: {}", total_samples);

    // Final evaluation
    eprintln!("\n--- Final evaluation ---");
    evaluate_model(&model, &benchmark_specs, eval_games);
}
