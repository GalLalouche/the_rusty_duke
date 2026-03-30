//! Supervised CNN training binary for The Duke.
//!
//! Trains a manual CNN (no burn) on labeled positions (LPOS format),
//! using the pure-Rust forward/backward in `duke_training::cnn`.
//!
//! Usage:
//!   supervised_train_cnn --input D:/temp/labeled_positions.bin \
//!                        --conv-channels 64,64,32 \
//!                        --fc-sizes 128 \
//!                        --kernel box \
//!                        --lr 0.001 \
//!                        --epochs 10 \
//!                        --batch-size 256 \
//!                        --eval-interval 50000 \
//!                        --eval-games 500 \
//!                        --benchmark base,random \
//!                        --checkpoint-dir D:/temp/supervised_cnn

use std::time::Instant;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use duke_training::cli::parse_flag;
use duke_training::cnn::{apply_diamond_mask, CnnEvaluator, CnnModel, KernelType};
use duke_training::encoding::BAG_FEATURES;
use duke_training::game_setup::{create_bag, create_initial_state};
use duke_training::loaded_model::LoadedModel;
use duke_training::match_runner::{run_matches, win_rate, Player};

// ── Labeled position data ────────────────────────────────────────────────

struct LabeledPosition {
    active_indices: Vec<u16>,
    bag_features: [f32; BAG_FEATURES],
    label: f32,
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
        positions.len(), file_mb, elapsed.as_secs_f64()
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

// ── Helpers ──────────────────────────────────────────────────────────────

const LABEL_CLAMP: f32 = 10.0;

#[inline]
fn label_to_target(label: f32) -> f32 {
    let clamped = label.clamp(-LABEL_CLAMP, LABEL_CLAMP);
    (clamped + LABEL_CLAMP) / (2.0 * LABEL_CLAMP)
}

// ── Adam optimizer ───────────────────────────────────────────────────────

struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u64,
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

// ── Evaluation ───────────────────────────────────────────────────────────

fn evaluate_model(
    model: &CnnModel,
    benchmark_specs: &[String],
    eval_games: u32,
) {
    let bag = create_bag();
    let gs = create_initial_state(&bag);

    // Clone the model for evaluation
    let eval_model = CnnModel {
        kernel_type: model.kernel_type,
        conv_channels: model.conv_channels.clone(),
        fc_sizes: model.fc_sizes.clone(),
        weights: model.weights.clone(),
        input_channels: model.input_channels,
        board_size: model.board_size,
        bag_features: model.bag_features,
    };
    let model_eval = CnnEvaluator { model: eval_model };
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

// ── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let input_path: String = parse_flag(&args, "--input")
        .unwrap_or_else(|| {
            eprintln!(
                "Usage: supervised_train_cnn --input <path> [--conv-channels 64,64,32] \
                 [--fc-sizes 128] [--kernel box|diamond|cross] [--lr 0.001] \
                 [--epochs 10] [--batch-size 256] [--label-index 0] \
                 [--eval-interval 50000] [--eval-games 500] [--benchmark base,random] \
                 [--checkpoint-dir D:/temp/supervised_cnn] [--seed 42] \
                 [--max-positions N]"
            );
            std::process::exit(1);
        });

    let conv_str: String = parse_flag(&args, "--conv-channels").unwrap_or_else(|| "64,64,32".to_string());
    let conv_channels: Vec<usize> = conv_str
        .split(',')
        .map(|s| s.trim().parse::<usize>().expect("--conv-channels values must be comma-separated integers"))
        .collect();
    assert!(!conv_channels.is_empty(), "--conv-channels must specify at least one channel count");

    let fc_str: String = parse_flag(&args, "--fc-sizes").unwrap_or_else(|| "128".to_string());
    let fc_sizes: Vec<usize> = fc_str
        .split(',')
        .map(|s| s.trim().parse::<usize>().expect("--fc-sizes values must be comma-separated integers"))
        .collect();

    let kernel_str: String = parse_flag(&args, "--kernel").unwrap_or_else(|| "box".to_string());
    let kernel_type = match kernel_str.as_str() {
        "box" => KernelType::Box,
        "diamond" => KernelType::Diamond,
        "cross" => KernelType::Cross,
        other => panic!("Unknown kernel type '{}'. Use box, diamond, or cross.", other),
    };

    let lr: f32 = parse_flag(&args, "--lr").unwrap_or(0.001);
    let epochs: usize = parse_flag(&args, "--epochs").unwrap_or(10);
    let batch_size: usize = parse_flag(&args, "--batch-size").unwrap_or(256);
    let label_index: usize = parse_flag(&args, "--label-index").unwrap_or(0);
    let eval_interval: usize = parse_flag(&args, "--eval-interval").unwrap_or(50000);
    let eval_games: u32 = parse_flag(&args, "--eval-games").unwrap_or(500);
    let checkpoint_dir: String = parse_flag(&args, "--checkpoint-dir")
        .unwrap_or_else(|| "D:/temp/supervised_cnn".to_string());
    let seed: u64 = parse_flag(&args, "--seed").unwrap_or(42);

    let benchmark_str: String = parse_flag(&args, "--benchmark")
        .unwrap_or_else(|| "base,random".to_string());
    let benchmark_specs: Vec<String> = benchmark_str.split(',').map(|s| s.trim().to_string()).collect();
    let max_positions: Option<usize> = parse_flag(&args, "--max-positions");

    // Print configuration
    eprintln!("=== Supervised CNN Training ===");
    let (mut positions, _label_min, _label_max) = load_lpos(&input_path, label_index);

    let input_channels = duke_training::encoding::NUM_BOARD_PLANES;
    let board_size = duke_training::encoding::BOARD_SIZE;
    let bag_features = duke_training::encoding::BAG_FEATURES;

    let num_params = CnnModel::param_count(
        kernel_type, input_channels, &conv_channels, &fc_sizes, board_size, bag_features,
    );

    eprintln!("  Input:          {}", input_path);
    eprintln!("  Kernel:         {:?}", kernel_type);
    eprintln!("  Conv channels:  {:?}", conv_channels);
    eprintln!("  FC sizes:       {:?}", fc_sizes);
    eprintln!("  Parameters:     {}", num_params);
    eprintln!("  Learning rate:  {}", lr);
    eprintln!("  Epochs:         {}", epochs);
    eprintln!("  Batch size:     {}", batch_size);
    eprintln!("  Label mapping:  clamp to +-{}, linear to (0..1)", LABEL_CLAMP);
    eprintln!("  Eval interval:  {} positions", eval_interval);
    eprintln!("  Eval games:     {}", eval_games);
    eprintln!("  Benchmark:      {:?}", benchmark_specs);
    eprintln!("  Checkpoint dir: {}", checkpoint_dir);
    eprintln!("  Seed:           {}", seed);
    if let Some(max) = max_positions {
        eprintln!("  Max positions:  {}", max);
    }
    eprintln!();

    std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoint directory");

    // Optionally truncate to --max-positions
    if let Some(max) = max_positions {
        if positions.len() > max {
            eprintln!("Truncating {} positions to {} ...", positions.len(), max);
            positions.truncate(max);
        }
    }
    let num_positions = positions.len();

    // Build weighted index array
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
        effective_total, num_positions, t_idx.elapsed().as_secs_f64()
    );

    // Initialize model
    let mut rng = StdRng::seed_from_u64(seed);
    let mut model = CnnModel::random(
        kernel_type,
        input_channels,
        conv_channels.clone(),
        fc_sizes.clone(),
        board_size,
        bag_features,
        &mut rng,
    );
    let mut adam = AdamState::new(num_params, lr);
    let mut batch_scratch = model.create_batch_scratch(batch_size);

    // Pre-allocate usize active index buffer (avoids per-position Vec<usize> alloc)
    let mut active_buf: Vec<usize> = Vec::with_capacity(64);

    // Per-position active indices saved for conv backward (need to replay per-position)
    let mut batch_active_indices: Vec<Vec<usize>> = (0..batch_size).map(|_| Vec::with_capacity(64)).collect();
    let mut batch_bag_features: Vec<[f32; BAG_FEATURES]> = vec![[0.0f32; BAG_FEATURES]; batch_size];

    // Initial evaluation
    eprintln!("\n--- Initial evaluation ---");
    evaluate_model(&model, &benchmark_specs, eval_games);

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

            let mut grad = vec![0.0f32; num_params];

            // Step 1: Per-position conv forward → write FC inputs into batch matrix
            for (bi, si) in (batch_start..batch_end).enumerate() {
                let pos_idx = shuffled_indices[si] as usize;
                let pos = &positions[pos_idx];

                // Convert u16 indices to usize (reuse buffer)
                active_buf.clear();
                active_buf.extend(pos.active_indices.iter().map(|&i| i as usize));

                // Save active indices and bag features for conv backward
                batch_active_indices[bi].clear();
                batch_active_indices[bi].extend_from_slice(&active_buf);
                batch_bag_features[bi] = pos.bag_features;

                model.conv_forward_into_batch(&active_buf, &pos.bag_features, bi, &mut batch_scratch);
            }

            // Step 2: Batched FC forward (sgemm)
            model.batch_fc_forward(actual_batch_size, &mut batch_scratch);

            // Compute batch targets and loss
            let mut batch_targets: Vec<f32> = Vec::with_capacity(actual_batch_size);
            let mut batch_loss = 0.0f64;
            for (bi, si) in (batch_start..batch_end).enumerate() {
                let pos_idx = shuffled_indices[si] as usize;
                let target = label_to_target(positions[pos_idx].label);
                batch_targets.push(target);
                let error = batch_scratch.outputs[bi] - target;
                batch_loss += (error * error) as f64;
            }

            // Step 3: Batched FC backward (sgemm) → FC weight gradients + d_fc_inputs
            model.batch_fc_backward(actual_batch_size, &batch_targets, inv_batch, &mut batch_scratch, &mut grad);

            // Step 4: Per-position conv backward using saved intermediates
            for bi in 0..actual_batch_size {
                model.conv_backward_from_batch(bi, &batch_active_indices[bi], &mut batch_scratch, &mut grad);
            }

            // Adam update
            adam.step(&mut model.weights, &grad);

            // Apply diamond mask if needed
            if kernel_type == KernelType::Diamond {
                apply_diamond_mask(&mut model);
            }

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
                    best_loss = recent_avg;
                    batches_since_improvement = 0;
                } else {
                    batches_since_improvement += recent_loss_count;
                    if recent_avg > best_loss * 1.05 {
                        let old_lr = adam.lr;
                        adam.lr = (adam.lr * 0.5).max(lr_min);
                        if adam.lr != old_lr {
                            eprintln!("  LR adjusted: {} -> {} (loss increased)", old_lr, adam.lr);
                        }
                        batches_since_improvement = 0;
                    } else if batches_since_improvement >= lr_stall_threshold {
                        let old_lr = adam.lr;
                        adam.lr = (adam.lr * 1.5).min(lr_max);
                        if adam.lr != old_lr {
                            eprintln!("  LR adjusted: {} -> {} (loss stalled)", old_lr, adam.lr);
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
                    batch_idx + 1, num_batches, avg_loss, epoch_samples, rate,
                );
            }

            // Periodic evaluation
            if eval_interval > 0 && (epoch_samples - last_eval_at) >= eval_interval {
                last_eval_at = epoch_samples;
                eprintln!("\n  --- Evaluation at {} samples (epoch {}) ---", epoch_samples, epoch + 1);
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
        let ckpt_path = format!("{}/epoch_{}.gcnn", checkpoint_dir, epoch + 1);
        model.save(&ckpt_path).expect("Failed to save checkpoint");
        eprintln!("  Saved checkpoint: {}", ckpt_path);

        // End-of-epoch evaluation
        eprintln!("  --- End-of-epoch evaluation ---");
        evaluate_model(&model, &benchmark_specs, eval_games);
    }

    // Save final model
    let final_path = format!("{}/final.gcnn", checkpoint_dir);
    model.save(&final_path).expect("Failed to save final model");
    eprintln!("\n=== Training complete ===");
    eprintln!("Final model saved to: {}", final_path);
    eprintln!("Total samples processed: {}", total_samples);

    // Final evaluation
    eprintln!("\n--- Final evaluation ---");
    evaluate_model(&model, &benchmark_specs, eval_games);
}
