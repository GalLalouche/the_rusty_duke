//! Evolutionary Strategies (ES) training: optimize NNUE weights by playing
//! against the heuristic opponent.
//!
//! Each iteration:
//! 1. Take current weight vector w (flattened NNUE weights)
//! 2. Generate N perturbation vectors epsilon_i ~ N(0, I)
//! 3. Evaluate w + sigma*epsilon_i and w - sigma*epsilon_i (mirrored sampling)
//!    by playing K games each against StaticHeuristicEvaluator
//! 4. Compute reward_i = win_rate for each perturbation
//! 5. Update: w += lr / (N * sigma) * sum((reward_plus_i - reward_minus_i) * epsilon_i)
//!
//! Usage: es_train [--resume <nnue_path>] [--l1 256] [--l2 32]
//!                 [--pop 50] [--games 10] [--sigma 0.01] [--lr 0.01]
//!                 [--iterations 200] [--eval-interval 20] [--eval-games 500]
//!                 [--checkpoint-dir <dir>]
//!                 [--append-combined]  — use 1147-input network (1106 NNUE + 41 combined)
//!                 [--input-features combined]  — use 41 combined features only

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use duke_rust::game::state::{GameResult, GameState};
use duke_rust::game::tile::Owner;

use duke_training::encoding::{active_board_features, bag_features, BOARD_FEATURES, TOTAL_FEATURES};
use duke_training::game_setup::{
    create_bag, create_initial_state, GameEvaluator, StaticHeuristicEvaluator,
};
use duke_training::learned_heuristic::{extract_combined_features, NUM_COMBINED_FEATURES};
use duke_training::match_runner::{play_match, run_matches, Player};
use duke_training::nnue::{NnueEvaluator, NnueWeights, NUM_FEATURES};

// ── Flatten / unflatten ──────────────────────────────────────────────────

fn flatten_weights(w: &NnueWeights) -> Vec<f32> {
    let mut flat = Vec::with_capacity(weight_count(w.l1_size, w.l2_size));
    flat.extend_from_slice(&w.l1_weight);
    flat.extend_from_slice(&w.l1_bias);
    flat.extend_from_slice(&w.l2_weight);
    flat.extend_from_slice(&w.l2_bias);
    flat.extend_from_slice(&w.l3_weight);
    flat.extend_from_slice(&w.l3_bias);
    flat
}

fn unflatten_weights(flat: &[f32], l1_size: usize, l2_size: usize) -> NnueWeights {
    let mut offset = 0;
    let take = |off: &mut usize, n: usize| -> Vec<f32> {
        let slice = flat[*off..*off + n].to_vec();
        *off += n;
        slice
    };
    let l1_weight = take(&mut offset, NUM_FEATURES * l1_size);
    let l1_bias = take(&mut offset, l1_size);
    let l2_weight = take(&mut offset, l2_size * l1_size);
    let l2_bias = take(&mut offset, l2_size);
    let l3_weight = take(&mut offset, l2_size);
    let l3_bias = take(&mut offset, 1);
    assert_eq!(offset, flat.len());
    NnueWeights {
        l1_size,
        l2_size,
        l1_weight,
        l1_bias,
        l2_weight,
        l2_bias,
        l3_weight,
        l3_bias,
    }
}

/// Flatten only l3_weight and l3_bias (last layer).
fn flatten_last_layer(w: &NnueWeights) -> Vec<f32> {
    let mut flat = Vec::with_capacity(last_layer_count(w.l2_size));
    flat.extend_from_slice(&w.l3_weight);
    flat.extend_from_slice(&w.l3_bias);
    flat
}

/// Unflatten only l3_weight and l3_bias, keeping everything else from `base`.
fn unflatten_last_layer(flat: &[f32], base: &NnueWeights) -> NnueWeights {
    let l2_size = base.l2_size;
    assert_eq!(flat.len(), l2_size + 1);
    NnueWeights {
        l1_size: base.l1_size,
        l2_size: base.l2_size,
        l1_weight: base.l1_weight.clone(),
        l1_bias: base.l1_bias.clone(),
        l2_weight: base.l2_weight.clone(),
        l2_bias: base.l2_bias.clone(),
        l3_weight: flat[..l2_size].to_vec(),
        l3_bias: flat[l2_size..].to_vec(),
    }
}

fn weight_count(l1_size: usize, l2_size: usize) -> usize {
    NUM_FEATURES * l1_size + l1_size       // L1 weight + bias
        + l2_size * l1_size + l2_size      // L2 weight + bias
        + l2_size + 1                      // L3 weight + bias
}

fn last_layer_count(l2_size: usize) -> usize {
    l2_size + 1  // l3_weight (l2_size) + l3_bias (1)
}

// ── Appended NNUE evaluator (1147 inputs) ────────────────────────────────

/// Total input size: 1106 NNUE features + 41 combined features = 1147
const APPENDED_INPUT_SIZE: usize = TOTAL_FEATURES + NUM_COMBINED_FEATURES; // 1147

/// Weight count for an appended-input network: input_size->l1->l2->1
fn appended_weight_count(l1_size: usize, l2_size: usize) -> usize {
    APPENDED_INPUT_SIZE * l1_size + l1_size   // L1 weight + bias
        + l2_size * l1_size + l2_size         // L2 weight + bias
        + l2_size + 1                         // L3 weight + bias
}

/// Evaluator for the 1147-input network. Stores flattened weights.
/// Layout: [l1_weight (1147*l1), l1_bias (l1), l2_weight (l2*l1), l2_bias (l2), l3_weight (l2), l3_bias (1)]
struct AppendedNnueEvaluator {
    weights: Vec<f32>,
    l1_size: usize,
    l2_size: usize,
}

impl AppendedNnueEvaluator {
    fn new(weights: Vec<f32>, l1_size: usize, l2_size: usize) -> Self {
        assert_eq!(weights.len(), appended_weight_count(l1_size, l2_size),
            "weight vector length mismatch: expected {}, got {}",
            appended_weight_count(l1_size, l2_size), weights.len());
        Self { weights, l1_size, l2_size }
    }

    /// Forward pass: computes features and evaluates in one step.
    /// Uses stack arrays to avoid heap allocation in the hot path.
    fn evaluate_state(&self, gs: &GameState) -> f32 {
        let l1 = self.l1_size;
        let l2 = self.l2_size;
        let w = &self.weights;

        // Compute weight offsets
        let l1_weight_end = APPENDED_INPUT_SIZE * l1;
        let l1_bias_end = l1_weight_end + l1;
        let l2_weight_end = l1_bias_end + l2 * l1;
        let l2_bias_end = l2_weight_end + l2;
        let l3_weight_end = l2_bias_end + l2;

        let l1_weight = &w[..l1_weight_end];
        let l1_bias = &w[l1_weight_end..l1_bias_end];
        let l2_weight = &w[l1_bias_end..l2_weight_end];
        let l2_bias = &w[l2_weight_end..l2_bias_end];
        let l3_weight = &w[l2_bias_end..l3_weight_end];
        let l3_bias = w[l3_weight_end];

        // L1: start with bias, accumulate sparse board features
        assert!(l1 <= 1024);
        let mut l1_out = [0.0f32; 1024];
        l1_out[..l1].copy_from_slice(l1_bias);

        // Sparse board features (binary, indices into first 1080 dimensions)
        let board_feats = active_board_features(gs);
        for &feat in board_feats.as_slice() {
            let col = &l1_weight[feat * l1..(feat + 1) * l1];
            for j in 0..l1 {
                l1_out[j] += col[j];
            }
        }

        // Bag features (dense, dimensions 1080..1106)
        let bag = bag_features(gs);
        for (i, &val) in bag.iter().enumerate() {
            if val != 0.0 {
                let feat = BOARD_FEATURES + i;
                let col = &l1_weight[feat * l1..(feat + 1) * l1];
                for j in 0..l1 {
                    l1_out[j] += col[j] * val;
                }
            }
        }

        // Combined features (dense, dimensions 1106..1147)
        let combined = extract_combined_features(gs);
        for (i, &val) in combined.iter().enumerate() {
            let fval = val as f32;
            if fval != 0.0 {
                let feat = TOTAL_FEATURES + i;
                let col = &l1_weight[feat * l1..(feat + 1) * l1];
                if fval == 1.0 {
                    for j in 0..l1 {
                        l1_out[j] += col[j];
                    }
                } else {
                    for j in 0..l1 {
                        l1_out[j] += col[j] * fval;
                    }
                }
            }
        }

        // ReLU L1
        for j in 0..l1 {
            l1_out[j] = l1_out[j].max(0.0);
        }

        // L2: row-major matmul + ReLU
        assert!(l2 <= 128);
        let mut l2_out = [0.0f32; 128];
        for i in 0..l2 {
            let mut sum = l2_bias[i];
            let row = &l2_weight[i * l1..(i + 1) * l1];
            for j in 0..l1 {
                sum += row[j] * l1_out[j];
            }
            l2_out[i] = sum.max(0.0);
        }

        // L3: output + sigmoid
        let mut output = l3_bias;
        for j in 0..l2 {
            output += l3_weight[j] * l2_out[j];
        }

        1.0 / (1.0 + (-output).exp())
    }

    /// Save weights to a binary file with header.
    fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"DKAP")?; // magic for "DuKe APpended"
        f.write_all(&1u32.to_le_bytes())?; // version
        f.write_all(&(self.l1_size as u32).to_le_bytes())?;
        f.write_all(&(self.l2_size as u32).to_le_bytes())?;
        f.write_all(&(APPENDED_INPUT_SIZE as u32).to_le_bytes())?;
        for &val in &self.weights {
            f.write_all(&val.to_le_bytes())?;
        }
        Ok(())
    }

    /// Load weights from a binary file.
    fn load(path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"DKAP" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid DKAP file magic"));
        }
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != 1 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("Unsupported DKAP version {}", version)));
        }
        f.read_exact(&mut buf4)?;
        let l1_size = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let l2_size = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let input_size = u32::from_le_bytes(buf4) as usize;
        if input_size != APPENDED_INPUT_SIZE {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("Input size mismatch: file has {}, expected {}", input_size, APPENDED_INPUT_SIZE)));
        }
        let n = appended_weight_count(l1_size, l2_size);
        let mut buf = vec![0u8; n * 4];
        f.read_exact(&mut buf)?;
        let weights: Vec<f32> = buf.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Self { weights, l1_size, l2_size })
    }
}

impl GameEvaluator for AppendedNnueEvaluator {
    fn evaluate(&self, gs: &GameState) -> f32 {
        self.evaluate_state(gs)
    }
}

/// Create random weights for the appended network using Kaiming init.
fn random_appended_weights(l1_size: usize, l2_size: usize, rng: &mut StdRng) -> Vec<f32> {
    let n = appended_weight_count(l1_size, l2_size);
    let mut weights = Vec::with_capacity(n);

    let rand_vec = |n: usize, fan_in: usize, rng: &mut StdRng| -> Vec<f32> {
        let scale = (2.0 / fan_in as f64).sqrt() as f32;
        (0..n).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect()
    };

    // L1 weight + bias
    weights.extend(rand_vec(APPENDED_INPUT_SIZE * l1_size, APPENDED_INPUT_SIZE, rng));
    weights.extend(vec![0.0f32; l1_size]);
    // L2 weight + bias
    weights.extend(rand_vec(l2_size * l1_size, l1_size, rng));
    weights.extend(vec![0.0f32; l2_size]);
    // L3 weight + bias
    weights.extend(rand_vec(l2_size, l2_size, rng));
    weights.push(0.0);

    assert_eq!(weights.len(), n);
    weights
}

/// Play K games of an appended-NNUE evaluator vs opponent. Returns win rate.
fn evaluate_appended_perturbation(
    evaluator: &AppendedNnueEvaluator,
    opponent: &(dyn GameEvaluator + Sync),
    gs: &GameState,
    k: u32,
    seed_base: u64,
) -> f32 {
    let nnue_player = Player::Evaluator(evaluator);
    let opp_player = Player::Evaluator(opponent);

    let mut score = 0.0f32;
    for i in 0..k {
        let mut rng = StdRng::seed_from_u64(seed_base + i as u64);
        let result = if i % 2 == 0 {
            play_match(gs, &nnue_player, &opp_player, &mut rng, 200)
        } else {
            play_match(gs, &opp_player, &nnue_player, &mut rng, 200)
        };
        let nnue_is_top = i % 2 == 0;
        match result {
            GameResult::Won(Owner::TopPlayer) => {
                if nnue_is_top { score += 1.0; }
            }
            GameResult::Won(Owner::BottomPlayer) => {
                if !nnue_is_top { score += 1.0; }
            }
            _ => { score += 0.5; }
        }
    }
    score / k as f32
}

/// Run the ES training loop in append-combined mode (1147-input network).
fn run_appended_training(
    l1_size: usize,
    l2_size: usize,
    pop_size: usize,
    games_per_eval: u32,
    sigma: f32,
    lr: f32,
    iterations: u32,
    eval_interval: u32,
    eval_games: u32,
    checkpoint_dir: &str,
    gs: &GameState,
) {
    let dim = appended_weight_count(l1_size, l2_size);

    println!("=== Evolutionary Strategies Training (APPENDED 1147) ===");
    println!("  network: {}->{}->{}->1", APPENDED_INPUT_SIZE, l1_size, l2_size);
    println!("  weight dimension: {}", dim);
    println!("  population: {} (x2 with mirroring = {})", pop_size, pop_size * 2);
    println!("  games per perturbation: {}", games_per_eval);
    println!("  sigma: {}, lr: {}", sigma, lr);
    println!("  mode: vs heuristic");
    println!("  iterations: {}", iterations);
    println!("  eval every {} iters with {} games", eval_interval, eval_games);
    println!("  starting from random weights");

    std::fs::create_dir_all(checkpoint_dir).expect("Failed to create checkpoint dir");

    let mut rng = StdRng::seed_from_u64(42);
    let mut w = random_appended_weights(l1_size, l2_size, &mut rng);

    // Evaluate initial win rate
    {
        let init_eval = AppendedNnueEvaluator::new(w.clone(), l1_size, l2_size);
        let nnue_player = Player::Evaluator(&init_eval);
        let heur_player = Player::Evaluator(&StaticHeuristicEvaluator::new());
        print!("  INIT: ");
        run_matches(gs, &nnue_player, &heur_player, eval_games, "Appended1147 vs Heuristic");
    }

    let total_start = Instant::now();

    for iter in 0..iterations {
        let iter_start = Instant::now();

        let perturbation_seeds: Vec<u64> = (0..pop_size)
            .map(|_| rng.gen::<u64>())
            .collect();
        let game_seed_base: u64 = rng.gen();

        let results: Vec<(usize, f32, f32)> = (0..pop_size * 2)
            .into_par_iter()
            .map(|idx| {
                let pert_idx = idx / 2;
                let is_positive = idx % 2 == 0;
                let pert_seed = perturbation_seeds[pert_idx];

                let mut pert_rng = StdRng::seed_from_u64(pert_seed);
                let epsilon = randn_vec(dim, &mut pert_rng);

                let perturbed: Vec<f32> = if is_positive {
                    w.iter().zip(epsilon.iter()).map(|(&wi, &ei)| wi + sigma * ei).collect()
                } else {
                    w.iter().zip(epsilon.iter()).map(|(&wi, &ei)| wi - sigma * ei).collect()
                };

                let evaluator = AppendedNnueEvaluator::new(perturbed, l1_size, l2_size);
                let game_seed = game_seed_base.wrapping_add(idx as u64 * 10000);
                let heuristic = StaticHeuristicEvaluator::new();
                let win_rate = evaluate_appended_perturbation(&evaluator, &heuristic, gs, games_per_eval, game_seed);

                (pert_idx, win_rate, 0.0)
            })
            .collect();

        let mut reward_plus = vec![0.0f32; pop_size];
        let mut reward_minus = vec![0.0f32; pop_size];
        for (i, &(pert_idx, win_rate, _)) in results.iter().enumerate() {
            if i % 2 == 0 {
                reward_plus[pert_idx] = win_rate;
            } else {
                reward_minus[pert_idx] = win_rate;
            }
        }

        let scale = lr / (pop_size as f32 * sigma);
        let mut grad = vec![0.0f32; dim];

        for i in 0..pop_size {
            let diff = reward_plus[i] - reward_minus[i];
            if diff.abs() < 1e-12 {
                continue;
            }
            let mut pert_rng = StdRng::seed_from_u64(perturbation_seeds[i]);
            let epsilon = randn_vec(dim, &mut pert_rng);

            for j in 0..dim {
                grad[j] += diff * epsilon[j];
            }
        }

        for j in 0..dim {
            w[j] += scale * grad[j];
        }

        let avg_plus: f32 = reward_plus.iter().sum::<f32>() / pop_size as f32;
        let avg_minus: f32 = reward_minus.iter().sum::<f32>() / pop_size as f32;
        let max_wr = reward_plus.iter().chain(reward_minus.iter())
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let games_this_iter = pop_size as u32 * 2 * games_per_eval;
        println!(
            "iter {:>4}/{}: avg_wr+={:.3} avg_wr-={:.3} max_wr={:.3} ({} games in {:.1?})",
            iter + 1, iterations,
            avg_plus, avg_minus, max_wr,
            games_this_iter, iter_start.elapsed()
        );

        if (iter + 1) % eval_interval == 0 || iter == iterations - 1 {
            let eval = AppendedNnueEvaluator::new(w.clone(), l1_size, l2_size);

            let ckpt_path = format!("{}/es_appended_iter_{}.bin", checkpoint_dir, iter + 1);
            eval.save(&ckpt_path).expect("Failed to save checkpoint");
            println!("  Saved checkpoint: {}", ckpt_path);

            let nnue_player = Player::Evaluator(&eval);
            let heur_player = Player::Evaluator(&StaticHeuristicEvaluator::new());
            print!("  EVAL: ");
            run_matches(
                gs, &nnue_player, &heur_player, eval_games,
                &format!("Appended1147(ES iter={}) vs Heuristic", iter + 1),
            );
        }
    }

    // Save final weights
    let final_eval = AppendedNnueEvaluator::new(w, l1_size, l2_size);
    let final_path = format!("{}/es_appended_final.bin", checkpoint_dir);
    final_eval.save(&final_path).expect("Failed to save final weights");
    println!("\nES appended training complete: {} iterations in {:.1?}", iterations, total_start.elapsed());
    println!("Final weights saved to: {}", final_path);
}

// ── Combined-features small network (41 inputs) ─────────────────────────
// A self-contained input_size->L1->L2->1 network with ReLU hidden layers
// and sigmoid output. Operates entirely on Vec<f32>.

/// Compact weight container for the combined-features network.
struct CombinedNet {
    input_size: usize,
    l1_size: usize,
    l2_size: usize,
    /// Flat weight vector: [l1_w, l1_b, l2_w, l2_b, l3_w, l3_b]
    weights: Vec<f32>,
}

impl CombinedNet {
    fn param_count(input_size: usize, l1_size: usize, l2_size: usize) -> usize {
        input_size * l1_size + l1_size         // L1 weight + bias
            + l1_size * l2_size + l2_size      // L2 weight + bias
            + l2_size + 1                      // L3 weight + bias (output)
    }

    fn from_flat(flat: Vec<f32>, input_size: usize, l1_size: usize, l2_size: usize) -> Self {
        assert_eq!(
            flat.len(),
            Self::param_count(input_size, l1_size, l2_size),
            "flat weight vector size mismatch: expected {}, got {}",
            Self::param_count(input_size, l1_size, l2_size), flat.len()
        );
        Self { input_size, l1_size, l2_size, weights: flat }
    }

    fn random(input_size: usize, l1_size: usize, l2_size: usize, rng: &mut StdRng) -> Self {
        let n = Self::param_count(input_size, l1_size, l2_size);
        let mut flat = Vec::with_capacity(n);

        // L1 weights: Kaiming init with fan_in = input_size
        let scale1 = (2.0 / input_size as f64).sqrt() as f32;
        for _ in 0..(input_size * l1_size) {
            flat.push(rng.gen::<f32>() * 2.0 * scale1 - scale1);
        }
        // L1 bias
        for _ in 0..l1_size { flat.push(0.0); }

        // L2 weights: fan_in = l1_size
        let scale2 = (2.0 / l1_size as f64).sqrt() as f32;
        for _ in 0..(l1_size * l2_size) {
            flat.push(rng.gen::<f32>() * 2.0 * scale2 - scale2);
        }
        // L2 bias
        for _ in 0..l2_size { flat.push(0.0); }

        // L3 weights: fan_in = l2_size
        let scale3 = (2.0 / l2_size as f64).sqrt() as f32;
        for _ in 0..l2_size {
            flat.push(rng.gen::<f32>() * 2.0 * scale3 - scale3);
        }
        // L3 bias
        flat.push(0.0);

        assert_eq!(flat.len(), n);
        Self { input_size, l1_size, l2_size, weights: flat }
    }

    /// Forward pass: input -> L1(ReLU) -> L2(ReLU) -> sigmoid output in [0, 1].
    fn forward(&self, input: &[f64; NUM_COMBINED_FEATURES]) -> f32 {
        let w = &self.weights;
        let mut off = 0;

        // L1: input_size -> l1_size, ReLU
        let l1_w = &w[off..off + self.input_size * self.l1_size];
        off += self.input_size * self.l1_size;
        let l1_b = &w[off..off + self.l1_size];
        off += self.l1_size;

        let mut h1 = vec![0.0f32; self.l1_size];
        for j in 0..self.l1_size {
            let mut sum = l1_b[j];
            for i in 0..self.input_size {
                sum += l1_w[i * self.l1_size + j] * input[i] as f32;
            }
            h1[j] = sum.max(0.0); // ReLU
        }

        // L2: l1_size -> l2_size, ReLU
        let l2_w = &w[off..off + self.l1_size * self.l2_size];
        off += self.l1_size * self.l2_size;
        let l2_b = &w[off..off + self.l2_size];
        off += self.l2_size;

        let mut h2 = vec![0.0f32; self.l2_size];
        for j in 0..self.l2_size {
            let mut sum = l2_b[j];
            for i in 0..self.l1_size {
                sum += l2_w[i * self.l2_size + j] * h1[i];
            }
            h2[j] = sum.max(0.0); // ReLU
        }

        // L3: l2_size -> 1, sigmoid
        let l3_w = &w[off..off + self.l2_size];
        off += self.l2_size;
        let l3_b = w[off];

        let mut logit = l3_b;
        for i in 0..self.l2_size {
            logit += l3_w[i] * h2[i];
        }

        // Sigmoid
        1.0 / (1.0 + (-logit).exp())
    }

    /// Save weights as a simple binary file: [magic, input_size, l1_size, l2_size, f32 weights...]
    fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"CNET")?;
        f.write_all(&(self.input_size as u32).to_le_bytes())?;
        f.write_all(&(self.l1_size as u32).to_le_bytes())?;
        f.write_all(&(self.l2_size as u32).to_le_bytes())?;
        for &val in &self.weights {
            f.write_all(&val.to_le_bytes())?;
        }
        Ok(())
    }

    /// Load weights from a binary file.
    #[allow(dead_code)]
    fn load(path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"CNET" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Not a CNET file",
            ));
        }
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let input_size = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let l1_size = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let l2_size = u32::from_le_bytes(buf4) as usize;

        let n = Self::param_count(input_size, l1_size, l2_size);
        let mut weights = vec![0.0f32; n];
        for val in &mut weights {
            f.read_exact(&mut buf4)?;
            *val = f32::from_le_bytes(buf4);
        }
        Ok(Self { input_size, l1_size, l2_size, weights })
    }
}

/// Evaluator that wraps a CombinedNet: extracts combined features then forward-passes.
struct CombinedNetEvaluator {
    net: CombinedNet,
}

impl CombinedNetEvaluator {
    fn new(net: CombinedNet) -> Self {
        Self { net }
    }
}

impl GameEvaluator for CombinedNetEvaluator {
    fn evaluate(&self, gs: &GameState) -> f32 {
        let features = extract_combined_features(gs);
        self.net.forward(&features)
    }
}

/// Play K games of a candidate evaluator vs an opponent and return win rate in [0, 1].
fn evaluate_generic(
    candidate: &(dyn GameEvaluator + Sync),
    opponent: &(dyn GameEvaluator + Sync),
    gs: &GameState,
    k: u32,
    seed_base: u64,
) -> f32 {
    let cand_player = Player::Evaluator(candidate);
    let opp_player = Player::Evaluator(opponent);

    let mut score = 0.0f32;
    for i in 0..k {
        let mut rng = StdRng::seed_from_u64(seed_base + i as u64);
        let result = if i % 2 == 0 {
            play_match(gs, &cand_player, &opp_player, &mut rng, 200)
        } else {
            play_match(gs, &opp_player, &cand_player, &mut rng, 200)
        };
        let cand_is_top = i % 2 == 0;
        match result {
            GameResult::Won(Owner::TopPlayer) => {
                if cand_is_top { score += 1.0; }
            }
            GameResult::Won(Owner::BottomPlayer) => {
                if !cand_is_top { score += 1.0; }
            }
            _ => { score += 0.5; }
        }
    }
    score / k as f32
}

/// Run the ES training loop with 41 combined features as input.
fn run_combined_training(
    l1_size: usize,
    l2_size: usize,
    pop_size: usize,
    games_per_eval: u32,
    sigma: f32,
    lr: f32,
    iterations: u32,
    eval_interval: u32,
    eval_games: u32,
    checkpoint_dir: &str,
    gs: &GameState,
    time_limit_secs: Option<u64>,
    resume_path: Option<&str>,
) {
    let input_size = NUM_COMBINED_FEATURES; // 41
    let dim = CombinedNet::param_count(input_size, l1_size, l2_size);

    println!("=== ES Training (Combined Features) ===");
    println!("  network: {}->{}->{}->1 ({} params)", input_size, l1_size, l2_size, dim);
    println!("  population: {} (x2 with mirroring = {})", pop_size, pop_size * 2);
    println!("  games per perturbation: {}", games_per_eval);
    println!("  sigma: {}, lr: {}", sigma, lr);
    println!("  mode: vs heuristic");
    println!("  iterations: {}", iterations);
    if let Some(tl) = time_limit_secs {
        println!("  time limit: {} seconds", tl);
    }
    println!("  eval every {} iters with {} games", eval_interval, eval_games);
    if resume_path.is_some() {
        println!("  resuming from: {}", resume_path.unwrap());
    } else {
        println!("  starting from random weights");
    }

    std::fs::create_dir_all(checkpoint_dir).expect("Failed to create checkpoint dir");

    let mut rng = StdRng::seed_from_u64(42);

    // Initialize flat weight vector
    let mut w: Vec<f32> = if let Some(path) = resume_path {
        let net = CombinedNet::load(path).expect("Failed to load combined net weights");
        assert_eq!(net.input_size, input_size, "input_size mismatch");
        assert_eq!(net.l1_size, l1_size, "l1 mismatch");
        assert_eq!(net.l2_size, l2_size, "l2 mismatch");
        net.weights
    } else {
        CombinedNet::random(input_size, l1_size, l2_size, &mut rng).weights
    };

    // Evaluate initial win rate
    {
        let init_net = CombinedNet::from_flat(w.clone(), input_size, l1_size, l2_size);
        let init_eval = CombinedNetEvaluator::new(init_net);
        let cand_player = Player::Evaluator(&init_eval);
        let heur_player = Player::Evaluator(&StaticHeuristicEvaluator::new());
        print!("  INIT: ");
        run_matches(gs, &cand_player, &heur_player, eval_games, "Combined41 vs Heuristic");
    }

    let total_start = Instant::now();
    let mut last_iter = 0u32;

    for iter in 0..iterations {
        // Check time limit
        if let Some(tl) = time_limit_secs {
            if total_start.elapsed().as_secs() >= tl {
                println!("Time limit reached ({} s), stopping at iter {}", tl, iter);
                break;
            }
        }

        last_iter = iter + 1;
        let iter_start = Instant::now();

        // Generate perturbation seeds
        let perturbation_seeds: Vec<u64> = (0..pop_size)
            .map(|_| rng.gen::<u64>())
            .collect();

        let game_seed_base: u64 = rng.gen();

        // Evaluate all perturbations in parallel
        let results: Vec<(usize, f32, f32)> = (0..pop_size * 2)
            .into_par_iter()
            .map(|idx| {
                let pert_idx = idx / 2;
                let is_positive = idx % 2 == 0;
                let pert_seed = perturbation_seeds[pert_idx];

                let mut pert_rng = StdRng::seed_from_u64(pert_seed);
                let epsilon = randn_vec(dim, &mut pert_rng);

                let perturbed: Vec<f32> = if is_positive {
                    w.iter().zip(epsilon.iter()).map(|(&wi, &ei)| wi + sigma * ei).collect()
                } else {
                    w.iter().zip(epsilon.iter()).map(|(&wi, &ei)| wi - sigma * ei).collect()
                };

                let net = CombinedNet::from_flat(perturbed, input_size, l1_size, l2_size);
                let evaluator = CombinedNetEvaluator::new(net);

                let game_seed = game_seed_base.wrapping_add(idx as u64 * 10000);
                let heuristic = StaticHeuristicEvaluator::new();
                let win_rate = evaluate_generic(&evaluator, &heuristic, gs, games_per_eval, game_seed);

                (pert_idx, win_rate, 0.0)
            })
            .collect();

        // Organize results
        let mut reward_plus = vec![0.0f32; pop_size];
        let mut reward_minus = vec![0.0f32; pop_size];
        for (i, &(pert_idx, win_rate, _)) in results.iter().enumerate() {
            if i % 2 == 0 {
                reward_plus[pert_idx] = win_rate;
            } else {
                reward_minus[pert_idx] = win_rate;
            }
        }

        // Compute gradient and update
        let scale = lr / (pop_size as f32 * sigma);
        let mut grad = vec![0.0f32; dim];

        for i in 0..pop_size {
            let diff = reward_plus[i] - reward_minus[i];
            if diff.abs() < 1e-12 { continue; }
            let mut pert_rng = StdRng::seed_from_u64(perturbation_seeds[i]);
            let epsilon = randn_vec(dim, &mut pert_rng);
            for j in 0..dim {
                grad[j] += diff * epsilon[j];
            }
        }

        for j in 0..dim {
            w[j] += scale * grad[j];
        }

        // Stats
        let avg_plus: f32 = reward_plus.iter().sum::<f32>() / pop_size as f32;
        let avg_minus: f32 = reward_minus.iter().sum::<f32>() / pop_size as f32;
        let max_wr = reward_plus.iter().chain(reward_minus.iter())
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let games_this_iter = pop_size as u32 * 2 * games_per_eval;
        println!(
            "iter {:>4}/{}: avg_wr+={:.3} avg_wr-={:.3} max_wr={:.3} ({} games in {:.1?})",
            iter + 1, iterations,
            avg_plus, avg_minus, max_wr,
            games_this_iter, iter_start.elapsed()
        );

        // Periodic evaluation + checkpoint
        if (iter + 1) % eval_interval == 0 || iter == iterations - 1 {
            let eval_net = CombinedNet::from_flat(w.clone(), input_size, l1_size, l2_size);

            let ckpt_path = format!("{}/es_combined_iter_{}.cnet", checkpoint_dir, iter + 1);
            eval_net.save(&ckpt_path).expect("Failed to save checkpoint");
            println!("  Saved checkpoint: {}", ckpt_path);

            let eval_comb = CombinedNetEvaluator::new(eval_net);
            let cand_player = Player::Evaluator(&eval_comb);
            let heur_player = Player::Evaluator(&StaticHeuristicEvaluator::new());
            print!("  EVAL: ");
            run_matches(
                gs, &cand_player, &heur_player, eval_games,
                &format!("Combined41(ES iter={}) vs Heuristic", iter + 1),
            );
        }
    }

    // Save final weights
    let final_net = CombinedNet::from_flat(w.clone(), input_size, l1_size, l2_size);
    let final_path = format!("{}/es_combined_final.cnet", checkpoint_dir);
    final_net.save(&final_path).expect("Failed to save final weights");
    println!("\nES combined training complete: {} iters in {:.1?}", last_iter, total_start.elapsed());
    println!("Final weights saved to: {}", final_path);
}

/// Create randomly initialized weights using Kaiming-like initialization.
fn random_weights(l1_size: usize, l2_size: usize, rng: &mut StdRng) -> NnueWeights {
    let rand_vec = |n: usize, fan_in: usize, rng: &mut StdRng| -> Vec<f32> {
        let scale = (2.0 / fan_in as f64).sqrt() as f32;
        (0..n).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect()
    };
    NnueWeights {
        l1_size,
        l2_size,
        l1_weight: rand_vec(NUM_FEATURES * l1_size, NUM_FEATURES, rng),
        l1_bias: vec![0.0; l1_size],
        l2_weight: rand_vec(l2_size * l1_size, l1_size, rng),
        l2_bias: vec![0.0; l2_size],
        l3_weight: rand_vec(l2_size, l2_size, rng),
        l3_bias: vec![0.0; 1],
    }
}

// ── Gaussian noise generation ────────────────────────────────────────────

/// Generate a vector of standard-normal samples using Box-Muller transform.
fn randn_vec(n: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1: f64 = rng.gen::<f64>().max(1e-30);
        let u2: f64 = rng.gen::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < n {
            out.push((r * theta.sin()) as f32);
        }
    }
    out
}

// ── Win-rate evaluation ──────────────────────────────────────────────────

/// Play K games of NNUE vs Heuristic and return win rate in [0, 1].
/// Alternates sides each game. Ties count as 0.5.
fn evaluate_perturbation(
    weights: &NnueWeights,
    opponent: &(dyn duke_training::game_setup::GameEvaluator + Sync),
    gs: &duke_rust::game::state::GameState,
    k: u32,
    seed_base: u64,
) -> f32 {
    let evaluator = NnueEvaluator::new(NnueWeights {
        l1_size: weights.l1_size,
        l2_size: weights.l2_size,
        l1_weight: weights.l1_weight.clone(),
        l1_bias: weights.l1_bias.clone(),
        l2_weight: weights.l2_weight.clone(),
        l2_bias: weights.l2_bias.clone(),
        l3_weight: weights.l3_weight.clone(),
        l3_bias: weights.l3_bias.clone(),
    });

    let nnue_player = Player::Evaluator(&evaluator);
    let opp_player = Player::Evaluator(opponent);

    let mut score = 0.0f32;
    for i in 0..k {
        let mut rng = StdRng::seed_from_u64(seed_base + i as u64);
        let result = if i % 2 == 0 {
            play_match(gs, &nnue_player, &opp_player, &mut rng, 200)
        } else {
            play_match(gs, &opp_player, &nnue_player, &mut rng, 200)
        };
        let nnue_is_top = i % 2 == 0;
        match result {
            GameResult::Won(Owner::TopPlayer) => {
                if nnue_is_top { score += 1.0; }
            }
            GameResult::Won(Owner::BottomPlayer) => {
                if !nnue_is_top { score += 1.0; }
            }
            _ => { score += 0.5; }
        }
    }
    score / k as f32
}

// ── CLI argument parsing ─────────────────────────────────────────────────

fn parse_flag<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn parse_flag_string(args: &[String], flag: &str) -> Option<String> {
    parse_flag(args, flag)
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let l1_size: usize = parse_flag(&args, "--l1").unwrap_or(256);
    let l2_size: usize = parse_flag(&args, "--l2").unwrap_or(32);
    let resume_path = parse_flag_string(&args, "--resume");
    let pop_size: usize = parse_flag(&args, "--pop").unwrap_or(50);
    let games_per_eval: u32 = parse_flag(&args, "--games").unwrap_or(10);
    let sigma: f32 = parse_flag(&args, "--sigma").unwrap_or(0.01);
    let lr: f32 = parse_flag(&args, "--lr").unwrap_or(0.01);
    let iterations: u32 = parse_flag(&args, "--iterations").unwrap_or(200);
    let eval_interval: u32 = parse_flag(&args, "--eval-interval").unwrap_or(20);
    let eval_games: u32 = parse_flag(&args, "--eval-games").unwrap_or(500);
    let checkpoint_dir = parse_flag_string(&args, "--checkpoint-dir")
        .unwrap_or_else(|| "es_checkpoints".to_string());
    let self_play = args.iter().any(|a| a == "--self-play");
    let last_layer_only = args.iter().any(|a| a == "--last-layer-only");
    let append_combined = args.iter().any(|a| a == "--append-combined");
    let input_features = parse_flag_string(&args, "--input-features")
        .unwrap_or_else(|| "nnue".to_string());
    let time_limit_secs: Option<u64> = parse_flag(&args, "--time-limit");

    // Dispatch to combined-features mode (41 inputs)
    if input_features == "combined" {
        let bag = create_bag();
        let gs = create_initial_state(&bag);
        run_combined_training(
            l1_size, l2_size, pop_size, games_per_eval,
            sigma, lr, iterations, eval_interval, eval_games,
            &checkpoint_dir, &gs, time_limit_secs,
            resume_path.as_deref(),
        );
        return;
    }

    // Dispatch to appended-input mode if requested
    if append_combined {
        let bag = create_bag();
        let gs = create_initial_state(&bag);
        run_appended_training(
            l1_size, l2_size, pop_size, games_per_eval,
            sigma, lr, iterations, eval_interval, eval_games,
            &checkpoint_dir, &gs,
        );
        return;
    }

    let dim = if last_layer_only {
        last_layer_count(l2_size)
    } else {
        weight_count(l1_size, l2_size)
    };

    println!("=== Evolutionary Strategies Training ===");
    println!("  network: {}->{}->{}->1", NUM_FEATURES, l1_size, l2_size);
    if last_layer_only {
        println!("  ** LAST-LAYER-ONLY mode: optimizing {} params (l3_weight + l3_bias) **", dim);
    } else {
        println!("  weight dimension: {}", dim);
    }
    println!("  population: {} (x2 with mirroring = {})", pop_size, pop_size * 2);
    println!("  games per perturbation: {}", games_per_eval);
    println!("  sigma: {}, lr: {}", sigma, lr);
    println!("  mode: {}", if self_play { "self-play" } else { "vs heuristic" });
    println!("  iterations: {}", iterations);
    println!("  eval every {} iters with {} games", eval_interval, eval_games);
    if resume_path.is_some() {
        println!("  resuming from: {}", resume_path.as_ref().unwrap());
    } else {
        println!("  starting from random weights");
    }

    std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoint dir");

    let bag = create_bag();
    let gs = create_initial_state(&bag);

    // Initialize weights
    let mut rng = StdRng::seed_from_u64(42);

    // In last-layer-only mode, we keep the frozen base weights separately
    // and only optimize the last layer (l3_weight + l3_bias).
    let base_weights: Option<NnueWeights> = if last_layer_only {
        if resume_path.is_none() {
            panic!("--last-layer-only requires --resume to provide frozen lower layers");
        }
        let weights = NnueWeights::load(resume_path.as_ref().unwrap())
            .expect("Failed to load NNUE weights");
        assert_eq!(weights.l1_size, l1_size, "l1 mismatch");
        assert_eq!(weights.l2_size, l2_size, "l2 mismatch");
        Some(weights)
    } else {
        None
    };

    let mut w: Vec<f32> = if last_layer_only {
        flatten_last_layer(base_weights.as_ref().unwrap())
    } else if let Some(ref path) = resume_path {
        let weights = NnueWeights::load(path).expect("Failed to load NNUE weights");
        assert_eq!(weights.l1_size, l1_size, "l1 mismatch");
        assert_eq!(weights.l2_size, l2_size, "l2 mismatch");
        flatten_weights(&weights)
    } else {
        let weights = random_weights(l1_size, l2_size, &mut rng);
        flatten_weights(&weights)
    };

    // Helper to reconstruct full weights from the optimized vector
    let reconstruct_weights = |w: &[f32]| -> NnueWeights {
        if last_layer_only {
            unflatten_last_layer(w, base_weights.as_ref().unwrap())
        } else {
            unflatten_weights(w, l1_size, l2_size)
        }
    };

    // Evaluate initial win rate
    {
        let init_weights = reconstruct_weights(&w);
        let init_eval = NnueEvaluator::new(init_weights);
        let nnue_player = Player::Evaluator(&init_eval);
        let heur_player = Player::Evaluator(&StaticHeuristicEvaluator::new());
        print!("  INIT: ");
        run_matches(&gs, &nnue_player, &heur_player, eval_games, "NNUE vs Heuristic");
    }

    let total_start = Instant::now();

    for iter in 0..iterations {
        let iter_start = Instant::now();

        // Generate perturbation seeds (one per population member)
        let perturbation_seeds: Vec<u64> = (0..pop_size)
            .map(|_| rng.gen::<u64>())
            .collect();

        // Game seed base for this iteration (each perturbation x game gets unique seed)
        let game_seed_base: u64 = rng.gen();

        // Evaluate all perturbations in parallel (both +sigma and -sigma)
        // Each element: (perturbation_index, is_positive, win_rate)
        let results: Vec<(usize, f32, f32)> = (0..pop_size * 2)
            .into_par_iter()
            .map(|idx| {
                let pert_idx = idx / 2;
                let is_positive = idx % 2 == 0;
                let pert_seed = perturbation_seeds[pert_idx];

                // Regenerate the same epsilon from the seed
                let mut pert_rng = StdRng::seed_from_u64(pert_seed);
                let epsilon = randn_vec(dim, &mut pert_rng);

                // Create perturbed weights
                let perturbed: Vec<f32> = if is_positive {
                    w.iter().zip(epsilon.iter()).map(|(&wi, &ei)| wi + sigma * ei).collect()
                } else {
                    w.iter().zip(epsilon.iter()).map(|(&wi, &ei)| wi - sigma * ei).collect()
                };

                let weights = if last_layer_only {
                    unflatten_last_layer(&perturbed, base_weights.as_ref().unwrap())
                } else {
                    unflatten_weights(&perturbed, l1_size, l2_size)
                };

                // Unique game seed per perturbation
                let game_seed = game_seed_base.wrapping_add(idx as u64 * 10000);
                let win_rate = if self_play {
                    // Self-play: perturbation plays against unperturbed base weights
                    let base_weights_copy = reconstruct_weights(&w);
                    let base_eval = NnueEvaluator::new(base_weights_copy);
                    evaluate_perturbation(&weights, &base_eval, &gs, games_per_eval, game_seed)
                } else {
                    let heuristic = StaticHeuristicEvaluator::new();
                    evaluate_perturbation(&weights, &heuristic, &gs, games_per_eval, game_seed)
                };

                (pert_idx, win_rate, 0.0) // third field unused, identified by idx parity
            })
            .collect();

        // Organize results: positive[i] and negative[i]
        let mut reward_plus = vec![0.0f32; pop_size];
        let mut reward_minus = vec![0.0f32; pop_size];
        for (i, &(pert_idx, win_rate, _)) in results.iter().enumerate() {
            if i % 2 == 0 {
                reward_plus[pert_idx] = win_rate;
            } else {
                reward_minus[pert_idx] = win_rate;
            }
        }

        // Compute gradient estimate and update weights
        // w += lr / (N * sigma) * sum((reward_plus_i - reward_minus_i) * epsilon_i)
        let scale = lr / (pop_size as f32 * sigma);
        let mut grad = vec![0.0f32; dim];

        for i in 0..pop_size {
            let diff = reward_plus[i] - reward_minus[i];
            if diff.abs() < 1e-12 {
                continue; // Skip zero-contribution perturbations
            }
            // Regenerate epsilon_i
            let mut pert_rng = StdRng::seed_from_u64(perturbation_seeds[i]);
            let epsilon = randn_vec(dim, &mut pert_rng);

            for j in 0..dim {
                grad[j] += diff * epsilon[j];
            }
        }

        // Apply update
        for j in 0..dim {
            w[j] += scale * grad[j];
        }

        // Compute stats for this iteration
        let avg_plus: f32 = reward_plus.iter().sum::<f32>() / pop_size as f32;
        let avg_minus: f32 = reward_minus.iter().sum::<f32>() / pop_size as f32;
        let max_wr = reward_plus.iter().chain(reward_minus.iter())
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let games_this_iter = pop_size as u32 * 2 * games_per_eval;
        println!(
            "iter {:>4}/{}: avg_wr+={:.3} avg_wr-={:.3} max_wr={:.3} ({} games in {:.1?})",
            iter + 1, iterations,
            avg_plus, avg_minus, max_wr,
            games_this_iter, iter_start.elapsed()
        );

        // Periodic evaluation + checkpoint
        if (iter + 1) % eval_interval == 0 || iter == iterations - 1 {
            let eval_weights = reconstruct_weights(&w);

            // Save checkpoint
            let ckpt_path = format!("{}/es_iter_{}.nnue", checkpoint_dir, iter + 1);
            eval_weights.save(&ckpt_path).expect("Failed to save checkpoint");
            println!("  Saved checkpoint: {}", ckpt_path);

            // Benchmark
            let eval_nnue = NnueEvaluator::new(eval_weights);
            let nnue_player = Player::Evaluator(&eval_nnue);
            let heur_player = Player::Evaluator(&StaticHeuristicEvaluator::new());
            print!("  EVAL: ");
            run_matches(
                &gs, &nnue_player, &heur_player, eval_games,
                &format!("NNUE(ES iter={}) vs Heuristic", iter + 1),
            );
        }
    }

    // Save final weights
    let final_weights = reconstruct_weights(&w);
    let final_path = format!("{}/es_final.nnue", checkpoint_dir);
    final_weights.save(&final_path).expect("Failed to save final weights");
    println!("\nES training complete: {} iterations in {:.1?}", iterations, total_start.elapsed());
    println!("Final weights saved to: {}", final_path);
}
