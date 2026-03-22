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

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use duke_rust::game::state::GameResult;
use duke_rust::game::tile::Owner;

use duke_training::game_setup::{
    create_bag, create_initial_state, StaticHeuristicEvaluator,
};
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
