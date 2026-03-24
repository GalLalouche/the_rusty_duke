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
//!                 [--layers 64,64,32]  — configurable hidden layer sizes
//!                 [--pop 50] [--games 10] [--sigma 0.01] [--lr 0.01]
//!                 [--iterations 200] [--eval-interval 20] [--eval-games 500]
//!                 [--checkpoint-dir <dir>] [--time-limit 3600]
//!                 [--append-combined]  — use 1147-input network (1106 NNUE + 41 combined)
//!                 [--input-features combined]  — use 41 combined features only
//!                 [--input-features guard]  — use 65 features (24 expensive + 41 combined)

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use duke_rust::game::ai::player::ArtificialPlayer;
use duke_rust::game::ai::stupid_sync_ai::StupidSyncAi;
use duke_rust::game::state::{GameResult, GameState};
use duke_rust::game::tile::Owner;

use duke_training::cli::parse_flag;
use duke_training::encoding::TOTAL_FEATURES;
use duke_training::game_setup::{
    create_bag, create_initial_state, greedy_move, GameEvaluator, StaticHeuristicEvaluator,
};
use duke_training::learned_heuristic::NUM_COMBINED_FEATURES;
use duke_training::match_runner::{run_matches, Player};
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

// Old AppendedNnueEvaluator code removed; appended mode now uses GenericMlp via run_generic_sparse_training.

use duke_training::generic_mlp::{
    GenericMlp, CombinedNetEvaluator, GuardFeatureEvaluator,
    GenericNnueEvaluator, GenericAppendedEvaluator, load_opponent,
    NUM_GUARD_ALL_FEATURES,
};



/// Play a single match where the candidate always plays greedily but the opponent
/// uses epsilon-greedy: with probability `opponent_epsilon` it makes a random move,
/// otherwise it picks its best greedy move.
fn play_match_with_epsilon(
    gs: &GameState,
    candidate: &(dyn GameEvaluator + Sync),
    opponent: &(dyn GameEvaluator + Sync),
    candidate_is_top: bool,
    rng: &mut StdRng,
    max_turns: u32,
    opponent_epsilon: f32,
) -> GameResult {
    let ai = StupidSyncAi {};
    let mut game = gs.clone();
    let mut turns = 0u32;

    loop {
        match game.game_result() {
            GameResult::Ongoing => {
                if turns >= max_turns {
                    return GameResult::Tie;
                }
                let current = game.current_player_turn();
                let is_candidate = (current == Owner::TopPlayer) == candidate_is_top;
                if is_candidate {
                    // Candidate always plays greedily
                    let mv = greedy_move(&game, candidate, rng);
                    mv.play(&mut game, rng);
                } else {
                    // Opponent: epsilon-greedy
                    if opponent_epsilon > 0.0 && rng.gen::<f32>() < opponent_epsilon {
                        ai.play_next_move(rng, &mut game);
                    } else {
                        let mv = greedy_move(&game, opponent, rng);
                        mv.play(&mut game, rng);
                    }
                }
                turns += 1;
            }
            result => return result,
        }
    }
}

/// Play K games of a candidate evaluator vs an opponent and return win rate in [0, 1].
/// `max_turns` controls per-game turn limit.
/// `opponent_epsilon` controls how often the opponent makes a random move (0.0 = full strength).
fn evaluate_generic(
    candidate: &(dyn GameEvaluator + Sync),
    opponent: &(dyn GameEvaluator + Sync),
    gs: &GameState,
    k: u32,
    seed_base: u64,
    max_turns: u32,
    opponent_epsilon: f32,
) -> f32 {
    let mut score = 0.0f32;
    for i in 0..k {
        let mut rng = StdRng::seed_from_u64(seed_base + i as u64);
        let cand_is_top = i % 2 == 0;
        let result = play_match_with_epsilon(
            gs, candidate, opponent, cand_is_top,
            &mut rng, max_turns, opponent_epsilon,
        );
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

/// Which dense feature set to use for the dense-input training paths.
#[derive(Clone, Copy, PartialEq)]
enum DenseFeatureMode {
    /// 41 cheap combined features (no guard checking)
    Combined,
    /// 65 features: 24 expensive guard + 41 combined (richest feature set)
    Guard,
}

impl DenseFeatureMode {
    fn input_size(self) -> usize {
        match self {
            DenseFeatureMode::Combined => NUM_COMBINED_FEATURES, // 41
            DenseFeatureMode::Guard => NUM_GUARD_ALL_FEATURES,    // 65
        }
    }

    fn label(self) -> &'static str {
        match self {
            DenseFeatureMode::Combined => "Combined41",
            DenseFeatureMode::Guard => "Guard65",
        }
    }
}

/// Run the ES training loop with dense features (combined 41 or guard 65) as input.
fn run_dense_training(
    feature_mode: DenseFeatureMode,
    hidden_layers: &[usize],
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
    seed: u64,
    initial_opponent_epsilon: f32,
    training_opponent: &TrainingOpponent,
) {
    let input_size = feature_mode.input_size();
    let dim = GenericMlp::param_count(input_size, hidden_layers);

    let mode_label = feature_mode.label();
    println!("=== ES Training ({} Features) ===", mode_label);
    let tmp_net = GenericMlp::from_flat(vec![0.0; dim], input_size, hidden_layers.to_vec());
    println!("  network: {} ({} params)", tmp_net.arch_string(), dim);
    println!("  population: {} (x2 with mirroring = {})", pop_size, pop_size * 2);
    println!("  games per perturbation: {}", games_per_eval);
    println!("  sigma: {}, lr: {}", sigma, lr);
    println!("  mode: vs heuristic");
    println!("  opponent epsilon: {:.2}", initial_opponent_epsilon);
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

    let mut rng = StdRng::seed_from_u64(seed);

    // Initialize flat weight vector
    let mut w: Vec<f32> = if let Some(path) = resume_path {
        let net = GenericMlp::load(path).expect("Failed to load combined net weights");
        assert_eq!(net.input_size, input_size, "input_size mismatch");
        assert_eq!(net.hidden_layers, hidden_layers, "hidden_layers mismatch");
        net.weights
    } else {
        GenericMlp::random(input_size, hidden_layers.to_vec(), &mut rng).weights
    };

    // Adaptive opponent epsilon
    let mut opponent_epsilon = initial_opponent_epsilon;

    // Evaluate initial win rate
    {
        let init_net = GenericMlp::from_flat(w.clone(), input_size, hidden_layers.to_vec());
        let heur_player = Player::Evaluator(&StaticHeuristicEvaluator::new());
        print!("  INIT: ");
        match feature_mode {
            DenseFeatureMode::Combined => {
                let init_eval = CombinedNetEvaluator::new(init_net);
                let cand_player = Player::Evaluator(&init_eval);
                run_matches(gs, &cand_player, &heur_player, eval_games,
                    &format!("{} vs Heuristic", mode_label));
            }
            DenseFeatureMode::Guard => {
                let init_eval = GuardFeatureEvaluator::new(init_net);
                let cand_player = Player::Evaluator(&init_eval);
                run_matches(gs, &cand_player, &heur_player, eval_games,
                    &format!("{} vs Heuristic", mode_label));
            }
        };
    }

    let total_start = Instant::now();
    let mut last_iter = 0u32;

    // Adaptive sigma: increase when stuck, reset when improving
    let sigma_base = sigma;
    let mut sigma_current = sigma;
    let mut best_eval_wr = 0.0f32;
    let mut evals_without_improvement = 0u32;
    let sigma_patience = 3u32;
    let sigma_grow = 2.0f32;
    let sigma_max = sigma_base * 8.0;

    // Adam optimizer state
    let mut adam_m = vec![0.0f32; dim]; // first moment
    let mut adam_v = vec![0.0f32; dim]; // second moment
    let adam_beta1 = 0.9f32;
    let adam_beta2 = 0.999f32;
    let adam_eps = 1e-8f32;

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
        let sigma_snap = sigma_current; // capture for closure
        let opp_eps_snap = opponent_epsilon; // capture for closure
        let results: Vec<(usize, f32, f32)> = (0..pop_size * 2)
            .into_par_iter()
            .map(|idx| {
                let pert_idx = idx / 2;
                let is_positive = idx % 2 == 0;
                let pert_seed = perturbation_seeds[pert_idx];

                let mut pert_rng = StdRng::seed_from_u64(pert_seed);
                let epsilon = randn_vec(dim, &mut pert_rng);

                let perturbed: Vec<f32> = if is_positive {
                    w.iter().zip(epsilon.iter()).map(|(&wi, &ei)| wi + sigma_snap * ei).collect()
                } else {
                    w.iter().zip(epsilon.iter()).map(|(&wi, &ei)| wi - sigma_snap * ei).collect()
                };

                let net = GenericMlp::from_flat(perturbed, input_size, hidden_layers.to_vec());

                let game_seed = game_seed_base.wrapping_add(idx as u64 * 10000);
                // Guard features are expensive (guard checking per eval), use 50-turn cap
                let train_max_turns = match feature_mode {
                    DenseFeatureMode::Guard => 50,
                    DenseFeatureMode::Combined => 200,
                };
                let win_rate = match training_opponent {
                    TrainingOpponent::Random => {
                        let fallback = StaticHeuristicEvaluator::new();
                        match feature_mode {
                            DenseFeatureMode::Combined => {
                                let evaluator = CombinedNetEvaluator::new(net);
                                evaluate_generic(&evaluator, &fallback, gs, games_per_eval, game_seed, train_max_turns, 1.0)
                            }
                            DenseFeatureMode::Guard => {
                                let evaluator = GuardFeatureEvaluator::new(net);
                                evaluate_generic(&evaluator, &fallback, gs, games_per_eval, game_seed, train_max_turns, 1.0)
                            }
                        }
                    }
                    TrainingOpponent::Eval(opp_eval) => {
                        match feature_mode {
                            DenseFeatureMode::Combined => {
                                let evaluator = CombinedNetEvaluator::new(net);
                                evaluate_generic(&evaluator, *opp_eval, gs, games_per_eval, game_seed, train_max_turns, opp_eps_snap)
                            }
                            DenseFeatureMode::Guard => {
                                let evaluator = GuardFeatureEvaluator::new(net);
                                evaluate_generic(&evaluator, *opp_eval, gs, games_per_eval, game_seed, train_max_turns, opp_eps_snap)
                            }
                        }
                    }
                };

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

        // Compute gradient
        let grad_scale = 1.0 / (pop_size as f32 * sigma_current);
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
            grad[j] *= grad_scale;
        }

        // Adam update (bias correction factors hoisted out of inner loop)
        let t = (iter + 1) as f32;
        let bc1 = 1.0 / (1.0 - adam_beta1.powf(t));
        let bc2 = 1.0 / (1.0 - adam_beta2.powf(t));
        for j in 0..dim {
            adam_m[j] = adam_beta1 * adam_m[j] + (1.0 - adam_beta1) * grad[j];
            adam_v[j] = adam_beta2 * adam_v[j] + (1.0 - adam_beta2) * grad[j] * grad[j];
            let m_hat = adam_m[j] * bc1;
            let v_hat = adam_v[j] * bc2;
            w[j] += lr * m_hat / (v_hat.sqrt() + adam_eps);
        }

        // Stats
        let avg_plus: f32 = reward_plus.iter().sum::<f32>() / pop_size as f32;
        let avg_minus: f32 = reward_minus.iter().sum::<f32>() / pop_size as f32;
        let max_wr = reward_plus.iter().chain(reward_minus.iter())
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let games_this_iter = pop_size as u32 * 2 * games_per_eval;
        println!(
            "iter {:>4}/{}: avg_wr+={:.3} avg_wr-={:.3} max_wr={:.3} sigma={:.4} opp_eps={:.2} ({} games in {:.1?})",
            iter + 1, iterations,
            avg_plus, avg_minus, max_wr, sigma_current, opponent_epsilon,
            games_this_iter, iter_start.elapsed()
        );

        // Periodic evaluation + checkpoint
        if (iter + 1) % eval_interval == 0 || iter == iterations - 1 {
            let eval_net = GenericMlp::from_flat(w.clone(), input_size, hidden_layers.to_vec());

            let ckpt_path = format!("{}/es_{}_iter_{}.gmlp", checkpoint_dir, mode_label.to_lowercase(), iter + 1);
            eval_net.save(&ckpt_path).expect("Failed to save checkpoint");
            println!("  Saved checkpoint: {}", ckpt_path);

            let heur_player = Player::Evaluator(&StaticHeuristicEvaluator::new());
            print!("  EVAL: ");
            let eval_result = match feature_mode {
                DenseFeatureMode::Combined => {
                    let eval_comb = CombinedNetEvaluator::new(eval_net);
                    let cand_player = Player::Evaluator(&eval_comb);
                    run_matches(
                        gs, &cand_player, &heur_player, eval_games,
                        &format!("{}(ES iter={}) vs Heuristic", mode_label, iter + 1),
                    )
                }
                DenseFeatureMode::Guard => {
                    let eval_guard = GuardFeatureEvaluator::new(eval_net);
                    let cand_player = Player::Evaluator(&eval_guard);
                    run_matches(
                        gs, &cand_player, &heur_player, eval_games,
                        &format!("{}(ES iter={}) vs Heuristic", mode_label, iter + 1),
                    )
                }
            };

            // Also benchmark vs training opponent
            {
                let eval_net2 = GenericMlp::from_flat(w.clone(), input_size, hidden_layers.to_vec());
                let opp_player = match training_opponent {
                    TrainingOpponent::Random => Player::Random,
                    TrainingOpponent::Eval(e) => Player::Evaluator(*e),
                };
                print!("  VS_OPP: ");
                match feature_mode {
                    DenseFeatureMode::Combined => {
                        let eval_comb = CombinedNetEvaluator::new(eval_net2);
                        let cand_player = Player::Evaluator(&eval_comb);
                        run_matches(gs, &cand_player, &opp_player, eval_games,
                            &format!("{}(ES iter={}) vs TrainOpp", mode_label, iter + 1));
                    }
                    DenseFeatureMode::Guard => {
                        let eval_guard = GuardFeatureEvaluator::new(eval_net2);
                        let cand_player = Player::Evaluator(&eval_guard);
                        run_matches(gs, &cand_player, &opp_player, eval_games,
                            &format!("{}(ES iter={}) vs TrainOpp", mode_label, iter + 1));
                    }
                };
            }

            // Adaptive sigma: track improvement
            let eval_wr = eval_result.player_a_wins as f32 / eval_games as f32;
            if eval_wr > best_eval_wr + 0.01 {
                best_eval_wr = eval_wr;
                evals_without_improvement = 0;
                sigma_current = sigma_base; // reset to base on improvement
            } else {
                evals_without_improvement += 1;
                if evals_without_improvement >= sigma_patience {
                    sigma_current = (sigma_current * sigma_grow).min(sigma_max);
                    println!("  Sigma adapted: {:.4} (no improvement for {} evals)", sigma_current, evals_without_improvement);
                }
            }

            // Adaptive opponent epsilon — based on TRAINING win rate, not eval
            let train_wr_pct = avg_plus * 100.0;
            let old_opp_eps = opponent_epsilon;
            if train_wr_pct > 60.0 {
                opponent_epsilon = (opponent_epsilon - 0.05).max(0.0);
            } else if train_wr_pct < 30.0 {
                opponent_epsilon = (opponent_epsilon + 0.05).min(0.5);
            }
            if (old_opp_eps - opponent_epsilon).abs() > 0.001 {
                println!(
                    "  Opponent epsilon: {:.2} -> {:.2} (training wr was {:.1}%)",
                    old_opp_eps, opponent_epsilon, train_wr_pct,
                );
            }
        }
    }

    // Save final weights
    let final_net = GenericMlp::from_flat(w.clone(), input_size, hidden_layers.to_vec());
    let final_path = format!("{}/es_{}_final.gmlp", checkpoint_dir, mode_label.to_lowercase());
    final_net.save(&final_path).expect("Failed to save final weights");
    println!("\nES {} training complete: {} iters in {:.1?}", mode_label, last_iter, total_start.elapsed());
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

/// Run the ES training loop using GenericMlp with sparse NNUE features.
/// `input_size` should be NUM_FEATURES (1106) for standard or APPENDED_INPUT_SIZE (1147) for appended.
/// `include_combined` controls whether combined features are appended.
fn run_generic_sparse_training(
    input_size: usize,
    include_combined: bool,
    hidden_layers: &[usize],
    pop_size: usize,
    games_per_eval: u32,
    sigma: f32,
    lr: f32,
    iterations: u32,
    eval_interval: u32,
    eval_games: u32,
    checkpoint_dir: &str,
    gs: &GameState,
    initial_opponent_epsilon: f32,
    time_limit_secs: Option<u64>,
    resume_path: Option<&str>,
    seed: u64,
    training_opponent: &TrainingOpponent,
) {
    let dim = GenericMlp::param_count(input_size, hidden_layers);
    let mode_name = if include_combined { "Appended" } else { "NNUE" };

    let tmp_net = GenericMlp::from_flat(vec![0.0; dim], input_size, hidden_layers.to_vec());
    println!("=== ES Training ({} Features, GenericMlp) ===", mode_name);
    println!("  network: {} ({} params)", tmp_net.arch_string(), dim);
    println!("  population: {} (x2 with mirroring = {})", pop_size, pop_size * 2);
    println!("  games per perturbation: {}", games_per_eval);
    println!("  sigma: {}, lr: {}", sigma, lr);
    println!("  mode: vs heuristic");
    println!("  opponent epsilon: {:.2}", initial_opponent_epsilon);
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

    let mut rng = StdRng::seed_from_u64(seed);

    // Initialize flat weight vector
    let mut w: Vec<f32> = if let Some(path) = resume_path {
        let net = GenericMlp::load(path).expect("Failed to load weights");
        assert_eq!(net.input_size, input_size, "input_size mismatch");
        assert_eq!(net.hidden_layers, hidden_layers, "hidden_layers mismatch");
        net.weights
    } else {
        GenericMlp::random(input_size, hidden_layers.to_vec(), &mut rng).weights
    };

    // Adaptive opponent epsilon
    let mut opponent_epsilon = initial_opponent_epsilon;

    // Evaluate initial win rate
    {
        let init_net = GenericMlp::from_flat(w.clone(), input_size, hidden_layers.to_vec());
        if include_combined {
            let eval = GenericAppendedEvaluator { net: init_net };
            let cand_player = Player::Evaluator(&eval);
            let heur_player = Player::Evaluator(&StaticHeuristicEvaluator::new());
            print!("  INIT: ");
            run_matches(gs, &cand_player, &heur_player, eval_games, &format!("{} vs Heuristic", mode_name));
        } else {
            let eval = GenericNnueEvaluator { net: init_net };
            let cand_player = Player::Evaluator(&eval);
            let heur_player = Player::Evaluator(&StaticHeuristicEvaluator::new());
            print!("  INIT: ");
            run_matches(gs, &cand_player, &heur_player, eval_games, &format!("{} vs Heuristic", mode_name));
        }
    }

    let total_start = Instant::now();
    let mut last_iter = 0u32;

    // Adaptive sigma: increase when stuck, reset when improving
    let sigma_base = sigma;
    let mut sigma_current = sigma;
    let mut best_eval_wr = 0.0f32;
    let mut evals_without_improvement = 0u32;
    let sigma_patience = 3u32;
    let sigma_grow = 2.0f32;
    let sigma_max = sigma_base * 8.0;

    // Adam optimizer state
    let mut adam_m = vec![0.0f32; dim]; // first moment
    let mut adam_v = vec![0.0f32; dim]; // second moment
    let adam_beta1 = 0.9f32;
    let adam_beta2 = 0.999f32;
    let adam_eps = 1e-8f32;

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

        let perturbation_seeds: Vec<u64> = (0..pop_size)
            .map(|_| rng.gen::<u64>())
            .collect();
        let game_seed_base: u64 = rng.gen();

        // Evaluate all perturbations in parallel
        let sigma_snap = sigma_current; // capture for closure
        let opp_eps_snap = opponent_epsilon; // capture for closure
        let results: Vec<(usize, f32, f32)> = (0..pop_size * 2)
            .into_par_iter()
            .map(|idx| {
                let pert_idx = idx / 2;
                let is_positive = idx % 2 == 0;
                let pert_seed = perturbation_seeds[pert_idx];

                let mut pert_rng = StdRng::seed_from_u64(pert_seed);
                let epsilon = randn_vec(dim, &mut pert_rng);

                let perturbed: Vec<f32> = if is_positive {
                    w.iter().zip(epsilon.iter()).map(|(&wi, &ei)| wi + sigma_snap * ei).collect()
                } else {
                    w.iter().zip(epsilon.iter()).map(|(&wi, &ei)| wi - sigma_snap * ei).collect()
                };

                let net = GenericMlp::from_flat(perturbed, input_size, hidden_layers.to_vec());
                let game_seed = game_seed_base.wrapping_add(idx as u64 * 10000);

                // Appended mode uses 50-turn cap because combined feature extraction
                // is expensive (involves full move generation per eval).
                let train_max_turns = if include_combined { 50 } else { 200 };
                let win_rate = match training_opponent {
                    TrainingOpponent::Random => {
                        let fallback = StaticHeuristicEvaluator::new();
                        if include_combined {
                            let evaluator = GenericAppendedEvaluator { net };
                            evaluate_generic(&evaluator, &fallback, gs, games_per_eval, game_seed, train_max_turns, 1.0)
                        } else {
                            let evaluator = GenericNnueEvaluator { net };
                            evaluate_generic(&evaluator, &fallback, gs, games_per_eval, game_seed, train_max_turns, 1.0)
                        }
                    }
                    TrainingOpponent::Eval(opp_eval) => {
                        if include_combined {
                            let evaluator = GenericAppendedEvaluator { net };
                            evaluate_generic(&evaluator, *opp_eval, gs, games_per_eval, game_seed, train_max_turns, opp_eps_snap)
                        } else {
                            let evaluator = GenericNnueEvaluator { net };
                            evaluate_generic(&evaluator, *opp_eval, gs, games_per_eval, game_seed, train_max_turns, opp_eps_snap)
                        }
                    }
                };

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

        // Compute gradient
        let grad_scale = 1.0 / (pop_size as f32 * sigma_current);
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
            grad[j] *= grad_scale;
        }

        // Adam update (bias correction factors hoisted out of inner loop)
        let t = (iter + 1) as f32;
        let bc1 = 1.0 / (1.0 - adam_beta1.powf(t));
        let bc2 = 1.0 / (1.0 - adam_beta2.powf(t));
        for j in 0..dim {
            adam_m[j] = adam_beta1 * adam_m[j] + (1.0 - adam_beta1) * grad[j];
            adam_v[j] = adam_beta2 * adam_v[j] + (1.0 - adam_beta2) * grad[j] * grad[j];
            let m_hat = adam_m[j] * bc1;
            let v_hat = adam_v[j] * bc2;
            w[j] += lr * m_hat / (v_hat.sqrt() + adam_eps);
        }

        let avg_plus: f32 = reward_plus.iter().sum::<f32>() / pop_size as f32;
        let avg_minus: f32 = reward_minus.iter().sum::<f32>() / pop_size as f32;
        let max_wr = reward_plus.iter().chain(reward_minus.iter())
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let games_this_iter = pop_size as u32 * 2 * games_per_eval;
        println!(
            "iter {:>4}/{}: avg_wr+={:.3} avg_wr-={:.3} max_wr={:.3} sigma={:.4} opp_eps={:.2} ({} games in {:.1?})",
            iter + 1, iterations,
            avg_plus, avg_minus, max_wr, sigma_current, opponent_epsilon,
            games_this_iter, iter_start.elapsed()
        );

        if (iter + 1) % eval_interval == 0 || iter == iterations - 1 {
            let eval_net = GenericMlp::from_flat(w.clone(), input_size, hidden_layers.to_vec());

            let ckpt_path = format!("{}/es_iter_{}.gmlp", checkpoint_dir, iter + 1);
            eval_net.save(&ckpt_path).expect("Failed to save checkpoint");
            println!("  Saved checkpoint: {}", ckpt_path);

            let eval_result = if include_combined {
                let eval = GenericAppendedEvaluator { net: eval_net };
                let cand_player = Player::Evaluator(&eval);
                let heur_player = Player::Evaluator(&StaticHeuristicEvaluator::new());
                print!("  EVAL: ");
                run_matches(
                    gs, &cand_player, &heur_player, eval_games,
                    &format!("{}(ES iter={}) vs Heuristic", mode_name, iter + 1),
                )
            } else {
                let eval = GenericNnueEvaluator { net: eval_net };
                let cand_player = Player::Evaluator(&eval);
                let heur_player = Player::Evaluator(&StaticHeuristicEvaluator::new());
                print!("  EVAL: ");
                run_matches(
                    gs, &cand_player, &heur_player, eval_games,
                    &format!("{}(ES iter={}) vs Heuristic", mode_name, iter + 1),
                )
            };

            // Adaptive sigma: track improvement
            let eval_wr = eval_result.player_a_wins as f32 / eval_games as f32;
            if eval_wr > best_eval_wr + 0.01 {
                best_eval_wr = eval_wr;
                evals_without_improvement = 0;
                sigma_current = sigma_base; // reset to base on improvement
            } else {
                evals_without_improvement += 1;
                if evals_without_improvement >= sigma_patience {
                    sigma_current = (sigma_current * sigma_grow).min(sigma_max);
                    println!("  Sigma adapted: {:.4} (no improvement for {} evals)", sigma_current, evals_without_improvement);
                }
            }

            // Adaptive opponent epsilon — based on TRAINING win rate, not eval
            let train_wr_pct = avg_plus * 100.0;
            let old_opp_eps = opponent_epsilon;
            if train_wr_pct > 60.0 {
                opponent_epsilon = (opponent_epsilon - 0.05).max(0.0);
            } else if train_wr_pct < 30.0 {
                opponent_epsilon = (opponent_epsilon + 0.05).min(0.5);
            }
            if (old_opp_eps - opponent_epsilon).abs() > 0.001 {
                println!(
                    "  Opponent epsilon: {:.2} -> {:.2} (training wr was {:.1}%)",
                    old_opp_eps, opponent_epsilon, train_wr_pct,
                );
            }
        }
    }

    // Save final weights
    let final_net = GenericMlp::from_flat(w.clone(), input_size, hidden_layers.to_vec());
    let final_path = format!("{}/es_final.gmlp", checkpoint_dir);
    final_net.save(&final_path).expect("Failed to save final weights");
    println!("\nES {} training complete: {} iters in {:.1?}", mode_name, last_iter, total_start.elapsed());
    println!("Final weights saved to: {}", final_path);
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

/// Play K games of NNUE vs opponent and return win rate in [0, 1].
/// Delegates to `evaluate_generic` after wrapping the weights in an evaluator.
fn evaluate_perturbation(
    weights: NnueWeights,
    opponent: &(dyn duke_training::game_setup::GameEvaluator + Sync),
    gs: &duke_rust::game::state::GameState,
    k: u32,
    seed_base: u64,
    opponent_epsilon: f32,
) -> f32 {
    let evaluator = NnueEvaluator::new(weights);
    evaluate_generic(&evaluator, opponent, gs, k, seed_base, 200, opponent_epsilon)
}

/// Describes the training opponent for use in parallel closures.
/// Either a shared reference to a boxed evaluator, or "random" / "base" keywords.
enum TrainingOpponent<'a> {
    /// Use `Player::Random` (no evaluator needed).
    Random,
    /// Use this evaluator reference as the opponent.
    Eval(&'a (dyn GameEvaluator + Sync)),
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let l1_size: usize = parse_flag(&args, "--l1").unwrap_or(256);
    let l2_size: usize = parse_flag(&args, "--l2").unwrap_or(32);
    let resume_path = parse_flag::<String>(&args, "--resume");
    let pop_size: usize = parse_flag(&args, "--pop").unwrap_or(50);
    let games_per_eval: u32 = parse_flag(&args, "--games").unwrap_or(10);
    let sigma: f32 = parse_flag(&args, "--sigma").unwrap_or(0.01);
    let lr: f32 = parse_flag(&args, "--lr").unwrap_or(0.01);
    let iterations: u32 = parse_flag(&args, "--iterations").unwrap_or(200);
    let eval_interval: u32 = parse_flag(&args, "--eval-interval").unwrap_or(20);
    let eval_games: u32 = parse_flag(&args, "--eval-games").unwrap_or(500);
    let checkpoint_dir = parse_flag::<String>(&args, "--checkpoint-dir")
        .unwrap_or_else(|| "es_checkpoints".to_string());
    let self_play = args.iter().any(|a| a == "--self-play");
    let last_layer_only = args.iter().any(|a| a == "--last-layer-only");
    let append_combined = args.iter().any(|a| a == "--append-combined");
    let input_features = parse_flag::<String>(&args, "--input-features")
        .unwrap_or_else(|| "nnue".to_string());
    let time_limit_secs: Option<u64> = parse_flag(&args, "--time-limit");
    let seed: u64 = parse_flag(&args, "--seed").unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });
    let opponent_epsilon: f32 = parse_flag(&args, "--opponent-epsilon").unwrap_or(0.0);

    // Parse --opponent flag: path to a saved model, "base", or "random"
    let opponent_spec = parse_flag::<String>(&args, "--opponent")
        .unwrap_or_else(|| "base".to_string());

    let (opponent_box, opponent_desc) = load_opponent(&opponent_spec);
    let training_opponent: TrainingOpponent = if opponent_box.is_none() {
        TrainingOpponent::Random
    } else {
        TrainingOpponent::Eval(opponent_box.as_ref().unwrap().as_ref())
    };

    println!("Training opponent: {}", opponent_desc);

    // Parse --layers flag: comma-separated hidden layer sizes (e.g. "64,64,32")
    let layers_str: Option<String> = parse_flag::<String>(&args, "--layers");
    let hidden_layers: Vec<usize> = if let Some(ref s) = layers_str {
        s.split(',')
            .map(|x| x.trim().parse::<usize>().expect("Invalid layer size in --layers"))
            .collect()
    } else {
        vec![l1_size, l2_size]
    };

    if hidden_layers.is_empty() {
        panic!("--layers must specify at least one hidden layer size");
    }

    // Dispatch to combined-features mode (41 inputs)
    if input_features == "combined" {
        let bag = create_bag();
        let gs = create_initial_state(&bag);
        run_dense_training(
            DenseFeatureMode::Combined,
            &hidden_layers, pop_size, games_per_eval,
            sigma, lr, iterations, eval_interval, eval_games,
            &checkpoint_dir, &gs, time_limit_secs,
            resume_path.as_deref(), seed, opponent_epsilon,
            &training_opponent,
        );
        return;
    }

    // Dispatch to guard-features mode (65 inputs: 24 expensive + 41 combined)
    if input_features == "guard" {
        let bag = create_bag();
        let gs = create_initial_state(&bag);
        run_dense_training(
            DenseFeatureMode::Guard,
            &hidden_layers, pop_size, games_per_eval,
            sigma, lr, iterations, eval_interval, eval_games,
            &checkpoint_dir, &gs, time_limit_secs,
            resume_path.as_deref(), seed, opponent_epsilon,
            &training_opponent,
        );
        return;
    }

    // Dispatch to appended-input mode if requested (uses GenericMlp with 1147 inputs)
    if append_combined {
        let bag = create_bag();
        let gs = create_initial_state(&bag);
        run_generic_sparse_training(
            APPENDED_INPUT_SIZE, true, &hidden_layers,
            pop_size, games_per_eval, sigma, lr, iterations,
            eval_interval, eval_games, &checkpoint_dir, &gs,
            opponent_epsilon, time_limit_secs, resume_path.as_deref(), seed,
            &training_opponent,
        );
        return;
    }

    // Standard NNUE path (1106 sparse features)
    // If --layers was explicitly set, or if there are more than 2 hidden layers,
    // use the GenericMlp path which supports any depth.
    let use_generic = layers_str.is_some() || hidden_layers.len() > 2;

    if use_generic {
        let bag = create_bag();
        let gs = create_initial_state(&bag);
        run_generic_sparse_training(
            NUM_FEATURES, false, &hidden_layers,
            pop_size, games_per_eval, sigma, lr, iterations,
            eval_interval, eval_games, &checkpoint_dir, &gs,
            opponent_epsilon, time_limit_secs, resume_path.as_deref(), seed,
            &training_opponent,
        );
        return;
    }

    // Legacy 2-hidden-layer NNUE path (kept for backward compatibility with .nnue files)
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
    println!("  opponent epsilon: {:.2}", opponent_epsilon);
    println!("  iterations: {}", iterations);
    if let Some(tl) = time_limit_secs {
        println!("  time limit: {} seconds", tl);
    }
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
    let mut rng = StdRng::seed_from_u64(seed);
    let mut opponent_epsilon = opponent_epsilon; // make mutable for adaptive adjustment

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
                let opp_eps = opponent_epsilon;
                let win_rate = if self_play {
                    // Self-play: perturbation plays against unperturbed base weights
                    let base_weights_copy = reconstruct_weights(&w);
                    let base_eval = NnueEvaluator::new(base_weights_copy);
                    evaluate_perturbation(weights, &base_eval, &gs, games_per_eval, game_seed, opp_eps)
                } else {
                    match &training_opponent {
                        TrainingOpponent::Random => {
                            // Play against random: use a dummy heuristic for the
                            // evaluate_generic opponent slot but with epsilon=1.0
                            // so every opponent move is random.
                            let heuristic = StaticHeuristicEvaluator::new();
                            evaluate_perturbation(weights, &heuristic, &gs, games_per_eval, game_seed, 1.0)
                        }
                        TrainingOpponent::Eval(opp_eval) => {
                            evaluate_perturbation(weights, *opp_eval, &gs, games_per_eval, game_seed, opp_eps)
                        }
                    }
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
            "iter {:>4}/{}: avg_wr+={:.3} avg_wr-={:.3} max_wr={:.3} sigma={:.4} opp_eps={:.2} ({} games in {:.1?})",
            iter + 1, iterations,
            avg_plus, avg_minus, max_wr, sigma, opponent_epsilon,
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
            let _eval_result = run_matches(
                &gs, &nnue_player, &heur_player, eval_games,
                &format!("NNUE(ES iter={}) vs Heuristic", iter + 1),
            );

            // Adaptive opponent epsilon — based on TRAINING win rate, not eval
            let train_wr_pct = avg_plus * 100.0;
            let old_opp_eps = opponent_epsilon;
            if train_wr_pct > 60.0 {
                opponent_epsilon = (opponent_epsilon - 0.05).max(0.0);
            } else if train_wr_pct < 30.0 {
                opponent_epsilon = (opponent_epsilon + 0.05).min(0.5);
            }
            if (old_opp_eps - opponent_epsilon).abs() > 0.001 {
                println!(
                    "  Opponent epsilon: {:.2} -> {:.2} (training wr was {:.1}%)",
                    old_opp_eps, opponent_epsilon, train_wr_pct,
                );
            }
        }
    }

    // Save final weights
    let final_weights = reconstruct_weights(&w);
    let final_path = format!("{}/es_final.nnue", checkpoint_dir);
    final_weights.save(&final_path).expect("Failed to save final weights");
    println!("\nES training complete: {} iterations in {:.1?}", last_iter, total_start.elapsed());
    println!("Final weights saved to: {}", final_path);
}
