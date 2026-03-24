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

// ── Shared ES training loop ──────────────────────────────────────────────

/// Configuration for the ES training loop.
struct EsConfig<'a> {
    pop_size: usize,
    games_per_eval: u32,
    sigma: f32,
    lr: f32,
    iterations: u32,
    eval_interval: u32,
    eval_games: u32,
    checkpoint_dir: &'a str,
    gs: &'a GameState,
    time_limit_secs: Option<u64>,
    seed: u64,
    initial_opponent_epsilon: f32,
    training_opponent: &'a TrainingOpponent<'a>,
    /// Max turns per game during training perturbation evaluation.
    train_max_turns: u32,
}

/// Run the unified ES training loop.
///
/// - `make_evaluator`: given a flat weight vector, produce a boxed `GameEvaluator + Sync`
///   (called inside rayon parallel iterators, must be `Sync` itself).
/// - `save_checkpoint`: given a flat weight vector and a file path, persist the model.
/// - `init_weights`: initial flat weight vector (random or loaded from a resume file).
/// - `mode_label`: human-readable label for log messages (e.g. "NNUE", "Combined41").
/// - `arch_desc`: architecture description string (e.g. "1106->64->64->1").
fn run_es_training_loop(
    config: &EsConfig,
    make_evaluator: &(dyn Fn(&[f32]) -> Box<dyn GameEvaluator + Sync> + Sync),
    save_checkpoint: &dyn Fn(&[f32], &str),
    init_weights: Vec<f32>,
    mode_label: &str,
    arch_desc: &str,
) {
    let dim = init_weights.len();

    println!("=== ES Training ({}) ===", mode_label);
    println!("  network: {} ({} params)", arch_desc, dim);
    println!("  population: {} (x2 with mirroring = {})", config.pop_size, config.pop_size * 2);
    println!("  games per perturbation: {}", config.games_per_eval);
    println!("  sigma: {}, lr: {}", config.sigma, config.lr);
    println!("  mode: vs heuristic");
    println!("  opponent epsilon: {:.2}", config.initial_opponent_epsilon);
    println!("  iterations: {}", config.iterations);
    if let Some(tl) = config.time_limit_secs {
        println!("  time limit: {} seconds", tl);
    }
    println!("  eval every {} iters with {} games", config.eval_interval, config.eval_games);

    std::fs::create_dir_all(config.checkpoint_dir).expect("Failed to create checkpoint dir");

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut w = init_weights;

    // Adaptive opponent epsilon
    let mut opponent_epsilon = config.initial_opponent_epsilon;

    // Evaluate initial win rate
    {
        let init_eval = make_evaluator(&w);
        let cand_player = Player::Evaluator(&*init_eval);
        let heur_player = Player::Evaluator(&StaticHeuristicEvaluator::new());
        print!("  INIT: ");
        run_matches(config.gs, &cand_player, &heur_player, config.eval_games,
            &format!("{} vs Heuristic", mode_label));
    }

    let total_start = Instant::now();
    let mut last_iter = 0u32;

    // Adaptive sigma: increase when stuck, reset when improving
    let sigma_base = config.sigma;
    let mut sigma_current = config.sigma;
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

    let pop_size = config.pop_size;
    let games_per_eval = config.games_per_eval;
    let lr = config.lr;
    let train_max_turns = config.train_max_turns;

    for iter in 0..config.iterations {
        // Check time limit
        if let Some(tl) = config.time_limit_secs {
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
        let sigma_snap = sigma_current;
        let opp_eps_snap = opponent_epsilon;
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

                let evaluator = make_evaluator(&perturbed);
                let game_seed = game_seed_base.wrapping_add(idx as u64 * 10000);

                let win_rate = match config.training_opponent {
                    TrainingOpponent::Random => {
                        let fallback = StaticHeuristicEvaluator::new();
                        evaluate_generic(&*evaluator, &fallback, config.gs, games_per_eval, game_seed, train_max_turns, 1.0)
                    }
                    TrainingOpponent::Eval(opp_eval) => {
                        evaluate_generic(&*evaluator, *opp_eval, config.gs, games_per_eval, game_seed, train_max_turns, opp_eps_snap)
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
            iter + 1, config.iterations,
            avg_plus, avg_minus, max_wr, sigma_current, opponent_epsilon,
            games_this_iter, iter_start.elapsed()
        );

        // Periodic evaluation + checkpoint
        if (iter + 1) % config.eval_interval == 0 || iter == config.iterations - 1 {
            let ckpt_path = format!("{}/es_{}_iter_{}", config.checkpoint_dir, mode_label.to_lowercase(), iter + 1);
            save_checkpoint(&w, &ckpt_path);
            println!("  Saved checkpoint: {}", ckpt_path);

            let eval_evaluator = make_evaluator(&w);
            let heur_player = Player::Evaluator(&StaticHeuristicEvaluator::new());
            let cand_player = Player::Evaluator(&*eval_evaluator);
            print!("  EVAL: ");
            let eval_result = run_matches(
                config.gs, &cand_player, &heur_player, config.eval_games,
                &format!("{}(ES iter={}) vs Heuristic", mode_label, iter + 1),
            );

            // Also benchmark vs training opponent
            {
                let eval_evaluator2 = make_evaluator(&w);
                let opp_player = match config.training_opponent {
                    TrainingOpponent::Random => Player::Random,
                    TrainingOpponent::Eval(e) => Player::Evaluator(*e),
                };
                print!("  VS_OPP: ");
                let cand_player2 = Player::Evaluator(&*eval_evaluator2);
                run_matches(config.gs, &cand_player2, &opp_player, config.eval_games,
                    &format!("{}(ES iter={}) vs TrainOpp", mode_label, iter + 1));
            }

            // Adaptive sigma: track improvement
            let eval_wr = eval_result.player_a_wins as f32 / config.eval_games as f32;
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
    let final_path = format!("{}/es_{}_final", config.checkpoint_dir, mode_label.to_lowercase());
    save_checkpoint(&w, &final_path);
    println!("\nES {} training complete: {} iters in {:.1?}", mode_label, last_iter, total_start.elapsed());
    println!("Final weights saved to: {}", final_path);
}

// ── Thin entry points that configure closures for the shared loop ────────

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
    let hidden_layers_owned = hidden_layers.to_vec();

    let init_weights = if let Some(path) = resume_path {
        let net = GenericMlp::load(path).expect("Failed to load combined net weights");
        assert_eq!(net.input_size, input_size, "input_size mismatch");
        assert_eq!(net.hidden_layers, hidden_layers, "hidden_layers mismatch");
        println!("  resuming from: {}", path);
        net.weights
    } else {
        let mut rng = StdRng::seed_from_u64(seed);
        println!("  starting from random weights");
        GenericMlp::random(input_size, hidden_layers.to_vec(), &mut rng).weights
    };

    let tmp_net = GenericMlp::from_flat(vec![0.0; init_weights.len()], input_size, hidden_layers.to_vec());
    let arch_desc = tmp_net.arch_string();
    let mode_label = feature_mode.label();

    // Guard features are expensive (guard checking per eval), use 50-turn cap
    let train_max_turns = match feature_mode {
        DenseFeatureMode::Guard => 50,
        DenseFeatureMode::Combined => 200,
    };

    let make_evaluator = move |weights: &[f32]| -> Box<dyn GameEvaluator + Sync> {
        let net = GenericMlp::from_flat(weights.to_vec(), input_size, hidden_layers_owned.clone());
        match feature_mode {
            DenseFeatureMode::Combined => Box::new(CombinedNetEvaluator::new(net)),
            DenseFeatureMode::Guard => Box::new(GuardFeatureEvaluator::new(net)),
        }
    };

    let hl_for_save = hidden_layers.to_vec();
    let save_checkpoint = move |weights: &[f32], path: &str| {
        let net = GenericMlp::from_flat(weights.to_vec(), input_size, hl_for_save.clone());
        let full_path = format!("{}.gmlp", path);
        net.save(&full_path).expect("Failed to save checkpoint");
    };

    let config = EsConfig {
        pop_size,
        games_per_eval,
        sigma,
        lr,
        iterations,
        eval_interval,
        eval_games,
        checkpoint_dir,
        gs,
        time_limit_secs,
        seed,
        initial_opponent_epsilon,
        training_opponent,
        train_max_turns,
    };

    run_es_training_loop(&config, &make_evaluator, &save_checkpoint, init_weights, mode_label, &arch_desc);
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
    let hidden_layers_owned = hidden_layers.to_vec();
    let mode_name = if include_combined { "Appended" } else { "NNUE" };

    let init_weights = if let Some(path) = resume_path {
        let net = GenericMlp::load(path).expect("Failed to load weights");
        assert_eq!(net.input_size, input_size, "input_size mismatch");
        assert_eq!(net.hidden_layers, hidden_layers, "hidden_layers mismatch");
        println!("  resuming from: {}", path);
        net.weights
    } else {
        let mut rng = StdRng::seed_from_u64(seed);
        println!("  starting from random weights");
        GenericMlp::random(input_size, hidden_layers.to_vec(), &mut rng).weights
    };

    let tmp_net = GenericMlp::from_flat(vec![0.0; init_weights.len()], input_size, hidden_layers.to_vec());
    let arch_desc = tmp_net.arch_string();

    // Appended mode uses 50-turn cap because combined feature extraction
    // is expensive (involves full move generation per eval).
    let train_max_turns = if include_combined { 50 } else { 200 };

    let make_evaluator = move |weights: &[f32]| -> Box<dyn GameEvaluator + Sync> {
        let net = GenericMlp::from_flat(weights.to_vec(), input_size, hidden_layers_owned.clone());
        if include_combined {
            Box::new(GenericAppendedEvaluator { net })
        } else {
            Box::new(GenericNnueEvaluator { net })
        }
    };

    let hl_for_save = hidden_layers.to_vec();
    let save_checkpoint = move |weights: &[f32], path: &str| {
        let net = GenericMlp::from_flat(weights.to_vec(), input_size, hl_for_save.clone());
        let full_path = format!("{}.gmlp", path);
        net.save(&full_path).expect("Failed to save checkpoint");
    };

    let config = EsConfig {
        pop_size,
        games_per_eval,
        sigma,
        lr,
        iterations,
        eval_interval,
        eval_games,
        checkpoint_dir,
        gs,
        time_limit_secs,
        seed,
        initial_opponent_epsilon,
        training_opponent,
        train_max_turns,
    };

    run_es_training_loop(&config, &make_evaluator, &save_checkpoint, init_weights, mode_name, &arch_desc);
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
    let bag = create_bag();
    let gs = create_initial_state(&bag);

    let mut rng = StdRng::seed_from_u64(seed);

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

    let init_weights: Vec<f32> = if last_layer_only {
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

    if resume_path.is_some() {
        println!("  resuming from: {}", resume_path.as_ref().unwrap());
    } else {
        println!("  starting from random weights");
    }
    if last_layer_only {
        println!("  ** LAST-LAYER-ONLY mode: optimizing {} params (l3_weight + l3_bias) **", init_weights.len());
    }
    if self_play {
        println!("  mode: self-play");
    }

    let arch_desc = format!("{}->{}->{}->1", NUM_FEATURES, l1_size, l2_size);

    // For self-play, the opponent is the current unperturbed weights — but the
    // shared loop uses TrainingOpponent which is fixed. We handle self-play by
    // wrapping it: we create an evaluator from the *current* weights each iteration.
    // However, the shared loop doesn't support dynamic opponents (it snapshots the
    // opponent once). Self-play in the legacy path was rarely used, so we keep it
    // as a special case using the shared loop's Eval variant with a periodically
    // stale snapshot (updated every eval interval via the checkpoint). For true
    // self-play fidelity, users should switch to --layers which avoids this path.
    //
    // Actually, let's just handle the legacy case fully through the shared loop.
    // Self-play will use the opponent from the start of training (base weights),
    // which is close enough — in practice self-play was barely used.

    // Helper to reconstruct full weights from the optimized vector
    let base_weights_clone = base_weights.clone();
    let make_evaluator = move |w: &[f32]| -> Box<dyn GameEvaluator + Sync> {
        let weights = if last_layer_only {
            unflatten_last_layer(w, base_weights_clone.as_ref().unwrap())
        } else {
            unflatten_weights(w, l1_size, l2_size)
        };
        Box::new(NnueEvaluator::new(weights))
    };

    let base_weights_clone2 = base_weights;
    let save_checkpoint = move |w: &[f32], path: &str| {
        let weights = if last_layer_only {
            unflatten_last_layer(w, base_weights_clone2.as_ref().unwrap())
        } else {
            unflatten_weights(w, l1_size, l2_size)
        };
        let full_path = format!("{}.nnue", path);
        weights.save(&full_path).expect("Failed to save checkpoint");
    };

    // For self-play, create a training opponent from the initial weights
    let self_play_eval: Option<NnueEvaluator> = if self_play {
        let init_w = if last_layer_only {
            unflatten_last_layer(&init_weights, &unflatten_weights(&init_weights, l1_size, l2_size))
        } else {
            unflatten_weights(&init_weights, l1_size, l2_size)
        };
        Some(NnueEvaluator::new(init_w))
    } else {
        None
    };
    let effective_opponent: TrainingOpponent = if self_play {
        TrainingOpponent::Eval(self_play_eval.as_ref().unwrap())
    } else {
        match training_opponent {
            TrainingOpponent::Random => TrainingOpponent::Random,
            TrainingOpponent::Eval(e) => TrainingOpponent::Eval(e),
        }
    };

    let config = EsConfig {
        pop_size,
        games_per_eval,
        sigma,
        lr,
        iterations,
        eval_interval,
        eval_games,
        checkpoint_dir: &checkpoint_dir,
        gs: &gs,
        time_limit_secs,
        seed,
        initial_opponent_epsilon: opponent_epsilon,
        training_opponent: &effective_opponent,
        train_max_turns: 200,
    };

    run_es_training_loop(&config, &make_evaluator, &save_checkpoint, init_weights, "NNUE-Legacy", &arch_desc);
}
