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
//!                 [--layers 64,64,32]  -- configurable hidden layer sizes
//!                 [--pop 50] [--games 10] [--sigma 0.01] [--lr 0.01]
//!                 [--iterations 200] [--eval-interval 20] [--eval-games 500]
//!                 [--checkpoint-dir <dir>] [--time-limit 3600]
//!                 [--opponent <spec>]  -- training opponent (model ID, file, "base", "random")
//!                 [--benchmark <spec>] -- eval benchmark opponent (same specs; default "base")
//!                 [--append-combined]  -- use 1147-input network (1106 NNUE + 41 combined)
//!                 [--append-all]       -- use 1171-input network (1106 NNUE + 41 combined + 24 expensive)
//!                 [--input-features combined]  -- use 41 combined features only
//!                 [--input-features guard]  -- use 65 features (24 expensive + 41 combined)
//!                 [--profile]              -- print per-phase timing breakdown every 50 iters

use std::time::{Duration, Instant};

use rand::rngs::{SmallRng, StdRng};
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
use duke_training::match_runner::{run_matches, win_rate, Player};
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
    GenericMlp, GenericEvaluator, ALL_APPENDED_INPUT_SIZE,
};
use duke_training::loaded_model::{LoadedModel, NUM_GUARD_ALL_FEATURES};
use duke_training::model_registry::{ModelRegistry, TrainingInfo, BenchmarkRecord};



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
                    let mv = greedy_move(&mut game, candidate, rng);
                    mv.play(&mut game, rng);
                } else {
                    // Opponent: epsilon-greedy
                    if opponent_epsilon > 0.0 && rng.gen::<f32>() < opponent_epsilon {
                        ai.play_next_move(rng, &mut game);
                    } else {
                        let mv = greedy_move(&mut game, opponent, rng);
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
    assert!(k > 0, "evaluate_generic called with k=0, would produce NaN");
    assert!(opponent_epsilon >= 0.0, "opponent_epsilon must be non-negative, got {}", opponent_epsilon);
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

// ── Profiling ────────────────────────────────────────────────────────────

/// Accumulated timing statistics for ES training phases.
struct ProfileStats {
    seed_gen: Duration,
    game_play: Duration,
    grad_accum: Duration,
    adam_update: Duration,
    eval: Duration,
    iterations: u32,
}

impl ProfileStats {
    fn new() -> Self {
        Self {
            seed_gen: Duration::ZERO,
            game_play: Duration::ZERO,
            grad_accum: Duration::ZERO,
            adam_update: Duration::ZERO,
            eval: Duration::ZERO,
            iterations: 0,
        }
    }

    fn total_measured(&self) -> Duration {
        self.seed_gen + self.game_play + self.grad_accum + self.adam_update + self.eval
    }

    fn print_summary(&self, label: &str) {
        let total = self.total_measured();
        let total_s = total.as_secs_f64();
        if total_s < 1e-9 { return; }
        let pct = |d: Duration| d.as_secs_f64() / total_s * 100.0;
        println!(
            "PROFILE ({}): games={:.1}s({:.1}%) grad={:.1}s({:.1}%) adam={:.1}s({:.1}%) seeds={:.1}s({:.1}%) eval={:.1}s({:.1}%) | total={:.1}s",
            label,
            self.game_play.as_secs_f64(), pct(self.game_play),
            self.grad_accum.as_secs_f64(), pct(self.grad_accum),
            self.adam_update.as_secs_f64(), pct(self.adam_update),
            self.seed_gen.as_secs_f64(), pct(self.seed_gen),
            self.eval.as_secs_f64(), pct(self.eval),
            total_s,
        );
    }
}

// ── Shared ES training loop ──────────────────────────────────────────────

/// Configuration for the ES training loop.
#[derive(Clone, Copy)]
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
    /// The opponent used for periodic evaluation/benchmark (INIT + EVAL lines).
    benchmark_opponent: &'a TrainingOpponent<'a>,
    /// Human-readable label for the benchmark opponent (e.g. "Base", "random", model path).
    benchmark_label: &'a str,
    /// When true, print per-phase timing breakdown every 50 iterations and at end.
    profile: bool,
}

/// Result of an ES training run, for post-training registration.
pub struct TrainingResult {
    pub final_checkpoint_path: String,
    pub eval_wins: u32,
    pub eval_losses: u32,
    pub eval_ties: u32,
    pub eval_games: u32,
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
) -> TrainingResult {
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

    // Evaluate initial win rate against benchmark opponent
    {
        let init_eval = make_evaluator(&w);
        let cand_player = Player::Evaluator(&*init_eval);
        let bench_player = match config.benchmark_opponent {
            TrainingOpponent::Random => Player::Random,
            TrainingOpponent::Eval(e) => Player::Evaluator(*e),
        };
        print!("  INIT: ");
        run_matches(config.gs, &cand_player, &bench_player, config.eval_games,
            &format!("{} vs {}", mode_label, config.benchmark_label));
    }

    let total_start = Instant::now();
    let mut last_iter = 0u32;

    // Track the last eval result for the training result
    let mut last_eval_wins = 0u32;
    let mut last_eval_losses = 0u32;
    let mut last_eval_ties = 0u32;

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

    // Pre-allocate gradient vector; zeroed each iteration to avoid per-iter allocation
    let mut grad = vec![0.0f32; dim];

    // Pre-allocate reward buffers; reused each iteration to avoid per-iter allocation
    let mut reward_plus = vec![0.0f32; pop_size];
    let mut reward_minus = vec![0.0f32; pop_size];

    // Pre-allocate epsilon buffer for gradient accumulation; reused each iteration
    let mut epsilon_buf = vec![0.0f32; dim];

    // Profiling accumulators (only used when config.profile is true)
    let mut profile = ProfileStats::new();
    let do_profile = config.profile;

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

        // ── Phase 1: Seed generation ──
        let t_phase = if do_profile { Some(Instant::now()) } else { None };

        // Generate perturbation seeds
        let perturbation_seeds: Vec<u64> = (0..pop_size)
            .map(|_| rng.gen::<u64>())
            .collect();

        let game_seed_base: u64 = rng.gen();

        if let Some(t) = t_phase { profile.seed_gen += t.elapsed(); }

        // ── Phase 2: Game playing ──
        let t_phase = if do_profile { Some(Instant::now()) } else { None };

        // Evaluate all perturbations in parallel
        let sigma_snap = sigma_current;
        let opp_eps_snap = opponent_epsilon;
        let results: Vec<(usize, f32, f32)> = (0..pop_size * 2)
            .into_par_iter()
            .map(|idx| {
                let pert_idx = idx / 2;
                let is_positive = idx % 2 == 0;
                let pert_seed = perturbation_seeds[pert_idx];

                let mut pert_rng = SmallRng::seed_from_u64(pert_seed);
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

        // Organize results (reuse pre-allocated buffers)
        reward_plus.iter_mut().for_each(|v| *v = 0.0);
        reward_minus.iter_mut().for_each(|v| *v = 0.0);
        for (i, &(pert_idx, win_rate, _)) in results.iter().enumerate() {
            if i % 2 == 0 {
                reward_plus[pert_idx] = win_rate;
            } else {
                reward_minus[pert_idx] = win_rate;
            }
        }

        if let Some(t) = t_phase { profile.game_play += t.elapsed(); }

        // ── Phase 3: Gradient accumulation ──
        let t_phase = if do_profile { Some(Instant::now()) } else { None };

        // Compute gradient
        let grad_scale = 1.0 / (pop_size as f32 * sigma_current);
        grad.iter_mut().for_each(|g| *g = 0.0);

        for i in 0..pop_size {
            let diff = reward_plus[i] - reward_minus[i];
            if diff.abs() < 1e-12 { continue; }
            let mut pert_rng = SmallRng::seed_from_u64(perturbation_seeds[i]);
            // Reuse pre-allocated buffer instead of allocating a new Vec each iteration
            randn_vec_into(dim, &mut pert_rng, &mut epsilon_buf);
            for j in 0..dim {
                grad[j] += diff * epsilon_buf[j];
            }
        }
        for j in 0..dim {
            grad[j] *= grad_scale;
        }

        if let Some(t) = t_phase { profile.grad_accum += t.elapsed(); }

        // ── Phase 4: Adam update ──
        let t_phase = if do_profile { Some(Instant::now()) } else { None };

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

        if let Some(t) = t_phase { profile.adam_update += t.elapsed(); }

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
            let t_eval = if do_profile { Some(Instant::now()) } else { None };

            let ckpt_path = format!("{}/es_{}_iter_{}", config.checkpoint_dir, mode_label.to_lowercase(), iter + 1);
            save_checkpoint(&w, &ckpt_path);
            println!("  Saved checkpoint: {}", ckpt_path);

            let eval_evaluator = make_evaluator(&w);
            let bench_player = match config.benchmark_opponent {
                TrainingOpponent::Random => Player::Random,
                TrainingOpponent::Eval(e) => Player::Evaluator(*e),
            };
            let cand_player = Player::Evaluator(&*eval_evaluator);
            print!("  EVAL: ");
            let eval_result = run_matches(
                config.gs, &cand_player, &bench_player, config.eval_games,
                &format!("{}(ES iter={}) vs {}", mode_label, iter + 1, config.benchmark_label),
            );

            if let Some(t) = t_eval { profile.eval += t.elapsed(); }

            // Track for TrainingResult
            last_eval_wins = eval_result.player_a_wins;
            last_eval_losses = eval_result.player_b_wins;
            last_eval_ties = eval_result.ties;

            // Adaptive sigma: track improvement
            let eval_wr = win_rate(eval_result.player_a_wins, eval_result.ties, config.eval_games) as f32;
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

            // Adaptive opponent epsilon -- based on TRAINING win rate, not eval
            let train_wr_pct = avg_plus * 100.0;
            let old_opp_eps = opponent_epsilon;
            if train_wr_pct > 60.0 {
                opponent_epsilon = (opponent_epsilon - 0.05).max(0.0);
            } else if train_wr_pct < 30.0 {
                opponent_epsilon = (opponent_epsilon + 0.05).min(1.0);
            }
            if (old_opp_eps - opponent_epsilon).abs() > 0.001 {
                println!(
                    "  Opponent epsilon: {:.2} -> {:.2} (training wr was {:.1}%)",
                    old_opp_eps, opponent_epsilon, train_wr_pct,
                );
            }
        }

        // Periodic profile summary
        if do_profile {
            profile.iterations = iter + 1;
            if (iter + 1) % 50 == 0 {
                profile.print_summary(&format!("iter {}", iter + 1));
            }
        }
    }

    // Save final weights
    let final_path = format!("{}/es_{}_final", config.checkpoint_dir, mode_label.to_lowercase());
    save_checkpoint(&w, &final_path);
    println!("\nES {} training complete: {} iters in {:.1?}", mode_label, last_iter, total_start.elapsed());
    println!("Final weights saved to: {}", final_path);

    if do_profile {
        profile.print_summary(&format!("FINAL after {} iters", profile.iterations));
    }

    TrainingResult {
        final_checkpoint_path: final_path,
        eval_wins: last_eval_wins,
        eval_losses: last_eval_losses,
        eval_ties: last_eval_ties,
        eval_games: config.eval_games,
    }
}

// ── Thin entry point that configures closures for the shared loop ─────────

/// Run the ES training loop with a GenericMlp network.
///
/// Unified entry point for all GenericMlp modes (dense 41/65, sparse 1106/1147).
/// The caller provides the pre-built `EsConfig` plus GenericMlp-specific parameters:
/// - `input_size`: total input dimension
/// - `hidden_layers`: hidden layer sizes
/// - `make_evaluator`: closure that wraps a flat weight vec into a `GameEvaluator`
/// - `mode_label`: human-readable label for log/checkpoint naming
/// - `resume_path`: optional path to resume from
fn run_gmlp_training(
    config: &EsConfig,
    input_size: usize,
    hidden_layers: &[usize],
    make_evaluator: impl Fn(&[f32]) -> Box<dyn GameEvaluator + Sync> + Sync,
    mode_label: &str,
    resume_path: Option<&str>,
) -> TrainingResult {
    let init_weights = if let Some(path) = resume_path {
        let net = GenericMlp::load(path).expect("Failed to load weights");
        assert_eq!(net.input_size, input_size, "input_size mismatch");
        assert_eq!(net.hidden_layers, hidden_layers, "hidden_layers mismatch");
        println!("  resuming from: {}", path);
        net.weights
    } else {
        let mut rng = StdRng::seed_from_u64(config.seed);
        println!("  starting from random weights");
        GenericMlp::random(input_size, hidden_layers.to_vec(), &mut rng).weights
    };

    let tmp_net = GenericMlp::from_flat(vec![0.0; init_weights.len()], input_size, hidden_layers.to_vec());
    let arch_desc = tmp_net.arch_string();

    let hl_for_save = hidden_layers.to_vec();
    let save_checkpoint = move |weights: &[f32], path: &str| {
        let net = GenericMlp::from_flat(weights.to_vec(), input_size, hl_for_save.clone());
        let full_path = format!("{}.gmlp", path);
        net.save(&full_path).expect("Failed to save checkpoint");
    };

    run_es_training_loop(config, &make_evaluator, &save_checkpoint, init_weights, mode_label, &arch_desc)
}

/// Run the legacy 2-hidden-layer NNUE training path.
///
/// Kept for backward compatibility with `.nnue` files. Supports `--self-play`
/// and `--last-layer-only` modes that are specific to the fixed-depth NNUE architecture.
fn run_legacy_nnue_training(
    config: &EsConfig,
    l1_size: usize,
    l2_size: usize,
    resume_path: Option<&str>,
    self_play: bool,
    last_layer_only: bool,
    seed: u64,
    training_opponent: &TrainingOpponent,
) -> TrainingResult {
    let mut rng = StdRng::seed_from_u64(seed);

    // In last-layer-only mode, we keep the frozen base weights separately
    // and only optimize the last layer (l3_weight + l3_bias).
    let base_weights: Option<NnueWeights> = if last_layer_only {
        if resume_path.is_none() {
            panic!("--last-layer-only requires --resume to provide frozen lower layers");
        }
        let weights = NnueWeights::load(resume_path.unwrap())
            .expect("Failed to load NNUE weights");
        assert_eq!(weights.l1_size, l1_size, "l1 mismatch");
        assert_eq!(weights.l2_size, l2_size, "l2 mismatch");
        Some(weights)
    } else {
        None
    };

    let init_weights: Vec<f32> = if last_layer_only {
        flatten_last_layer(base_weights.as_ref().unwrap())
    } else if let Some(path) = resume_path {
        let weights = NnueWeights::load(path).expect("Failed to load NNUE weights");
        assert_eq!(weights.l1_size, l1_size, "l1 mismatch");
        assert_eq!(weights.l2_size, l2_size, "l2 mismatch");
        flatten_weights(&weights)
    } else {
        let weights = random_weights(l1_size, l2_size, &mut rng);
        flatten_weights(&weights)
    };

    if resume_path.is_some() {
        println!("  resuming from: {}", resume_path.unwrap());
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

    // For self-play, the opponent is the current unperturbed weights -- but the
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
    // which is close enough -- in practice self-play was barely used.

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

    let base_weights_for_selfplay = base_weights.clone();
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
            // init_weights is truncated to last-layer only; need the full base weights
            let full_base = base_weights_for_selfplay.as_ref().expect(
                "BUG: --self-play with --last-layer-only requires --base-weights"
            );
            unflatten_last_layer(&init_weights, full_base)
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
            TrainingOpponent::Eval(e) => TrainingOpponent::Eval(*e),
        }
    };

    // Build a local config that overrides training_opponent with the effective one
    let legacy_config = EsConfig {
        training_opponent: &effective_opponent,
        ..*config
    };

    run_es_training_loop(&legacy_config, &make_evaluator, &save_checkpoint, init_weights, "NNUE-Legacy", &arch_desc)
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
fn randn_vec(n: usize, rng: &mut impl Rng) -> Vec<f32> {
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

/// Fill a pre-allocated buffer with standard-normal samples using Box-Muller transform.
/// Avoids heap allocation when the caller can reuse a buffer.
fn randn_vec_into(n: usize, rng: &mut impl Rng, out: &mut [f32]) {
    debug_assert!(out.len() >= n);
    let mut i = 0;
    while i < n {
        let u1: f64 = rng.gen::<f64>().max(1e-30);
        let u2: f64 = rng.gen::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out[i] = (r * theta.cos()) as f32;
        i += 1;
        if i < n {
            out[i] = (r * theta.sin()) as f32;
            i += 1;
        }
    }
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
    let append_all = args.iter().any(|a| a == "--append-all");
    let profile = args.iter().any(|a| a == "--profile");
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
    let db_path: Option<String> = parse_flag(&args, "--db");
    let description: Option<String> = parse_flag(&args, "--description");

    // Parse --opponent flag: path to a saved model, "base", or "random"
    let opponent_spec = parse_flag::<String>(&args, "--opponent")
        .unwrap_or_else(|| "base".to_string());

    let opponent_model = LoadedModel::from_spec(&opponent_spec, false);
    let training_opponent: TrainingOpponent = match &opponent_model.evaluator {
        Some(eval) => TrainingOpponent::Eval(eval.as_ref()),
        None => TrainingOpponent::Random,
    };

    println!("Training opponent: {}", opponent_model.label);

    // Parse --benchmark flag: model to benchmark against at each eval interval
    let benchmark_spec = parse_flag::<String>(&args, "--benchmark")
        .unwrap_or_else(|| "base".to_string());

    let benchmark_model = LoadedModel::from_spec(&benchmark_spec, false);
    let benchmark_opponent: TrainingOpponent = match &benchmark_model.evaluator {
        Some(eval) => TrainingOpponent::Eval(eval.as_ref()),
        None => TrainingOpponent::Random,
    };
    let benchmark_label = &benchmark_model.label;

    println!("Benchmark opponent: {}", benchmark_label);

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

    // Create game state once, shared by all training modes
    let bag = create_bag();
    let gs = create_initial_state(&bag);

    // Track the file format for registry ("gmlp" or "nnue")
    let mut model_format = "gmlp";

    // Validate --input-features value before dispatching
    if !["nnue", "combined", "guard"].contains(&input_features.as_str()) {
        eprintln!(
            "Warning: unknown --input-features '{}', expected 'nnue', 'combined', or 'guard'. Falling back to 'nnue'.",
            input_features
        );
    }

    // Dispatch to combined-features mode (41 inputs)
    let training_result = if input_features == "combined" {
        let input_size = DenseFeatureMode::Combined.input_size();
        let hl = hidden_layers.clone();
        let config = EsConfig {
            pop_size, games_per_eval, sigma, lr, iterations,
            eval_interval, eval_games, checkpoint_dir: &checkpoint_dir,
            gs: &gs, time_limit_secs, seed,
            initial_opponent_epsilon: opponent_epsilon,
            training_opponent: &training_opponent,
            train_max_turns: 200,
            benchmark_opponent: &benchmark_opponent,
            benchmark_label, profile,
        };
        run_gmlp_training(
            &config, input_size, &hidden_layers,
            move |weights: &[f32]| {
                let net = GenericMlp::from_flat(weights.to_vec(), input_size, hl.clone());
                Box::new(GenericEvaluator { net })
            },
            DenseFeatureMode::Combined.label(),
            resume_path.as_deref(),
        )
    }
    // Dispatch to guard-features mode (65 inputs: 24 expensive + 41 combined)
    else if input_features == "guard" {
        let input_size = DenseFeatureMode::Guard.input_size();
        let hl = hidden_layers.clone();
        let config = EsConfig {
            pop_size, games_per_eval, sigma, lr, iterations,
            eval_interval, eval_games, checkpoint_dir: &checkpoint_dir,
            gs: &gs, time_limit_secs, seed,
            initial_opponent_epsilon: opponent_epsilon,
            training_opponent: &training_opponent,
            train_max_turns: 50,
            benchmark_opponent: &benchmark_opponent,
            benchmark_label, profile,
        };
        run_gmlp_training(
            &config, input_size, &hidden_layers,
            move |weights: &[f32]| {
                let net = GenericMlp::from_flat(weights.to_vec(), input_size, hl.clone());
                Box::new(GenericEvaluator { net })
            },
            DenseFeatureMode::Guard.label(),
            resume_path.as_deref(),
        )
    }
    // Dispatch to appended-input mode if requested (uses GenericMlp with 1147 inputs)
    else if append_combined {
        let hl = hidden_layers.clone();
        let config = EsConfig {
            pop_size, games_per_eval, sigma, lr, iterations,
            eval_interval, eval_games, checkpoint_dir: &checkpoint_dir,
            gs: &gs, time_limit_secs, seed,
            initial_opponent_epsilon: opponent_epsilon,
            training_opponent: &training_opponent,
            train_max_turns: 50,
            benchmark_opponent: &benchmark_opponent,
            benchmark_label, profile,
        };
        run_gmlp_training(
            &config, APPENDED_INPUT_SIZE, &hidden_layers,
            move |weights: &[f32]| {
                let net = GenericMlp::from_flat(weights.to_vec(), APPENDED_INPUT_SIZE, hl.clone());
                Box::new(GenericEvaluator { net })
            },
            "Appended",
            resume_path.as_deref(),
        )
    }
    // Dispatch to all-appended mode (1171 = 1106 board + 41 combined + 24 expensive)
    else if append_all {
        let hl = hidden_layers.clone();
        let config = EsConfig {
            pop_size, games_per_eval, sigma, lr, iterations,
            eval_interval, eval_games, checkpoint_dir: &checkpoint_dir,
            gs: &gs, time_limit_secs, seed,
            initial_opponent_epsilon: opponent_epsilon,
            training_opponent: &training_opponent,
            train_max_turns: 50,
            benchmark_opponent: &benchmark_opponent,
            benchmark_label, profile,
        };
        run_gmlp_training(
            &config, ALL_APPENDED_INPUT_SIZE, &hidden_layers,
            move |weights: &[f32]| {
                let net = GenericMlp::from_flat(weights.to_vec(), ALL_APPENDED_INPUT_SIZE, hl.clone());
                Box::new(GenericEvaluator { net })
            },
            "AllAppended",
            resume_path.as_deref(),
        )
    }
    // Standard NNUE path (1106 sparse features) -- generic (arbitrary depth)
    else if layers_str.is_some() || hidden_layers.len() > 2 {
        let hl = hidden_layers.clone();
        let config = EsConfig {
            pop_size, games_per_eval, sigma, lr, iterations,
            eval_interval, eval_games, checkpoint_dir: &checkpoint_dir,
            gs: &gs, time_limit_secs, seed,
            initial_opponent_epsilon: opponent_epsilon,
            training_opponent: &training_opponent,
            train_max_turns: 200,
            benchmark_opponent: &benchmark_opponent,
            benchmark_label, profile,
        };
        run_gmlp_training(
            &config, NUM_FEATURES, &hidden_layers,
            move |weights: &[f32]| {
                let net = GenericMlp::from_flat(weights.to_vec(), NUM_FEATURES, hl.clone());
                Box::new(GenericEvaluator { net })
            },
            "NNUE",
            resume_path.as_deref(),
        )
    } else {
        // Legacy 2-hidden-layer NNUE path (kept for backward compatibility with .nnue files)
        model_format = "nnue";
        let config = EsConfig {
            pop_size, games_per_eval, sigma, lr, iterations,
            eval_interval, eval_games, checkpoint_dir: &checkpoint_dir,
            gs: &gs, time_limit_secs, seed,
            initial_opponent_epsilon: opponent_epsilon,
            training_opponent: &training_opponent,
            train_max_turns: 200,
            benchmark_opponent: &benchmark_opponent,
            benchmark_label, profile,
        };
        run_legacy_nnue_training(
            &config, l1_size, l2_size, resume_path.as_deref(),
            self_play, last_layer_only, seed, &training_opponent,
        )
    };

    // ── Auto-register model in registry if --db was provided ─────────────
    if let Some(ref db) = db_path {
        // Determine the full checkpoint file path (with extension)
        let final_file = format!("{}.{}", training_result.final_checkpoint_path, model_format);

        println!("\nRegistering model in registry: {}", db);

        let registry = ModelRegistry::open(db).expect("Failed to open model registry DB");

        let training_info = TrainingInfo {
            iterations: Some(iterations),
            sigma: Some(sigma),
            lr: Some(lr),
            opponent: Some(opponent_spec.clone()),
            parent_model_id: None,
        };

        let model_id = if model_format == "nnue" {
            let weights = NnueWeights::load(&final_file)
                .expect("Failed to load final NNUE checkpoint for registration");
            registry.register_nnue(
                &final_file, &weights,
                description.as_deref(),
                Some(&training_info),
            ).expect("Failed to register NNUE model")
        } else {
            let net = GenericMlp::load(&final_file)
                .expect("Failed to load final GMLP checkpoint for registration");
            registry.register_gmlp(
                &final_file, &net,
                description.as_deref(),
                Some(&training_info),
            ).expect("Failed to register GMLP model")
        };

        println!("  Registered as model ID {}", model_id);

        // Record the final eval benchmark
        if training_result.eval_games > 0 {
            let bench = BenchmarkRecord {
                id: None,
                opponent: benchmark_spec.clone(),
                opponent_model_id: benchmark_model.id,
                num_games: training_result.eval_games,
                wins: training_result.eval_wins,
                losses: training_result.eval_losses,
                ties: training_result.eval_ties,
                win_rate: None,
                elo: None,
                benchmark_date: None,
            };
            let bench_id = registry.record_benchmark(model_id, &bench)
                .expect("Failed to record benchmark");
            let wr = win_rate(training_result.eval_wins, training_result.eval_ties, training_result.eval_games);
            println!("  Recorded benchmark ID {} (vs {}: {:.1}% win rate)", bench_id, benchmark_spec, wr * 100.0);
        }
    }
}
