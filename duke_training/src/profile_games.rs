//! Micro-profiling binary for the game-playing hot loop.
//!
//! Plays N games single-threaded (no rayon) and reports a per-move timing
//! breakdown for: move generation, state cloning, move application, and
//! evaluation.
//!
//! Usage:
//!   profile_games [--model <spec>] [--opponent <spec>] [--games N] [--eval-breakdown]
//!
//! Defaults: model=base, opponent=same as model, games=1000
//!
//! The --eval-breakdown flag, when the model is a 1147-input .gmlp file,
//! reports a sub-breakdown of evaluation time into feature extraction
//! vs forward pass (matmul).

use std::env;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use duke_rust::game::ai::player::AiMove;
use duke_rust::game::state::GameResult;

use duke_training::cli::parse_flag;
use duke_training::game_setup::{create_bag, create_initial_state, GameEvaluator};
use duke_training::generic_mlp::{GenericMlp, L1Accumulator};
use duke_training::loaded_model::LoadedModel;

/// Safety limit: if a game exceeds this many turns, force a draw.
const MAX_TURNS: u32 = 500;

fn main() {
    let args: Vec<String> = env::args().collect();

    let model_spec: String = parse_flag(&args, "--model").unwrap_or_else(|| "base".to_string());
    let opponent_spec: String = parse_flag(&args, "--opponent").unwrap_or_else(|| model_spec.clone());
    let num_games: u32 = parse_flag(&args, "--games").unwrap_or(1000);
    let eval_breakdown = args.iter().any(|a| a == "--eval-breakdown");

    // Check if model qualifies for eval breakdown (1147-input .gmlp)
    let breakdown_active = eval_breakdown && model_spec.ends_with(".gmlp") && {
        let net = GenericMlp::load(&model_spec).expect("Failed to load .gmlp for breakdown check");
        net.input_size == 1147
    };

    println!("=== Profile Games ===");
    println!("  Model:    {}", model_spec);
    println!("  Opponent: {}", opponent_spec);
    println!("  Games:    {}", num_games);
    if eval_breakdown {
        if breakdown_active {
            println!("  Eval breakdown: ON (1147-input model)");
        } else {
            println!("  Eval breakdown: requested but model is not a 1147-input .gmlp — skipping");
        }
    }
    println!();

    let model = LoadedModel::from_spec(&model_spec);
    let opponent = LoadedModel::from_spec(&opponent_spec);

    let bag = create_bag();
    let init_state = create_initial_state(&bag);

    // Accumulators
    let mut move_gen_time = Duration::ZERO;
    let mut clone_time = Duration::ZERO;
    let mut play_time = Duration::ZERO;
    let mut eval_time = Duration::ZERO;
    let mut total_moves: u64 = 0;
    let mut total_candidates: u64 = 0;
    let mut total_games_completed: u32 = 0;

    // Eval breakdown accumulators (only used when breakdown_active)
    let mut feature_time = Duration::ZERO;
    let mut forward_time = Duration::ZERO;
    let mut total_evals: u64 = 0;

    // Pre-load the network reference for breakdown mode
    let breakdown_net: Option<GenericMlp> = if breakdown_active {
        Some(GenericMlp::load(&model_spec).expect("Failed to load .gmlp for breakdown"))
    } else {
        None
    };

    // Deterministic eval rng (same as greedy_move uses)
    let base_eval_rng = StdRng::seed_from_u64(0);

    let wall_start = Instant::now();

    for game_idx in 0..num_games {
        let mut rng = StdRng::seed_from_u64(game_idx as u64);
        let mut game_state = init_state.clone();
        let mut turns = 0u32;

        loop {
            match game_state.game_result() {
                GameResult::Ongoing => {
                    if turns >= MAX_TURNS {
                        break;
                    }
                    turns += 1;
                    total_moves += 1;

                    // Pick the evaluator for the current player
                    let current = game_state.current_player_turn();
                    let is_model_player = matches!(current, duke_rust::game::tile::Owner::TopPlayer);
                    let evaluator: &dyn GameEvaluator = match current {
                        duke_rust::game::tile::Owner::TopPlayer => {
                            match &model.evaluator {
                                Some(e) => e.as_ref(),
                                None => {
                                    // Random player: just play a random move, no profiling detail
                                    let ai = duke_rust::game::ai::stupid_sync_ai::StupidSyncAi {};
                                    duke_rust::game::ai::player::ArtificialPlayer::play_next_move(
                                        &ai, &mut rng, &mut game_state,
                                    );
                                    continue;
                                }
                            }
                        }
                        duke_rust::game::tile::Owner::BottomPlayer => {
                            match &opponent.evaluator {
                                Some(e) => e.as_ref(),
                                None => {
                                    let ai = duke_rust::game::ai::stupid_sync_ai::StupidSyncAi {};
                                    duke_rust::game::ai::player::ArtificialPlayer::play_next_move(
                                        &ai, &mut rng, &mut game_state,
                                    );
                                    continue;
                                }
                            }
                        }
                    };

                    // --- Move generation ---
                    let t = Instant::now();
                    let mut moves: Vec<AiMove> = AiMove::all_moves(&game_state).collect();
                    move_gen_time += t.elapsed();

                    assert!(!moves.is_empty(), "no legal moves in Ongoing state");
                    moves.shuffle(&mut rng);
                    total_candidates += moves.len() as u64;

                    // --- Evaluate each candidate ---
                    let mut best_score = f64::NEG_INFINITY;
                    let mut best_move = None;

                    for mv in &moves {
                        let t = Instant::now();
                        let mut clone = game_state.clone();
                        clone_time += t.elapsed();

                        let t = Instant::now();
                        let mut eval_rng = base_eval_rng.clone();
                        mv.play(&mut clone, &mut eval_rng);
                        play_time += t.elapsed();

                        // When breakdown is active for the model player,
                        // manually decompose into feature extraction + forward
                        // pass instead of calling evaluator.evaluate() (which
                        // does its own internal extraction, making a separate
                        // extraction measurement meaningless).
                        let score = if breakdown_active && is_model_player {
                            let net = breakdown_net.as_ref().unwrap();

                            let t = Instant::now();
                            let acc = L1Accumulator::from_state(net, &clone, true);
                            let feat_elapsed = t.elapsed();
                            feature_time += feat_elapsed;

                            let t = Instant::now();
                            let val = acc.forward(net);
                            let fwd_elapsed = t.elapsed();
                            forward_time += fwd_elapsed;

                            eval_time += feat_elapsed + fwd_elapsed;
                            total_evals += 1;

                            -(val as f64)
                        } else {
                            let t = Instant::now();
                            let val = evaluator.evaluate(&clone);
                            eval_time += t.elapsed();
                            -(val as f64)
                        };

                        if score > best_score {
                            best_score = score;
                            best_move = Some(mv.clone());
                        }
                    }

                    // Apply best move to the real game state
                    best_move.unwrap().play(&mut game_state, &mut rng);
                }
                _ => break,
            }
        }

        total_games_completed += 1;

        // Progress indicator every 10% of games
        if num_games >= 10 && (game_idx + 1) % (num_games / 10) == 0 {
            eprint!(
                "\r  Progress: {}/{} games ({:.0}%)",
                game_idx + 1,
                num_games,
                (game_idx + 1) as f64 / num_games as f64 * 100.0,
            );
        }
    }
    eprintln!();

    let wall_elapsed = wall_start.elapsed();

    // Compute totals
    let measured_total = move_gen_time + clone_time + play_time + eval_time;
    let other_time = wall_elapsed.saturating_sub(measured_total);
    let wall_secs = wall_elapsed.as_secs_f64();

    let pct = |d: Duration| -> f64 {
        if wall_secs > 0.0 {
            d.as_secs_f64() / wall_secs * 100.0
        } else {
            0.0
        }
    };

    let avg_us = |d: Duration| -> f64 {
        if total_moves > 0 {
            d.as_secs_f64() * 1_000_000.0 / total_moves as f64
        } else {
            0.0
        }
    };

    let avg_moves_per_game = if total_games_completed > 0 {
        total_moves as f64 / total_games_completed as f64
    } else {
        0.0
    };

    let avg_candidates = if total_moves > 0 {
        total_candidates as f64 / total_moves as f64
    } else {
        0.0
    };

    println!(
        "=== Game Playing Profile ({} games, {} moves) ===",
        total_games_completed, total_moves
    );
    println!(
        "  Move generation:  {:>7.1}s ({:>5.1}%)  avg {:>7.0}us/move",
        move_gen_time.as_secs_f64(),
        pct(move_gen_time),
        avg_us(move_gen_time),
    );
    println!(
        "  State cloning:    {:>7.1}s ({:>5.1}%)  avg {:>7.0}us/move",
        clone_time.as_secs_f64(),
        pct(clone_time),
        avg_us(clone_time),
    );
    println!(
        "  Move application: {:>7.1}s ({:>5.1}%)  avg {:>7.0}us/move",
        play_time.as_secs_f64(),
        pct(play_time),
        avg_us(play_time),
    );
    println!(
        "  Evaluation:       {:>7.1}s ({:>5.1}%)  avg {:>7.0}us/move",
        eval_time.as_secs_f64(),
        pct(eval_time),
        avg_us(eval_time),
    );
    println!(
        "  Other:            {:>7.1}s ({:>5.1}%)",
        other_time.as_secs_f64(),
        pct(other_time),
    );
    println!(
        "  Total:            {:>7.1}s          avg {:>7.0}us/move",
        wall_secs,
        avg_us(wall_elapsed),
    );
    println!("  Avg moves/game:   {:.1}", avg_moves_per_game);
    println!("  Avg candidates:   {:.1}", avg_candidates);

    // Evaluation breakdown report
    if breakdown_active && total_evals > 0 {
        let feature_secs = feature_time.as_secs_f64();
        let forward_secs = forward_time.as_secs_f64();
        let breakdown_total = feature_secs + forward_secs;

        let feature_pct = if breakdown_total > 0.0 { feature_secs / breakdown_total * 100.0 } else { 0.0 };
        let forward_pct = if breakdown_total > 0.0 { forward_secs / breakdown_total * 100.0 } else { 0.0 };

        let avg_feature_us = feature_secs * 1_000_000.0 / total_evals as f64;
        let avg_forward_us = forward_secs * 1_000_000.0 / total_evals as f64;
        let avg_eval_us = breakdown_total * 1_000_000.0 / total_evals as f64;

        println!();
        println!("=== Evaluation Breakdown (1147-input model) ===");
        println!(
            "  Feature extraction: {:>6.1}s ({:>5.1}%)  avg {:>7.0}us/eval",
            feature_secs, feature_pct, avg_feature_us,
        );
        println!(
            "  Forward pass:       {:>6.1}s ({:>5.1}%)  avg {:>7.0}us/eval",
            forward_secs, forward_pct, avg_forward_us,
        );
        println!(
            "  Total evaluation:   {:>6.1}s          avg {:>7.0}us/eval",
            breakdown_total, avg_eval_us,
        );
        println!("  Eval count:         {}", total_evals);
    }
}
