//! RL training: play games with NNUE vs heuristic, update NNUE toward winning.
//!
//! Each iteration:
//! 1. Play N games: NNUE vs Heuristic (parallel with rayon)
//! 2. For each state: target = 1.0 if current player won, 0.0 if lost, 0.5 if tie
//! 3. Backprop MSE loss through the NNUE, update weights
//! 4. Repeat
//!
//! Usage: rl_train [--resume <nnue_path>] [--l1 512] [--l2 64]
//!                 [--games-per-iter 100] [--iterations 500]
//!                 [--lr 0.01] [--lr-end 0.001]
//!                 [--checkpoint-dir <dir>] [--checkpoint-interval 50]

use std::time::Instant;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use duke_rust::game::state::{GameResult, GameState};
use duke_rust::game::tile::Owner;

use duke_training::cli::parse_flag;
use duke_training::encoding::encode_state_flat;
use duke_training::fc_td_training::FcTdTrainer;
use duke_training::game_setup::{
    create_bag, create_initial_state, play_two_player_game,
    StaticHeuristicEvaluator,
};
use duke_training::match_runner::{run_matches, Player};
use duke_training::nnue::NnueEvaluator;
use duke_training::weight_export::export_weights;

type MyBackend = Autodiff<Wgpu>;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let l1_size: usize = parse_flag(&args, "--l1").unwrap_or(512);
    let l2_size: usize = parse_flag(&args, "--l2").unwrap_or(64);
    let resume_path = parse_flag::<String>(&args, "--resume");
    let games_per_iter: u32 = parse_flag(&args, "--games-per-iter").unwrap_or(100);
    let iterations: u32 = parse_flag(&args, "--iterations").unwrap_or(500);
    let lr_start: f64 = parse_flag(&args, "--lr").unwrap_or(0.01);
    let lr_end: f64 = parse_flag(&args, "--lr-end").unwrap_or(lr_start / 10.0);
    let checkpoint_dir = parse_flag::<String>(&args, "--checkpoint-dir")
        .unwrap_or_else(|| "rl_checkpoints".to_string());
    let checkpoint_interval: u32 = parse_flag(&args, "--checkpoint-interval").unwrap_or(50);
    let epsilon: f64 = parse_flag(&args, "--epsilon").unwrap_or(0.05);
    let eval_games: u32 = parse_flag(&args, "--eval-games").unwrap_or(500);

    let bag = create_bag();
    let gs = create_initial_state(&bag);
    let heuristic = StaticHeuristicEvaluator::new();

    let device = WgpuDevice::default();
    let mut trainer: FcTdTrainer<MyBackend> = FcTdTrainer::new(device.clone(), lr_start, &[l1_size, l2_size]);

    if let Some(ref path) = resume_path {
        println!("Loading pre-trained model from: {}", path);
        trainer.load_model(path);
    }

    println!("RL Training: NNUE vs Heuristic");
    println!("  network: 1106→{}→{}→1", l1_size, l2_size);
    println!("  games_per_iter={}, iterations={}, lr={}->{}, epsilon={}",
        games_per_iter, iterations, lr_start, lr_end, epsilon);
    if resume_path.is_none() {
        println!("  WARNING: starting from random weights (no --resume)");
    }

    std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoint dir");

    // Initialize NNUE evaluator for game-playing
    let mut nnue_weights = export_weights(&trainer.model, l1_size, l2_size);
    let mut nnue_evaluator = NnueEvaluator::new(nnue_weights);

    let start = Instant::now();
    let mut total_games_played: u64 = 0;

    for iter in 0..iterations {
        let iter_start = Instant::now();

        // LR decay
        let progress = iter as f64 / iterations.max(1) as f64;
        let lr = lr_start + (lr_end - lr_start) * progress;
        trainer.set_lr(lr);

        // Play games: NNUE vs Heuristic (parallel)
        let seed_base = iter as u64 * games_per_iter as u64;
        let game_results: Vec<(Vec<GameState>, GameResult)> = (0..games_per_iter)
            .into_par_iter()
            .map(|i| {
                let seed = seed_base + i as u64;
                let mut rng = StdRng::seed_from_u64(seed);
                // Alternate sides
                if seed % 2 == 0 {
                    // NNUE is top player
                    play_two_player_game(
                        &gs, &nnue_evaluator, &heuristic, &mut rng, epsilon,
                    )
                } else {
                    // NNUE is bottom player
                    play_two_player_game(
                        &gs, &heuristic, &nnue_evaluator, &mut rng, epsilon,
                    )
                }
            })
            .collect();

        // Count wins/losses
        let mut nnue_wins = 0u32;
        let mut heur_wins = 0u32;
        let mut ties = 0u32;
        for (i, (_, result)) in game_results.iter().enumerate() {
            let seed = seed_base + i as u64;
            let nnue_is_top = seed % 2 == 0;
            match result {
                GameResult::Won(Owner::TopPlayer) => {
                    if nnue_is_top { nnue_wins += 1; } else { heur_wins += 1; }
                }
                GameResult::Won(Owner::BottomPlayer) => {
                    if nnue_is_top { heur_wins += 1; } else { nnue_wins += 1; }
                }
                _ => ties += 1,
            }
        }

        // Build training data: for states where NNUE was the current player,
        // target = 1.0 if NNUE won, 0.0 if lost, 0.5 if tie
        let mut all_states: Vec<&GameState> = Vec::new();
        let mut all_targets: Vec<f32> = Vec::new();

        for (i, (states, result)) in game_results.iter().enumerate() {
            let seed = seed_base + i as u64;
            let nnue_is_top = seed % 2 == 0;
            let nnue_owner = if nnue_is_top { Owner::TopPlayer } else { Owner::BottomPlayer };

            let nnue_outcome = match result {
                GameResult::Won(winner) => {
                    if *winner == nnue_owner { 1.0f32 } else { 0.0f32 }
                }
                GameResult::Tie => 0.5f32,
                GameResult::Ongoing => continue,
            };

            for state in states {
                if state.game_result() != GameResult::Ongoing { continue; }
                let current = state.current_player_turn();
                // Target from current player's perspective
                let target = if current == nnue_owner {
                    nnue_outcome
                } else {
                    1.0 - nnue_outcome
                };
                all_states.push(state);
                all_targets.push(target);
            }
        }

        // Backprop: encode states, compute loss, update
        if all_states.len() >= 2 {
            let loss = train_on_outcome(&mut trainer, &all_states, &all_targets);

            total_games_played += games_per_iter as u64;

            // Log
            if iter % 10 == 0 || iter == iterations - 1 {
                println!(
                    "iter {:>4}: NNUE={} Heur={} Tie={}, loss={:.6}, lr={:.6}, states={}, {:.1?}",
                    iter, nnue_wins, heur_wins, ties, loss, lr,
                    all_states.len(), iter_start.elapsed()
                );
            }
        }

        // Update NNUE evaluator with new weights
        nnue_weights = export_weights(&trainer.model, l1_size, l2_size);
        nnue_evaluator = NnueEvaluator::new(nnue_weights);

        // Checkpoint + evaluation
        if (iter + 1) % checkpoint_interval == 0 || iter == iterations - 1 {
            let nnue_path = format!("{}/nnue_iter_{}.nnue", checkpoint_dir, iter + 1);
            let w = export_weights(&trainer.model, l1_size, l2_size);
            w.save(&nnue_path).expect("Failed to save NNUE");

            // Benchmark
            let eval_nnue = NnueEvaluator::new(w);
            let nnue_player = Player::Evaluator(&eval_nnue);
            let heur_player = Player::Evaluator(&StaticHeuristicEvaluator::new());
            print!("  EVAL: ");
            run_matches(&gs, &nnue_player, &heur_player, eval_games,
                &format!("NNUE(iter={}) vs Heuristic", iter + 1));
        }
    }

    println!("\nRL training complete: {} iterations, {} games in {:.1?}",
        iterations, total_games_played, start.elapsed());
}

/// Train on game outcomes: encode states, compute MSE loss vs targets, backprop.
fn train_on_outcome<B: burn::tensor::backend::AutodiffBackend>(
    trainer: &mut FcTdTrainer<B>,
    states: &[&GameState],
    targets: &[f32],
) -> f32 {
    assert_eq!(states.len(), targets.len(),
        "states/targets length mismatch: {} states vs {} targets",
        states.len(), targets.len());
    use burn::prelude::*;
    use burn::optim::GradientsParams;

    let device = trainer.device().clone();
    let tensors: Vec<Tensor<B, 1>> = states
        .iter()
        .map(|gs| encode_state_flat::<B>(gs, &device))
        .collect();
    let batch = Tensor::stack(tensors, 0);
    let predictions = trainer.model.forward(batch);
    let predictions = predictions.squeeze::<1>(1);

    let target_tensor = Tensor::<B, 1>::from_floats(targets, &device);

    let diff = predictions - target_tensor;
    let loss = diff.clone().mul(diff).mean();

    let loss_value: f32 = loss
        .clone()
        .into_data()
        .to_vec::<f32>()
        .expect("loss")[0];

    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &trainer.model);
    trainer.model = trainer.optimizer_step(grads);
    loss_value
}
