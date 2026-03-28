//! TD(lambda) self-play training binary for pure NN.
//!
//! Trains a neural network via temporal difference learning with self-play,
//! similar to TD-Gammon. Uses the burn-based FcValueNetwork with variable depth.
//!
//! Usage:
//!   td_selfplay --games 1500000 --lr 0.001 --lambda 0.7 --epsilon 0.1
//!               --hidden 128 --batch-size 100 --eval-interval 10000
//!               --eval-games 500 --benchmark base,random --checkpoint-dir D:/temp/td_selfplay
//!
//! The --benchmark flag accepts a comma-separated list of opponents (default: base,random).
//! Each opponent can be "base", "random", or a file path (.gmlp/.nnue/.json).

use std::time::Instant;

use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::tensor::backend::Backend;
use rand::rngs::StdRng;
use rand::SeedableRng;

use duke_rust::game::state::{GameResult, GameState};

use duke_training::cli::parse_flag;
use duke_training::encoding::{encode_state_flat, TOTAL_FEATURES};
use duke_training::fc_model::FcValueNetwork;
use duke_training::fc_td_training::{FcTdTrainer, GameTrajectory};
use duke_training::game_setup::{create_bag, create_initial_state, play_selfplay_game};
use duke_training::generic_mlp::{GenericMlp, GenericEvaluator};
use duke_training::loaded_model::LoadedModel;
use duke_training::match_runner::{run_matches, Player};

type MyBackend = Autodiff<NdArray>;

/// Extract weights from a burn FcValueNetwork and build a GenericMlp.
///
/// Works for any number of layers. Burn Linear stores weights as [d_input, d_output]
/// row-major, and GenericMlp stores them as [prev * cur] with the same row-major
/// indexing (weight[i * cur + j] = input i to output j), so no transpose is needed.
fn to_generic_mlp<B: Backend>(model: &FcValueNetwork<B>, hidden_sizes: &[usize]) -> GenericMlp {
    let num_layers = model.num_layers();
    assert_eq!(num_layers, hidden_sizes.len() + 1,
        "model has {} layers but hidden_sizes has {} entries",
        num_layers, hidden_sizes.len());

    let mut flat: Vec<f32> = Vec::new();

    // Build the sizes list: [input_size, h1, h2, ..., 1]
    let mut sizes = Vec::with_capacity(num_layers + 1);
    sizes.push(TOTAL_FEATURES);
    sizes.extend_from_slice(hidden_sizes);
    sizes.push(1);

    for (i, layer) in model.layers.iter().enumerate() {
        let d_in = sizes[i];
        let d_out = sizes[i + 1];

        // Extract weight [d_in, d_out] row-major -- same layout as GenericMlp
        let weight: Vec<f32> = layer.weight.val().into_data()
            .to_vec().expect("weight extraction");
        assert_eq!(weight.len(), d_in * d_out,
            "Layer {} weight size mismatch: expected {}x{}={}, got {}",
            i, d_in, d_out, d_in * d_out, weight.len());
        flat.extend_from_slice(&weight);

        // Extract bias
        let bias: Vec<f32> = layer.bias.as_ref()
            .expect("layer must have bias")
            .val().into_data()
            .to_vec().expect("bias extraction");
        assert_eq!(bias.len(), d_out,
            "Layer {} bias size mismatch: expected {}, got {}",
            i, d_out, bias.len());
        flat.extend_from_slice(&bias);
    }

    GenericMlp::from_flat(flat, TOTAL_FEATURES, hidden_sizes.to_vec())
}

/// Train on a batch of trajectories using TD(lambda) targets.
///
/// For non-terminal states:
///   target = lambda * (1 - V(s_{t+1})) + (1 - lambda) * outcome
/// For terminal states:
///   target = outcome
///
/// This blends the TD(0) bootstrap target with the Monte Carlo outcome.
fn train_td_lambda<B: burn::tensor::backend::AutodiffBackend>(
    trainer: &mut FcTdTrainer<B>,
    games: &[GameTrajectory],
    lambda: f32,
) -> f32 {
    use burn::optim::GradientsParams;
    // Flatten all states and record game boundary information
    let mut all_states: Vec<&GameState> = Vec::new();
    // For each state: (is_terminal, game_outcome_for_current_player)
    struct StateInfo {
        is_terminal: bool,
        outcome: f32, // outcome from current player's perspective
    }
    let mut state_info: Vec<StateInfo> = Vec::new();

    for game in games {
        if game.states.len() < 2 {
            continue;
        }
        let n = game.states.len();
        for (i, state) in game.states.iter().enumerate() {
            let current_player = state.current_player_turn();
            let outcome = match game.result {
                GameResult::Won(winner) => {
                    if winner == current_player { 1.0f32 } else { 0.0f32 }
                }
                GameResult::Tie => 0.5f32,
                GameResult::Ongoing => unreachable!("Game should be finished"),
            };
            let is_terminal = i == n - 1;
            all_states.push(state);
            state_info.push(StateInfo { is_terminal, outcome });
        }
    }

    if all_states.len() < 2 {
        return 0.0;
    }

    let total_states = all_states.len();
    let device = trainer.device().clone();

    // Encode all states as a batch and run forward pass
    let tensors: Vec<Tensor<B, 1>> = all_states
        .iter()
        .map(|gs| encode_state_flat::<B>(gs, &device))
        .collect();
    let batch = Tensor::stack(tensors, 0); // [total_states, TOTAL_FEATURES]
    let predictions = trainer.model.forward(batch); // [total_states, 1]
    let predictions = predictions.squeeze::<1>(1); // [total_states]

    // Detach predictions for building targets
    let pred_data: Vec<f32> = predictions
        .clone()
        .into_data()
        .to_vec()
        .expect("Failed to convert predictions to vec");

    // Build TD(lambda) targets
    // We need to track game boundaries properly. We rebuild them by scanning state_info.
    let mut targets = Vec::with_capacity(total_states);
    for t in 0..total_states {
        if state_info[t].is_terminal {
            // Terminal state: target = outcome
            targets.push(state_info[t].outcome);
        } else {
            // Non-terminal: blend TD(0) bootstrap with Monte Carlo outcome
            // TD(0) target: 1 - V(s_{t+1}) (opponent's perspective)
            let td0_target = 1.0 - pred_data[t + 1];
            // MC target: outcome from current player's perspective
            let mc_target = state_info[t].outcome;
            // TD(lambda) blend
            let target = lambda * td0_target + (1.0 - lambda) * mc_target;
            targets.push(target);
        }
    }

    let target_tensor =
        Tensor::<B, 1>::from_floats(targets.as_slice(), &device);

    // MSE loss
    let diff = predictions - target_tensor;
    let loss = diff.clone().mul(diff).mean();

    let loss_value: f32 = loss
        .clone()
        .into_data()
        .to_vec::<f32>()
        .expect("loss")[0];

    // Backward pass and optimizer step
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &trainer.model);
    trainer.model = trainer.optimizer_step(grads);

    loss_value
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse CLI flags
    let hidden_str = parse_flag::<String>(&args, "--hidden")
        .unwrap_or_else(|| "128".to_string());
    let hidden_sizes: Vec<usize> = hidden_str
        .split(',')
        .map(|s| s.trim().parse::<usize>().expect("invalid hidden size"))
        .collect();
    let total_games: u64 = parse_flag(&args, "--games").unwrap_or(1_500_000);
    let lr: f64 = parse_flag(&args, "--lr").unwrap_or(0.001);
    let lambda: f32 = parse_flag(&args, "--lambda").unwrap_or(0.7);
    let epsilon: f64 = parse_flag(&args, "--epsilon").unwrap_or(0.1);
    let batch_size: u64 = parse_flag(&args, "--batch-size").unwrap_or(100);
    let eval_interval: u64 = parse_flag(&args, "--eval-interval").unwrap_or(10_000);
    let eval_games: u32 = parse_flag(&args, "--eval-games").unwrap_or(500);
    let benchmark_str = parse_flag::<String>(&args, "--benchmark")
        .unwrap_or_else(|| "base,random".to_string());
    let benchmark_specs: Vec<String> = benchmark_str
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let checkpoint_dir = parse_flag::<String>(&args, "--checkpoint-dir")
        .unwrap_or_else(|| "D:/temp/td_selfplay".to_string());

    // Print configuration
    let arch_str = hidden_sizes.iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",");
    println!("TD(lambda) Self-Play Training");
    println!("  network: 1106->{}->1", arch_str.replace(",", "->"));
    println!("  games={}, lr={}, lambda={}, epsilon={}", total_games, lr, lambda, epsilon);
    println!("  batch_size={}, eval_interval={}, eval_games={}", batch_size, eval_interval, eval_games);
    println!("  benchmark={}, checkpoint_dir={}", benchmark_str, checkpoint_dir);
    println!();

    // Create checkpoint directory
    std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoint dir");

    // Initialize game state
    let bag = create_bag();
    let gs = create_initial_state(&bag);

    // Load benchmark opponents once at startup
    let benchmark_opponents: Vec<LoadedModel> = benchmark_specs
        .iter()
        .map(|spec| LoadedModel::from_spec(spec, false))
        .collect();
    println!("Loaded {} benchmark opponent(s): {}",
        benchmark_opponents.len(),
        benchmark_opponents.iter().map(|o| o.label.as_str()).collect::<Vec<_>>().join(", "));
    println!();

    // Create trainer with NdArray backend (CPU)
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let mut trainer: FcTdTrainer<MyBackend> =
        FcTdTrainer::new(device.clone(), lr, &hidden_sizes);

    // Main training loop
    let mut trajectories: Vec<GameTrajectory> = Vec::new();
    let mut games_played: u64 = 0;
    let mut last_eval_at: u64 = 0;
    let mut cumulative_loss: f32 = 0.0;
    let mut loss_count: u32 = 0;
    let start = Instant::now();
    let mut last_log_at: u64 = 0;

    while games_played < total_games {
        // Extract burn weights -> GenericMlp once per batch for fast inference
        let gmlp = to_generic_mlp(&trainer.model.valid(), &hidden_sizes);
        let evaluator = GenericEvaluator { net: gmlp };

        // Play a batch of games using fast GenericMlp sparse inference
        let games_this_batch = batch_size.min(total_games - games_played);
        for i in 0..games_this_batch {
            let seed = games_played + i;
            let mut rng = StdRng::seed_from_u64(seed);

            let (states, result) = play_selfplay_game(&gs, &evaluator, &mut rng, epsilon);
            if result != GameResult::Ongoing {
                trajectories.push(GameTrajectory { states, result });
            }
        }

        games_played += games_this_batch;

        // Train on accumulated trajectories
        if !trajectories.is_empty() {
            let loss = train_td_lambda(&mut trainer, &trajectories, lambda);
            cumulative_loss += loss;
            loss_count += 1;
            trajectories.clear();
        }

        // Progress output every 1000 games
        if games_played / 1000 > last_log_at / 1000 {
            let elapsed = start.elapsed().as_secs_f64();
            let games_per_sec = games_played as f64 / elapsed;
            let avg_loss = if loss_count > 0 {
                cumulative_loss / loss_count as f32
            } else {
                0.0
            };
            println!(
                "[{}/{}] loss={:.4} epsilon={:.2} ({:.1} games/sec)",
                games_played, total_games, avg_loss, epsilon, games_per_sec,
            );
            last_log_at = games_played;
        }

        // Evaluation and checkpoint
        if games_played >= last_eval_at + eval_interval || games_played >= total_games {
            last_eval_at = games_played;

            // Save checkpoint as .gmlp
            let gmlp = to_generic_mlp(&trainer.model.valid(), &hidden_sizes);
            let ckpt_path = format!("{}/td_iter_{}.gmlp", checkpoint_dir, games_played);
            gmlp.save(&ckpt_path).expect("Failed to save checkpoint");

            // Benchmark against all opponents
            let gmlp_eval = GenericEvaluator { net: gmlp };
            let trained_player = Player::Evaluator(&gmlp_eval);

            println!("EVAL ({} games):", games_played);
            for opponent in &benchmark_opponents {
                let opp_player = opponent.as_player();
                run_matches(
                    &gs,
                    &trained_player,
                    &opp_player,
                    eval_games,
                    &format!("  vs {}", opponent.label),
                );
            }

            println!("Saved checkpoint: {}", ckpt_path);
            println!();
        }
    }

    let total_elapsed = start.elapsed();
    println!(
        "Training complete: {} games in {:.1?} ({:.1} games/sec)",
        games_played,
        total_elapsed,
        games_played as f64 / total_elapsed.as_secs_f64(),
    );
}
