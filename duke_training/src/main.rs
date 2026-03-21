use std::sync::Arc;
use std::time::Instant;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use duke_rust::game::ai::player::{AiMove, ArtificialPlayer};
use duke_rust::game::ai::stupid_sync_ai::StupidSyncAi;
use duke_rust::game::bag::TileBag;
use duke_rust::game::board_setup::{DukeInitialLocation, FootmenSetup};
use duke_rust::game::state::{GameResult, GameState};
use duke_rust::game::units;

use duke_training::fc_td_training::FcTdTrainer;
use duke_training::nnue::{NnueEvaluator, NnueWeights};
use duke_training::weight_export::export_weights;

type MyBackend = Autodiff<Wgpu>;

fn create_bag() -> TileBag {
    TileBag::new(vec![
        Arc::new(units::footman()),
        Arc::new(units::bowman()),
        Arc::new(units::knight()),
        Arc::new(units::pikeman()),
        Arc::new(units::pikeman()),
        Arc::new(units::champion()),
        Arc::new(units::priest()),
        Arc::new(units::wizard()),
        Arc::new(units::dragoon()),
        // Assassin excluded: JumpSlide not yet implemented in board logic
        Arc::new(units::general()),
        Arc::new(units::marshall()),
        Arc::new(units::longbowman()),
    ])
}

fn create_initial_state(bag: &TileBag) -> GameState {
    GameState::new(
        bag,
        (DukeInitialLocation::Left, FootmenSetup::Left),
        (DukeInitialLocation::Right, FootmenSetup::Right),
    )
}

/// Play a game using random moves, collecting states at each turn.
fn play_random_game(gs: &GameState, rng: &mut StdRng) -> (Vec<GameState>, GameResult) {
    let ai = StupidSyncAi {};
    let mut game = gs.clone();
    let mut states = Vec::new();

    loop {
        match game.game_result() {
            GameResult::Ongoing => {
                states.push(game.clone());
                ai.play_next_move(rng, &mut game);
            }
            result => {
                states.push(game.clone());
                return (states, result);
            }
        }
    }
}

/// Play a game using NNUE self-play with epsilon-greedy exploration.
/// With probability `epsilon`, picks a random move; otherwise picks the
/// move that maximizes the NNUE evaluation.
fn play_nnue_game(
    gs: &GameState,
    evaluator: &NnueEvaluator,
    rng: &mut StdRng,
    epsilon: f64,
) -> (Vec<GameState>, GameResult) {
    let ai = StupidSyncAi {};
    let mut game = gs.clone();
    let mut states = Vec::new();

    loop {
        match game.game_result() {
            GameResult::Ongoing => {
                states.push(game.clone());

                if rng.gen::<f64>() < epsilon {
                    // Random exploration move
                    ai.play_next_move(rng, &mut game);
                } else {
                    // Greedy NNUE move
                    let mv = nnue_greedy_move(&game, evaluator, rng);
                    mv.play(&mut game, rng);
                }
            }
            result => {
                states.push(game.clone());
                return (states, result);
            }
        }
    }
}

/// Pick the move that minimizes the opponent's value (= maximizes our value).
fn nnue_greedy_move(gs: &GameState, evaluator: &NnueEvaluator, rng: &mut impl Rng) -> AiMove {
    let mut moves: Vec<AiMove> = AiMove::all_moves(gs).collect();
    moves.shuffle(rng); // randomize among equal-scored moves

    let mut best_score = f64::NEG_INFINITY;
    let mut best_move = None;

    for mv in &moves {
        let mut clone = gs.clone();
        mv.play(&mut clone, rng);
        let prediction = evaluator.evaluate_state(&clone);
        // After our move it's opponent's turn, so opponent's value = prediction.
        // Our value = 1 - prediction.
        let score = 1.0 - prediction as f64;
        if score > best_score {
            best_score = score;
            best_move = Some(mv.clone());
        }
    }

    best_move.unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let self_play = args.iter().any(|a| a == "--self-play");
    let total_games: u64 = args.iter()
        .position(|a| a == "--games")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let epsilon: f64 = args.iter()
        .position(|a| a == "--epsilon")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.15);
    let update_interval: u64 = args.iter()
        .position(|a| a == "--update-interval")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let bag = create_bag();
    let gs = create_initial_state(&bag);

    let device = WgpuDevice::default();
    let mut trainer: FcTdTrainer<MyBackend> = FcTdTrainer::new(device, 0.001);

    // Initialize NNUE evaluator from the fresh model
    let mut nnue_weights = export_weights(&trainer.model);
    let mut nnue_evaluator = NnueEvaluator::new(nnue_weights);

    if self_play {
        println!("Phase 2: NNUE self-play training");
        println!("  epsilon={}, update_interval={}, total_games={}", epsilon, update_interval, total_games);
    } else {
        println!("Phase 1: Random play training");
        println!("  total_games={}", total_games);
    }

    let start = Instant::now();
    let mut total_loss = 0.0f32;
    let mut recent_loss = 0.0f32;
    let mut wins = [0u32; 2];
    let mut ties = 0u32;

    for game_num in 0..total_games {
        let mut rng = StdRng::seed_from_u64(game_num);

        let (states, result) = if self_play {
            play_nnue_game(&gs, &nnue_evaluator, &mut rng, epsilon)
        } else {
            play_random_game(&gs, &mut rng)
        };

        let loss = trainer.train_on_game(&states, result);
        total_loss += loss;
        recent_loss += loss;

        match result {
            GameResult::Won(duke_rust::game::tile::Owner::TopPlayer) => wins[0] += 1,
            GameResult::Won(duke_rust::game::tile::Owner::BottomPlayer) => wins[1] += 1,
            GameResult::Tie => ties += 1,
            _ => {}
        }

        if (game_num + 1) % 100 == 0 {
            let avg_loss = total_loss / (game_num + 1) as f32;
            let recent_avg = recent_loss / 100.0;
            let elapsed = start.elapsed();
            println!(
                "Game {}: avg_loss={:.6}, recent_loss={:.6}, wins=[{}, {}], ties={}, elapsed={:.1?}",
                game_num + 1, avg_loss, recent_avg, wins[0], wins[1], ties, elapsed
            );
            recent_loss = 0.0;
        }

        if (game_num + 1) % update_interval == 0 {
            // Save checkpoints
            let checkpoint_path = format!("checkpoints/fc_model_game_{}", game_num + 1);
            std::fs::create_dir_all("checkpoints").expect("Failed to create checkpoints dir");
            trainer.save_model(&checkpoint_path);

            let nnue_path = format!("checkpoints/nnue_game_{}.nnue", game_num + 1);
            let new_weights = export_weights(&trainer.model);
            new_weights.save(&nnue_path).expect("Failed to save NNUE weights");

            if self_play {
                // Update NNUE evaluator with latest trained weights
                nnue_evaluator = NnueEvaluator::new(new_weights);
                println!("NNUE weights updated + checkpoint saved: {}", nnue_path);
            } else {
                println!("Checkpoint saved: {}", nnue_path);
            }
        }
    }

    let elapsed = start.elapsed();
    println!("\nTraining complete: {} games in {:.1?}", total_games, elapsed);
    println!("Avg time per game: {:.1?}", elapsed / total_games as u32);
    println!("Final avg loss: {:.6}", total_loss / total_games as f32);
}
