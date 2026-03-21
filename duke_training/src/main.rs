use std::sync::Arc;
use std::time::Instant;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use rand::rngs::StdRng;
use rand::SeedableRng;

use duke_rust::game::ai::player::ArtificialPlayer;
use duke_rust::game::ai::stupid_sync_ai::StupidSyncAi;
use duke_rust::game::bag::TileBag;
use duke_rust::game::board_setup::{DukeInitialLocation, FootmenSetup};
use duke_rust::game::state::{GameResult, GameState};
use duke_rust::game::units;

use duke_training::td_training::TdTrainer;

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

/// Play a random game, collecting states at each turn.
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

fn main() {
    let bag = create_bag();
    let gs = create_initial_state(&bag);

    let device = WgpuDevice::default();
    let mut trainer: TdTrainer<MyBackend> = TdTrainer::new(device, 0.001);

    let total_games: u64 = 1000;
    let start = Instant::now();
    let mut total_loss = 0.0f32;
    let mut wins = [0u32; 2]; // [TopPlayer, BottomPlayer]
    let mut ties = 0u32;

    for seed in 0..total_games {
        let mut rng = StdRng::seed_from_u64(seed);
        let (states, result) = play_random_game(&gs, &mut rng);
        let loss = trainer.train_on_game(&states, result);
        total_loss += loss;

        match result {
            GameResult::Won(duke_rust::game::tile::Owner::TopPlayer) => wins[0] += 1,
            GameResult::Won(duke_rust::game::tile::Owner::BottomPlayer) => wins[1] += 1,
            GameResult::Tie => ties += 1,
            _ => {}
        }

        if (seed + 1) % 100 == 0 {
            let avg_loss = total_loss / (seed + 1) as f32;
            let elapsed = start.elapsed();
            println!(
                "Game {}: avg_loss={:.6}, wins=[{}, {}], ties={}, elapsed={:.1?}",
                seed + 1, avg_loss, wins[0], wins[1], ties, elapsed
            );
        }

        if (seed + 1) % 1000 == 0 {
            let checkpoint_path = format!("checkpoints/model_game_{}", seed + 1);
            std::fs::create_dir_all("checkpoints").expect("Failed to create checkpoints dir");
            trainer.save_model(&checkpoint_path);
            println!("Checkpoint saved: {}", checkpoint_path);
        }
    }

    let elapsed = start.elapsed();
    println!("\nTraining complete: {} games in {:.1?}", total_games, elapsed);
    println!("Avg time per game: {:.1?}", elapsed / total_games as u32);
    println!("Final avg loss: {:.6}", total_loss / total_games as f32);
}
