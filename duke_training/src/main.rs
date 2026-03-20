use std::sync::Arc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use duke_rust::game::ai::player::ArtificialPlayer;
use duke_rust::game::ai::stupid_sync_ai::StupidSyncAi;
use duke_rust::game::bag::TileBag;
use duke_rust::game::board_setup::{DukeInitialLocation, FootmenSetup};
use duke_rust::game::state::{GameResult, GameState};
use duke_rust::game::tile::Owner;
use duke_rust::game::units;

fn main() {
    let bag = TileBag::new(vec![
        Arc::new(units::footman()),
        Arc::new(units::bowman()),
        Arc::new(units::knight()),
        Arc::new(units::pikeman()),
        Arc::new(units::pikeman()),
        Arc::new(units::champion()),
        Arc::new(units::priest()),
        Arc::new(units::wizard()),
        Arc::new(units::dragoon()),
        Arc::new(units::assassin()),
        Arc::new(units::general()),
        Arc::new(units::marshall()),
        Arc::new(units::longbowman()),
    ]);

    let mut gs = GameState::new(
        &bag,
        (DukeInitialLocation::Left, FootmenSetup::Left),
        (DukeInitialLocation::Right, FootmenSetup::Right),
    );

    let mut rng = StdRng::seed_from_u64(42);
    let mut completed = 0u32;
    let mut panicked = 0u32;
    let total_games = 1000;

    let start = Instant::now();

    for seed in 0..total_games {
        let mut game = gs.clone();
        let mut game_rng = StdRng::seed_from_u64(seed);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let ai = StupidSyncAi {};
            let mut turns = 0u32;
            loop {
                match game.game_result() {
                    GameResult::Ongoing => {
                        ai.play_next_move(&mut game_rng, &mut game);
                        turns += 1;
                    }
                    GameResult::Won(winner) => return (turns, Some(winner)),
                    GameResult::Tie => return (turns, None),
                }
            }
        }));
        match result {
            Ok((turns, winner)) => {
                completed += 1;
                if seed < 5 {
                    println!("Game {}: {} turns, winner: {:?}", seed, turns, winner);
                }
            }
            Err(_) => panicked += 1,
        }
    }

    let elapsed = start.elapsed();
    println!("\n{} games completed, {} panicked (pre-existing bugs)", completed, panicked);
    println!("Total time: {:.3?}", elapsed);
    println!("Avg per game: {:.3?}", elapsed / completed);
}
