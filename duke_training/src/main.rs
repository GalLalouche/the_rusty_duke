use std::sync::Arc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use duke_rust::game::ai::player::ArtificialPlayer;
use duke_rust::game::ai::stupid_sync_ai::StupidSyncAi;
use duke_rust::game::bag::TileBag;
use duke_rust::game::board_setup::{DukeInitialLocation, FootmenSetup};
use duke_rust::game::state::{GameResult, GameState};
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

    let ai = StupidSyncAi {};
    let mut rng = StdRng::seed_from_u64(42);
    let mut turn_count: u32 = 0;

    let start = Instant::now();

    loop {
        let result = gs.game_result();
        match result {
            GameResult::Ongoing => {
                ai.play_next_move(&mut rng, &mut gs);
                turn_count += 1;
            }
            GameResult::Won(winner) => {
                println!("Game over after {} turns. Winner: {:?}", turn_count, winner);
                break;
            }
            GameResult::Tie => {
                println!("Game over after {} turns. Result: Tie", turn_count);
                break;
            }
        }
    }

    let elapsed = start.elapsed();
    println!("Elapsed time: {:.3?}", elapsed);
}
