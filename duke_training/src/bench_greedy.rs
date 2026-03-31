use std::time::Instant;
use rand::rngs::StdRng;
use rand::SeedableRng;
use duke_rust::game::state::GameResult;
use duke_training::game_setup::{
    create_bag, create_initial_state, greedy_move, GameEvaluator, StaticHeuristicEvaluator,
};

const NUM_GAMES: u32 = 2000;
const MAX_TURNS: u32 = 500;

fn main() {
    let bag = create_bag();
    let init_state = create_initial_state(&bag);
    let eval = StaticHeuristicEvaluator::new();

    let mut total_moves = 0u64;
    let start = Instant::now();

    for seed in 0..NUM_GAMES {
        let mut game = init_state.clone();
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let mut turns = 0u32;
        loop {
            if game.game_result() != GameResult::Ongoing || turns >= MAX_TURNS {
                break;
            }
            turns += 1;
            let mv = greedy_move(&mut game, &eval, &mut rng);
            mv.play(&mut game, &mut rng);
        }
        total_moves += turns as u64;
    }

    let elapsed = start.elapsed();
    println!("Games: {}, Moves: {}, Time: {:.3}s", NUM_GAMES, total_moves, elapsed.as_secs_f64());
    println!("  {:.1} us/move, {:.1} ms/game",
        elapsed.as_secs_f64() * 1_000_000.0 / total_moves as f64,
        elapsed.as_secs_f64() * 1_000.0 / NUM_GAMES as f64);
}
