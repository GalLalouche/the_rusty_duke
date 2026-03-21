//! Binary: play N random games, train heuristic weights via ridge regression, save to file.

use rand::rngs::StdRng;
use rand::SeedableRng;

use duke_training::game_setup::{create_bag, create_initial_state, play_random_game};
use duke_training::learned_heuristic::train_weights;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let num_games: usize = args
        .iter()
        .position(|a| a == "--games")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let output_path = args
        .iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("learned_heuristic.json");

    let seed: u64 = args
        .iter()
        .position(|a| a == "--seed")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    println!("Playing {} random games (seed={})...", num_games, seed);

    let bag = create_bag();
    let gs = create_initial_state(&bag);
    let mut rng = StdRng::seed_from_u64(seed);

    let mut games = Vec::with_capacity(num_games);
    for i in 0..num_games {
        let (states, result) = play_random_game(&gs, &mut rng);
        if (i + 1) % 100 == 0 || i + 1 == num_games {
            eprintln!("  Game {}/{}: {} states, result={:?}", i + 1, num_games, states.len(), result);
        }
        games.push((states, result));
    }

    println!("Training weights on {} games...", games.len());
    let weights = train_weights(&games);

    println!("Learned weights:");
    for (i, w) in weights.weights.iter().enumerate() {
        println!("  w[{:2}] = {:+.6e}", i, w);
    }

    weights.save(output_path).expect("Failed to save weights");
    println!("Weights saved to: {}", output_path);
}
