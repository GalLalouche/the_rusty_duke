//! Extract heuristic features from saved game trajectories and cache to disk.
//!
//! Usage: extract_features --trajectories <path.dtrj> --output <features.bin> [--cheap]

use std::time::Instant;

use duke_training::feature_cache::{CachedGame, CachedState, save_feature_cache};
use duke_training::learned_heuristic::{extract_features, extract_combined_features, NUM_FEATURES, NUM_COMBINED_FEATURES};
use duke_training::trajectory_io::load_trajectories;

use duke_rust::game::state::GameResult;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let traj_path = args.iter().position(|a| a == "--trajectories")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .expect("Usage: extract_features --trajectories <path.dtrj> [--output <features.bin>] [--cheap]");

    let output_path = args.iter().position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("features.bin");

    let combined = args.iter().any(|a| a == "--combined");
    let num_features = if combined { NUM_COMBINED_FEATURES } else { NUM_FEATURES };

    println!("Loading trajectories from: {}", traj_path);
    let t = Instant::now();
    let games = load_trajectories(traj_path).expect("Failed to load trajectories");
    let total_states: usize = games.iter().map(|g| g.states.len()).sum();
    println!("Loaded {} games ({} states) in {:.1?}", games.len(), total_states, t.elapsed());

    println!("Extracting {} {} features per state...", num_features, if combined { "combined" } else { "full (legacy)" });
    let t = Instant::now();
    let mut cached_games: Vec<CachedGame> = Vec::with_capacity(games.len());
    let mut n_samples = 0usize;

    for (i, game) in games.iter().enumerate() {
        let mut cached_states = Vec::with_capacity(game.states.len());
        for state in &game.states {
            if state.game_result() != GameResult::Ongoing {
                continue;
            }
            let features: Vec<f64> = if combined {
                extract_combined_features(state).to_vec()
            } else {
                extract_features(state).to_vec()
            };
            cached_states.push(CachedState {
                current_player: state.current_player_turn(),
                features,
            });
            n_samples += 1;
        }
        cached_games.push(CachedGame {
            result: game.result,
            states: cached_states,
        });

        if (i + 1) % 10000 == 0 {
            eprintln!("  {}/{} games ({} samples, {:.1?})", i + 1, games.len(), n_samples, t.elapsed());
        }
    }
    println!("Extracted {} samples in {:.1?}", n_samples, t.elapsed());

    println!("Saving to: {}", output_path);
    let t = Instant::now();
    save_feature_cache(output_path, num_features, &cached_games)
        .expect("Failed to save feature cache");
    println!("Saved in {:.1?}", t.elapsed());
}
