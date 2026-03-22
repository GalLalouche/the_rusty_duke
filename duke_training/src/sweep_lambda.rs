//! Sweep ridge regression lambda values on saved game trajectories.

use std::time::Instant;

use duke_training::game_setup::{create_bag, create_initial_state, StaticHeuristicEvaluator};
use duke_training::learned_heuristic::RegressionAccumulator;
use duke_training::match_runner::{run_matches, Player};
use duke_training::trajectory_io::load_trajectories;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let traj_path = args.iter().position(|a| a == "--trajectories")
        .and_then(|i| args.get(i + 1))
        .expect("Usage: sweep_lambda --trajectories <path.dtrj> [--bench-games N]");

    let bench_games: u32 = args.iter().position(|a| a == "--bench-games")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    let t = Instant::now();
    let games = load_trajectories(traj_path).expect("Failed to load trajectories");
    let total_states: usize = games.iter().map(|g| g.states.len()).sum();
    println!("Loaded {} games ({} states) in {:.1?}", games.len(), total_states, t.elapsed());

    let t = Instant::now();
    let mut acc = RegressionAccumulator::new();
    for game in &games {
        acc.add_game(&game.states, &game.result);
    }
    println!("Accumulated {} samples in {:.1?}\n", acc.n_samples(), t.elapsed());

    let bag = create_bag();
    let gs = create_initial_state(&bag);
    let heuristic_eval = StaticHeuristicEvaluator::new();
    let heuristic_player = Player::Evaluator(&heuristic_eval);

    let lambdas = [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0];

    println!("=== Lambda sweep: {} games per matchup vs Heuristic ===\n", bench_games);
    println!("{:>12} {:>8} {:>8} {:>8}", "lambda", "Win%", "Loss%", "Tie%");
    println!("{}", "-".repeat(44));

    for &lambda in &lambdas {
        let weights = acc.solve_with_lambda(lambda);
        let lr_player = Player::Evaluator(&weights);
        let label = format!("LR(l={:.0e})", lambda);
        let result = run_matches(&gs, &lr_player, &heuristic_player, bench_games, &label);

        let total = bench_games as f64;
        println!("{:>12.0e} {:>7.1}% {:>7.1}% {:>7.1}%",
            lambda,
            result.player_a_wins as f64 / total * 100.0,
            result.player_b_wins as f64 / total * 100.0,
            result.ties as f64 / total * 100.0,
        );
    }
}
