use duke_rust::game::ai::heuristics::{HeuristicAi, Heuristics};

use duke_training::game_setup::{create_bag, create_initial_state, HeuristicEvaluator};
use duke_training::match_runner::{run_matches, Player};
use duke_training::nnue::NnueEvaluator;

fn run_all_benchmarks(nnue_path: &str) {
    let bag = create_bag();
    let gs = create_initial_state(&bag);

    println!("Loading NNUE weights from: {}", nnue_path);
    let weights =
        duke_training::nnue::NnueWeights::load(nnue_path).expect("Failed to load NNUE weights");
    let nnue_evaluator = NnueEvaluator::new(weights);
    println!("NNUE weights loaded.\n");

    let heuristic_ai = HeuristicAi::new(vec![
        Box::new(Heuristics::DukeMovementOptions),
        Box::new(Heuristics::TotalTilesOnBoard),
        Box::new(Heuristics::TotalMovementOptions),
        Box::new(Heuristics::DiscardedUnits),
    ]);
    let heuristic_evaluator = HeuristicEvaluator::new(&heuristic_ai);

    let random = Player::Random;
    let heuristic = Player::Evaluator(&heuristic_evaluator);
    let nnue_player = Player::Evaluator(&nnue_evaluator);

    let num_games = std::env::var("NUM_GAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200u32);

    println!("=== Benchmark: {} games per matchup ===\n", num_games);

    println!("--- NNUE vs Random ---");
    run_matches(&gs, &nnue_player, &random, num_games, "NNUE vs Random");

    println!("\n--- NNUE vs Heuristic ---");
    run_matches(
        &gs,
        &nnue_player,
        &heuristic,
        num_games,
        "NNUE vs Heuristic",
    );

    println!("\n--- Heuristic vs Random (baseline) ---");
    run_matches(&gs, &heuristic, &random, num_games, "Heuristic vs Random");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let nnue_path = args
        .iter()
        .position(|a| a == "--nnue")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    if let Some(path) = nnue_path {
        run_all_benchmarks(path);
    } else {
        eprintln!("Usage: benchmark --nnue <weights_path>");
        eprintln!("  e.g.: benchmark --nnue model.nnue");
        std::process::exit(1);
    }
}
