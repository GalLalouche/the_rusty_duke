use duke_rust::game::ai::heuristics::{HeuristicAi, Heuristics};
use duke_rust::game::state::GameState;

use duke_training::game_setup::{create_bag, create_initial_state, HeuristicEvaluator};
use duke_training::learned_heuristic::LearnedHeuristicWeights;
use duke_training::match_runner::{run_matches, Player};
use duke_training::nnue::NnueEvaluator;

fn num_games() -> u32 {
    std::env::var("NUM_GAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200u32)
}

fn create_heuristic_ai() -> HeuristicAi {
    HeuristicAi::new(vec![
        Box::new(Heuristics::DukeMovementOptions),
        Box::new(Heuristics::TotalTilesOnBoard),
        Box::new(Heuristics::TotalMovementOptions),
        Box::new(Heuristics::DiscardedUnits),
    ])
}

fn run_learned_matchups(
    gs: &GameState,
    learned: &LearnedHeuristicWeights,
    heuristic: &Player,
    random: &Player,
    nnue_player: Option<&Player>,
    num_games: u32,
) {
    let learned_player = Player::Evaluator(learned);

    println!("\n--- Learned vs Random ---");
    run_matches(gs, &learned_player, random, num_games, "Learned vs Random");

    println!("\n--- Learned vs Heuristic ---");
    run_matches(gs, &learned_player, heuristic, num_games, "Learned vs Heuristic");

    if let Some(nnue) = nnue_player {
        println!("\n--- Learned vs NNUE ---");
        run_matches(gs, &learned_player, nnue, num_games, "Learned vs NNUE");
    }
}

fn run_all_benchmarks(nnue_path: &str, learned_weights: Option<&LearnedHeuristicWeights>) {
    let bag = create_bag();
    let gs = create_initial_state(&bag);

    println!("Loading NNUE weights from: {}", nnue_path);
    let weights =
        duke_training::nnue::NnueWeights::load(nnue_path).expect("Failed to load NNUE weights");
    let nnue_evaluator = NnueEvaluator::new(weights);
    println!("NNUE weights loaded.\n");

    let heuristic_ai = create_heuristic_ai();
    let heuristic_evaluator = HeuristicEvaluator::new(&heuristic_ai);

    let random = Player::Random;
    let heuristic = Player::Evaluator(&heuristic_evaluator);
    let nnue_player = Player::Evaluator(&nnue_evaluator);

    let num_games = num_games();

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

    if let Some(learned) = learned_weights {
        run_learned_matchups(&gs, learned, &heuristic, &random, Some(&nnue_player), num_games);
    }
}

fn run_learned_only(learned_weights: &LearnedHeuristicWeights) {
    let bag = create_bag();
    let gs = create_initial_state(&bag);

    let heuristic_ai = create_heuristic_ai();
    let heuristic_evaluator = HeuristicEvaluator::new(&heuristic_ai);

    let random = Player::Random;
    let heuristic = Player::Evaluator(&heuristic_evaluator);

    let num_games = num_games();

    println!("=== Benchmark (learned only): {} games per matchup ===\n", num_games);

    run_learned_matchups(&gs, learned_weights, &heuristic, &random, None, num_games);

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

    let learned_path = args
        .iter()
        .position(|a| a == "--learned")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    let learned_weights = learned_path.map(|path| {
        println!("Loading learned heuristic weights from: {}", path);
        let w = LearnedHeuristicWeights::load(path).expect("Failed to load learned heuristic weights");
        println!("Learned heuristic weights loaded.\n");
        w
    });

    match (nnue_path, &learned_weights) {
        (Some(nnue), _) => run_all_benchmarks(nnue, learned_weights.as_ref()),
        (None, Some(learned)) => run_learned_only(learned),
        (None, None) => {
            eprintln!("Usage: benchmark --nnue <weights_path> [--learned <weights_path>]");
            eprintln!("       benchmark --learned <weights_path>");
            eprintln!("  e.g.: benchmark --nnue model.nnue");
            eprintln!("        benchmark --learned learned_heuristic.json");
            eprintln!("        benchmark --nnue model.nnue --learned learned_heuristic.json");
            std::process::exit(1);
        }
    }
}
