use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use duke_rust::game::ai::heuristics::{HeuristicAi, Heuristics};
use duke_rust::game::ai::player::ArtificialPlayer;
use duke_rust::game::ai::stupid_sync_ai::StupidSyncAi;
use duke_rust::game::state::{GameResult, GameState};
use duke_rust::game::tile::Owner;

use duke_training::game_setup::{
    create_bag, create_initial_state, greedy_move, GameEvaluator, HeuristicEvaluator,
};
use duke_training::nnue::NnueEvaluator;

enum Player<'a> {
    Random,
    Evaluator(&'a dyn GameEvaluator),
}

fn play_match(
    gs: &GameState,
    top_player: &Player,
    bottom_player: &Player,
    rng: &mut StdRng,
    max_turns: u32,
) -> GameResult {
    let ai = StupidSyncAi {};
    let mut game = gs.clone();
    let mut turns = 0u32;

    loop {
        match game.game_result() {
            GameResult::Ongoing => {
                if turns >= max_turns {
                    return GameResult::Tie;
                }
                let current = game.current_player_turn();
                let player = match current {
                    Owner::TopPlayer => top_player,
                    Owner::BottomPlayer => bottom_player,
                };
                match player {
                    Player::Random => {
                        ai.play_next_move(rng, &mut game);
                    }
                    Player::Evaluator(eval) => {
                        let mv = greedy_move(&game, *eval, rng);
                        mv.play(&mut game, rng);
                    }
                }
                turns += 1;
            }
            result => return result,
        }
    }
}

struct MatchResult {
    player_a_wins: u32,
    player_b_wins: u32,
    ties: u32,
}

fn run_matches(
    gs: &GameState,
    player_a: &Player,
    player_b: &Player,
    num_games: u32,
    label: &str,
) -> MatchResult {
    let start = Instant::now();
    let mut result = MatchResult {
        player_a_wins: 0,
        player_b_wins: 0,
        ties: 0,
    };

    for seed in 0..num_games {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let game_result = if seed % 2 == 0 {
            let r = play_match(gs, player_a, player_b, &mut rng, 200);
            match r {
                GameResult::Won(Owner::TopPlayer) => GameResult::Won(Owner::TopPlayer),
                GameResult::Won(Owner::BottomPlayer) => GameResult::Won(Owner::BottomPlayer),
                other => other,
            }
        } else {
            let r = play_match(gs, player_b, player_a, &mut rng, 200);
            match r {
                GameResult::Won(Owner::TopPlayer) => GameResult::Won(Owner::BottomPlayer),
                GameResult::Won(Owner::BottomPlayer) => GameResult::Won(Owner::TopPlayer),
                other => other,
            }
        };
        match game_result {
            GameResult::Won(Owner::TopPlayer) => result.player_a_wins += 1,
            GameResult::Won(Owner::BottomPlayer) => result.player_b_wins += 1,
            _ => result.ties += 1,
        }
    }

    let elapsed = start.elapsed();
    let total = num_games as f64;
    println!(
        "{}: A={:.1}% B={:.1}% Tie={:.1}% ({} games in {:.1?})",
        label,
        result.player_a_wins as f64 / total * 100.0,
        result.player_b_wins as f64 / total * 100.0,
        result.ties as f64 / total * 100.0,
        num_games,
        elapsed,
    );

    result
}

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
