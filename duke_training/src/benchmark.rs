use duke_training::game_setup::{create_bag, create_initial_state, StaticHeuristicEvaluator};
use duke_training::learned_heuristic::LearnedHeuristicWeights;
use duke_training::match_runner::{run_matches, Player};
use duke_training::nnue::{NnueEvaluator, NnueWeights};

fn num_games() -> u32 {
    match std::env::var("NUM_GAMES") {
        Ok(val) => match val.parse::<u32>() {
            Ok(n) => n,
            Err(e) => {
                eprintln!(
                    "Warning: NUM_GAMES='{}' is not a valid u32 ({}), falling back to 200",
                    val, e
                );
                200
            }
        },
        Err(_) => 200,
    }
}

/// Parse all values for a repeated flag: --flag v1 --flag v2
fn parse_all_flags(args: &[String], flag: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag {
            if let Some(val) = args.get(i + 1) {
                values.push(val.clone());
                i += 2;
                continue;
            } else {
                panic!(
                    "Flag '{}' at position {} has no value (appears as last argument)",
                    flag, i
                );
            }
        }
        i += 1;
    }
    values
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_games = num_games();

    let nnue_paths = parse_all_flags(&args, "--nnue");
    let learned_paths = parse_all_flags(&args, "--learned");

    if nnue_paths.is_empty() && learned_paths.is_empty() {
        eprintln!("Usage: benchmark [--nnue <path>]... [--learned <path>]...");
        eprintln!("  Runs a full round-robin of all players.");
        eprintln!("  Always includes: Random, Heuristic");
        eprintln!("  Example:");
        eprintln!("    benchmark --nnue random.nnue --nnue heuristic.nnue --learned random_lr.json --learned heur_lr.json");
        std::process::exit(1);
    }

    let bag = create_bag();
    let gs = create_initial_state(&bag);

    // Build player roster: always include Random and Heuristic
    let heuristic_eval = StaticHeuristicEvaluator::new();

    // Load NNUE models
    let nnue_evaluators: Vec<(String, NnueEvaluator)> = nnue_paths
        .iter()
        .map(|path| {
            let label = short_label(path, "NNUE");
            println!("Loading {}: {}", label, path);
            let weights = NnueWeights::load(path).expect("Failed to load NNUE weights");
            let eval = NnueEvaluator::new(weights);
            (label, eval)
        })
        .collect();

    // Load learned heuristic models
    let learned_evaluators: Vec<(String, LearnedHeuristicWeights)> = learned_paths
        .iter()
        .map(|path| {
            let label = short_label(path, "LR");
            println!("Loading {}: {}", label, path);
            let w = LearnedHeuristicWeights::load(path).expect("Failed to load learned weights");
            (label, w)
        })
        .collect();

    // Build named player list
    let mut players: Vec<(&str, Player)> = Vec::new();

    // Fixed players
    players.push(("Random", Player::Random));
    players.push(("Heuristic", Player::Evaluator(&heuristic_eval)));

    // Dynamically loaded players (need stable references)
    let nnue_names: Vec<String> = nnue_evaluators.iter().map(|(l, _)| l.clone()).collect();
    let learned_names: Vec<String> = learned_evaluators.iter().map(|(l, _)| l.clone()).collect();

    for (i, (_, eval)) in nnue_evaluators.iter().enumerate() {
        players.push((&nnue_names[i], Player::Evaluator(eval)));
    }
    for (i, (_, eval)) in learned_evaluators.iter().enumerate() {
        players.push((&learned_names[i], Player::Evaluator(eval)));
    }

    let n = players.len();
    println!("\n=== Full Round-Robin: {} players, {} games per matchup ===\n", n, num_games);

    // Print player list
    for (i, (name, _)) in players.iter().enumerate() {
        println!("  [{}] {}", i, name);
    }
    println!();

    // Results matrix: wins[i][j] = how many times player i beat player j
    let mut wins = vec![vec![0u32; n]; n];
    let mut ties = vec![vec![0u32; n]; n];

    // Round-robin: every pair plays
    for i in 0..n {
        for j in (i + 1)..n {
            let label = format!("{} vs {}", players[i].0, players[j].0);
            let result = run_matches(&gs, &players[i].1, &players[j].1, num_games, &label);
            wins[i][j] = result.player_a_wins;
            wins[j][i] = result.player_b_wins;
            ties[i][j] = result.ties;
            ties[j][i] = result.ties;
        }
    }

    // Print summary matrix
    println!("\n=== Win Rate Matrix (row vs column) ===\n");

    // Header
    let names: Vec<&str> = players.iter().map(|(n, _)| *n).collect();
    let max_name = names.iter().map(|n| n.len()).max().unwrap_or(8);
    print!("{:>width$} ", "", width = max_name);
    for name in &names {
        print!(" {:>8}", &name[..name.len().min(8)]);
    }
    println!();

    for i in 0..n {
        print!("{:>width$} ", names[i], width = max_name);
        for j in 0..n {
            if i == j {
                print!("      -- ");
            } else {
                let total = wins[i][j] + wins[j][i] + ties[i][j];
                if total > 0 {
                    let pct = wins[i][j] as f64 / total as f64 * 100.0;
                    print!("  {:5.1}% ", pct);
                } else {
                    print!("      -- ");
                }
            }
        }
        println!();
    }

    // Ranking by total win rate
    println!("\n=== Ranking (total win rate across all opponents) ===\n");
    let mut rankings: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let total_wins: u32 = (0..n).filter(|&j| j != i).map(|j| wins[i][j]).sum();
            let total_games: u32 = (0..n)
                .filter(|&j| j != i)
                .map(|j| wins[i][j] + wins[j][i] + ties[i][j])
                .sum();
            let pct = if total_games > 0 {
                total_wins as f64 / total_games as f64 * 100.0
            } else {
                0.0
            };
            (i, pct)
        })
        .collect();
    rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (rank, (idx, pct)) in rankings.iter().enumerate() {
        println!("  {}. {} — {:.1}%", rank + 1, names[*idx], pct);
    }
}

/// Extract a short label from a file path for display.
fn short_label(path: &str, prefix: &str) -> String {
    // Use parent directory name if it's informative, otherwise the filename
    let path_obj = std::path::Path::new(path);
    let parent = path_obj.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str());
    match parent {
        Some(dir) if dir != "." && dir != ".." => format!("{}({})", prefix, dir),
        _ => {
            let stem = path_obj.file_stem().and_then(|s| s.to_str()).unwrap_or(path);
            format!("{}({})", prefix, stem)
        }
    }
}
