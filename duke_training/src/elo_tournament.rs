//! Elo tournament: round-robin play between multiple players with Elo rating computation.
//!
//! Usage: elo_tournament [--games N] <player1> <player2> [player3 ...]
//!
//! Player specs:
//!   - "base"    -> StaticHeuristicEvaluator
//!   - "random"  -> Random move selection
//!   - path.gmlp -> GenericMlp model (dispatched by input_size)
//!   - path.nnue -> NNUE model
//!
//! Flags:
//!   --games N   Number of games per matchup (default 1000)

use std::path::Path;
use std::time::Instant;

use duke_training::game_setup::{create_bag, create_initial_state, GameEvaluator};
use duke_training::generic_mlp::load_opponent;
use duke_training::match_runner::{run_matches, Player};

/// Short display label for a player spec.
fn short_label(spec: &str) -> String {
    match spec {
        "base" => "Base".to_string(),
        "random" => "Random".to_string(),
        path => {
            let stem = Path::new(path)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| path.to_string());
            // For .gmlp files, try to load and get arch string
            if path.ends_with(".gmlp") {
                // Load to get the arch description
                if let Ok(net) = duke_training::generic_mlp::GenericMlp::load(path) {
                    format!("{}({})", stem, net.arch_string())
                } else {
                    stem
                }
            } else if path.ends_with(".nnue") {
                format!("{}(nnue)", stem)
            } else {
                stem
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // Parse --games flag
    let mut games_per_matchup: u32 = 1000;
    let mut player_specs: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        if args[i] == "--games" {
            i += 1;
            if i >= args.len() {
                eprintln!("Error: --games requires a value");
                std::process::exit(1);
            }
            games_per_matchup = args[i].parse().expect("--games must be a positive integer");
        } else if args[i].starts_with("--") {
            eprintln!("Unknown flag: {}", args[i]);
            std::process::exit(1);
        } else {
            player_specs.push(args[i].clone());
        }
        i += 1;
    }

    if player_specs.len() < 2 {
        eprintln!("Usage: elo_tournament [--games N] <player1> <player2> [player3 ...]");
        eprintln!("  Player specs: base, random, or a path to .gmlp / .nnue file");
        std::process::exit(1);
    }

    let n = player_specs.len();

    // Load all players
    let labels: Vec<String> = player_specs.iter().map(|s| short_label(s)).collect();
    let mut evaluators: Vec<Option<Box<dyn GameEvaluator + Sync + Send>>> = Vec::with_capacity(n);

    println!("=== Elo Tournament ===");
    println!("  {} players, {} games per matchup", n, games_per_matchup);
    println!();

    for (idx, spec) in player_specs.iter().enumerate() {
        let (eval_box, desc) = load_opponent(spec);
        println!("  [{}] {} -- {}", idx, labels[idx], desc);
        evaluators.push(eval_box);
    }
    println!();

    // Setup game state
    let bag = create_bag();
    let gs = create_initial_state(&bag);

    // Round-robin: play every pair
    // wins[i][j] = number of games player i won against player j
    // ties[i][j] = number of ties between player i and player j
    let mut wins = vec![vec![0u32; n]; n];
    let mut ties = vec![vec![0u32; n]; n];

    let total_start = Instant::now();
    let num_matchups = n * (n - 1) / 2;
    let mut matchup_idx = 0;

    for a in 0..n {
        for b in (a + 1)..n {
            matchup_idx += 1;
            let label = format!(
                "[{}/{}] {} vs {}",
                matchup_idx, num_matchups, labels[a], labels[b]
            );
            print!("  ");

            let player_a = match &evaluators[a] {
                Some(eval) => Player::Evaluator(eval.as_ref()),
                None => Player::Random,
            };
            let player_b = match &evaluators[b] {
                Some(eval) => Player::Evaluator(eval.as_ref()),
                None => Player::Random,
            };

            let result = run_matches(&gs, &player_a, &player_b, games_per_matchup, &label);

            wins[a][b] = result.player_a_wins;
            wins[b][a] = result.player_b_wins;
            ties[a][b] = result.ties;
            ties[b][a] = result.ties;
        }
    }

    let total_elapsed = total_start.elapsed();
    println!();
    println!("All matches complete in {:.1?}", total_elapsed);
    println!();

    // ── Win-rate matrix ──────────────────────────────────────────────────

    // Find max label width for formatting
    let max_label_len = labels.iter().map(|l| l.len()).max().unwrap_or(4).max(6);

    println!("=== Win-Rate Matrix (row vs col) ===");
    print!("{:>width$}", "", width = max_label_len + 2);
    for j in 0..n {
        print!("  {:>8}", &labels[j][..labels[j].len().min(8)]);
    }
    println!();

    for i in 0..n {
        print!("{:>width$}", labels[i], width = max_label_len + 2);
        for j in 0..n {
            if i == j {
                print!("       ---");
            } else {
                let total = wins[i][j] + wins[j][i] + ties[i][j];
                let wr = if total > 0 {
                    (wins[i][j] as f64 + 0.5 * (ties[i][j] as f64)) / total as f64 * 100.0
                } else {
                    0.0
                };
                print!("    {:5.1}%", wr);
            }
        }
        println!();
    }
    println!();

    // ── Elo computation ──────────────────────────────────────────────────
    // Iterative Elo: start at 1500, K=32, iterate 10 passes to converge.

    let mut elo = vec![1500.0f64; n];
    let k = 32.0f64;
    let passes = 10;

    for _ in 0..passes {
        let mut delta = vec![0.0f64; n];
        for a in 0..n {
            for b in (a + 1)..n {
                let total = wins[a][b] + wins[b][a] + ties[a][b];
                if total == 0 {
                    continue;
                }
                let total_f = total as f64;

                // Expected score for a against b
                let e_a = 1.0 / (1.0 + 10.0f64.powf((elo[b] - elo[a]) / 400.0));
                let e_b = 1.0 - e_a;

                // Actual score
                let s_a = (wins[a][b] as f64 + 0.5 * (ties[a][b] as f64)) / total_f;
                let s_b = (wins[b][a] as f64 + 0.5 * (ties[b][a] as f64)) / total_f;

                delta[a] += k * (s_a - e_a);
                delta[b] += k * (s_b - e_b);
            }
        }
        for i in 0..n {
            elo[i] += delta[i];
        }
    }

    // Sort by Elo descending
    let mut ranked: Vec<(usize, f64)> = (0..n).map(|i| (i, elo[i])).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("=== Elo Ratings (K={}, {} passes) ===", k as u32, passes);
    println!("{:>4}  {:>width$}  {:>6}", "Rank", "Player", "Elo", width = max_label_len);
    for (rank, &(idx, rating)) in ranked.iter().enumerate() {
        println!(
            "{:>4}  {:>width$}  {:>6.0}",
            rank + 1,
            labels[idx],
            rating,
            width = max_label_len,
        );
    }
    println!();

    // ── Matchup details ──────────────────────────────────────────────────

    println!("=== Matchup Details ===");
    for a in 0..n {
        for b in (a + 1)..n {
            let total = wins[a][b] + wins[b][a] + ties[a][b];
            let wr_a = if total > 0 {
                (wins[a][b] as f64 + 0.5 * (ties[a][b] as f64)) / total as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "  {} vs {}: {}-{}-{} (W-L-T)  {:.1}% win rate for {}",
                labels[a], labels[b],
                wins[a][b], wins[b][a], ties[a][b],
                wr_a, labels[a],
            );
        }
    }
}
