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
use duke_training::model_registry::{BenchmarkRecord, ModelRegistry};

/// Derive a short display label from the player spec and the description
/// returned by `load_opponent`, avoiding redundant file loads.
fn derive_label(spec: &str, desc: &str) -> String {
    match spec {
        "base" => "Base".to_string(),
        "random" => "Random".to_string(),
        path => {
            let stem = Path::new(path)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| path.to_string());
            // desc from load_opponent is e.g. "GMLP (41->64->1)" or "NNUE (1106->256->32->1)"
            // Extract the parenthesized arch portion if present.
            if let Some(start) = desc.find('(') {
                format!("{}{}", stem, &desc[start..])
            } else {
                stem
            }
        }
    }
}

/// Parse CLI arguments into player specs, the number of games per matchup, and optional DB path.
fn parse_args() -> (Vec<String>, u32, Option<String>) {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut games_per_matchup: u32 = 1000;
    let mut player_specs: Vec<String> = Vec::new();
    let mut db_path: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        if args[i] == "--games" {
            i += 1;
            if i >= args.len() {
                eprintln!("Error: --games requires a value");
                std::process::exit(1);
            }
            games_per_matchup = args[i].parse().expect("--games must be a positive integer");
        } else if args[i] == "--db" {
            i += 1;
            if i >= args.len() {
                eprintln!("Error: --db requires a value");
                std::process::exit(1);
            }
            db_path = Some(args[i].clone());
        } else if args[i].starts_with("--") {
            eprintln!("Unknown flag: {}", args[i]);
            std::process::exit(1);
        } else {
            player_specs.push(args[i].clone());
        }
        i += 1;
    }

    if player_specs.len() < 2 {
        eprintln!("Usage: elo_tournament [--games N] [--db <path>] <player1> <player2> [player3 ...]");
        eprintln!("  Player specs: base, random, or a path to .gmlp / .nnue file");
        std::process::exit(1);
    }

    (player_specs, games_per_matchup, db_path)
}

/// Load all players from their specs.
/// Returns a list of (label, optional evaluator) pairs.
fn load_players(
    specs: &[String],
) -> Vec<(String, Option<Box<dyn GameEvaluator + Sync + Send>>)> {
    let mut players = Vec::with_capacity(specs.len());
    for (idx, spec) in specs.iter().enumerate() {
        let (eval_box, desc) = load_opponent(spec);
        let label = derive_label(spec, &desc);
        println!("  [{}] {} -- {}", idx, label, desc);
        players.push((label, eval_box));
    }
    players
}

/// Run round-robin matches between all player pairs.
/// Returns (wins, ties) matrices where wins[i][j] = games player i won vs j.
fn run_round_robin(
    players: &[(String, Option<Box<dyn GameEvaluator + Sync + Send>>)],
    games_per_matchup: u32,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let n = players.len();
    let bag = create_bag();
    let gs = create_initial_state(&bag);

    let mut wins = vec![vec![0u32; n]; n];
    let mut ties = vec![vec![0u32; n]; n];

    let num_matchups = n * (n - 1) / 2;
    let mut matchup_idx = 0;

    for a in 0..n {
        for b in (a + 1)..n {
            matchup_idx += 1;
            let label = format!(
                "[{}/{}] {} vs {}",
                matchup_idx, num_matchups, players[a].0, players[b].0
            );
            print!("  ");

            let player_a = match &players[a].1 {
                Some(eval) => Player::Evaluator(eval.as_ref()),
                None => Player::Random,
            };
            let player_b = match &players[b].1 {
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

    (wins, ties)
}

/// Compute Elo ratings from win/loss/tie matrices using iterative updates.
fn compute_elo(wins: &[Vec<u32>], ties: &[Vec<u32>], n: usize) -> Vec<f64> {
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

                let e_a = 1.0 / (1.0 + 10.0f64.powf((elo[b] - elo[a]) / 400.0));
                let e_b = 1.0 - e_a;

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

    elo
}

/// Print the win-rate matrix, Elo rankings, and per-matchup details.
fn print_results(
    labels: &[String],
    wins: &[Vec<u32>],
    ties: &[Vec<u32>],
    elo: &[f64],
) {
    let n = labels.len();
    let max_label_len = labels.iter().map(|l| l.len()).max().unwrap_or(4).max(6);

    // ── Win-rate matrix ──────────────────────────────────────────────────
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

    // ── Elo rankings ─────────────────────────────────────────────────────
    let mut ranked: Vec<(usize, f64)> = (0..n).map(|i| (i, elo[i])).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("=== Elo Ratings (K=32, 10 passes) ===");
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

fn main() {
    let (player_specs, games_per_matchup, db_path) = parse_args();
    let n = player_specs.len();

    println!("=== Elo Tournament ===");
    println!("  {} players, {} games per matchup", n, games_per_matchup);
    if let Some(ref db) = db_path {
        println!("  registry DB: {}", db);
    }
    println!();

    let players = load_players(&player_specs);
    println!();

    let total_start = Instant::now();
    let (wins, ties) = run_round_robin(&players, games_per_matchup);
    let total_elapsed = total_start.elapsed();

    println!();
    println!("All matches complete in {:.1?}", total_elapsed);
    println!();

    let labels: Vec<String> = players.iter().map(|(l, _)| l.clone()).collect();
    let elo = compute_elo(&wins, &ties, n);
    print_results(&labels, &wins, &ties, &elo);

    // ── Record benchmarks in registry if --db was provided ───────────────
    if let Some(ref db) = db_path {
        println!();
        println!("Recording benchmarks in registry: {}", db);

        let registry = ModelRegistry::open(db).expect("Failed to open model registry DB");

        // For each player that is a file (not "base" or "random"), look up by path
        let mut model_ids: Vec<Option<i64>> = Vec::with_capacity(n);
        for spec in &player_specs {
            if spec == "base" || spec == "random" {
                model_ids.push(None);
            } else {
                match registry.find_by_path(spec) {
                    Ok(Some(record)) => {
                        println!("  Found model ID {} for {}", record.id, spec);
                        model_ids.push(Some(record.id));
                    }
                    Ok(None) => {
                        println!("  Model not registered: {} (skipping)", spec);
                        model_ids.push(None);
                    }
                    Err(e) => {
                        eprintln!("  Error looking up {}: {}", spec, e);
                        model_ids.push(None);
                    }
                }
            }
        }

        // Record pairwise benchmark results for each registered model
        for a in 0..n {
            if model_ids[a].is_none() {
                continue;
            }
            let model_id = model_ids[a].unwrap();

            for b in 0..n {
                if a == b {
                    continue;
                }

                let total_games = wins[a][b] + wins[b][a] + ties[a][b];
                if total_games == 0 {
                    continue;
                }

                let bench = BenchmarkRecord {
                    id: None,
                    opponent: player_specs[b].clone(),
                    opponent_model_id: model_ids[b],
                    num_games: total_games,
                    wins: wins[a][b],
                    losses: wins[b][a],
                    ties: ties[a][b],
                    win_rate: None,
                    elo: Some(elo[a]),
                    benchmark_date: None,
                };

                match registry.record_benchmark(model_id, &bench) {
                    Ok(bench_id) => {
                        let wr = (wins[a][b] as f64 + 0.5 * ties[a][b] as f64)
                            / total_games as f64;
                        println!(
                            "  Recorded benchmark ID {}: {} vs {} ({:.1}% wr, elo {:.0})",
                            bench_id, labels[a], labels[b], wr * 100.0, elo[a]
                        );
                    }
                    Err(e) => {
                        eprintln!("  Error recording benchmark for {} vs {}: {}", labels[a], labels[b], e);
                    }
                }
            }
        }
    }
}
