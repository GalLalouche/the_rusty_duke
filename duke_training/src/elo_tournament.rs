//! Elo tournament: round-robin play between multiple players with Elo rating computation.
//!
//! Usage: elo_tournament --models 1,2,3,base,random [--games N] [--db <path>] [--quantize]
//!
//! Flags:
//!   --models <list>  Comma-separated player list (REQUIRED). Each entry is either:
//!                       - a number (DB model ID)
//!                       - "base"   (built-in heuristic evaluator)
//!                       - "random" (random move selection)
//!   --games N        Number of games per matchup (default 1000)
//!   --db <path>      Path to model registry SQLite DB (default D:/temp/duke_models.db)
//!   --quantize       Quantize .gmlp models (int8 hidden weights, f32 L1 and output)

use std::time::Instant;

use duke_training::game_setup::{create_bag, create_initial_state};
use duke_training::generic_mlp::LoadedModel;
use duke_training::match_runner::run_matches;
use duke_training::model_registry::{BenchmarkRecord, ModelRegistry};

const DEFAULT_DB_PATH: &str = "D:/temp/duke_models.db";

fn print_usage_and_exit() -> ! {
    eprintln!("Usage: elo_tournament --models 1,2,3,base,random [--games N] [--db <path>] [--quantize]");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --models <list>  Comma-separated player list (required)");
    eprintln!("                   Each entry: a number (DB model ID), \"base\", or \"random\"");
    eprintln!("  --games N        Number of games per matchup (default 1000)");
    eprintln!("  --db <path>      Path to model registry SQLite DB (default {})", DEFAULT_DB_PATH);
    eprintln!("  --quantize       Quantize .gmlp models (int8 hidden weights)");
    std::process::exit(1);
}

/// Parse CLI arguments into the models list, number of games per matchup, DB path, and quantize flag.
fn parse_args() -> (Vec<String>, u32, String, bool) {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut games_per_matchup: u32 = 1000;
    let mut models_raw: Option<String> = None;
    let mut db_path: Option<String> = None;
    let mut quantize = false;

    let mut i = 0;
    while i < args.len() {
        if args[i] == "--models" {
            i += 1;
            if i >= args.len() {
                eprintln!("Error: --models requires a value");
                std::process::exit(1);
            }
            models_raw = Some(args[i].clone());
        } else if args[i] == "--games" {
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
        } else if args[i] == "--quantize" {
            quantize = true;
        } else if args[i].starts_with("--") {
            eprintln!("Unknown flag: {}", args[i]);
            std::process::exit(1);
        } else {
            eprintln!("Unexpected positional argument: {}", args[i]);
            print_usage_and_exit();
        }
        i += 1;
    }

    let models_str = match models_raw {
        Some(s) => s,
        None => {
            eprintln!("Error: --models is required");
            print_usage_and_exit();
        }
    };

    let models: Vec<String> = models_str
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if models.len() < 2 {
        eprintln!("Error: --models must specify at least 2 players");
        print_usage_and_exit();
    }

    let db_path = db_path.unwrap_or_else(|| DEFAULT_DB_PATH.to_string());

    (models, games_per_matchup, db_path, quantize)
}

/// Load all players from the models list.
/// Numeric entries are loaded as DB model IDs from the registry.
/// "base" and "random" are loaded via `LoadedModel::from_spec`.
/// When `quantize` is true, .gmlp models are loaded with int8-quantized hidden layers.
fn load_players(models: &[String], registry: &ModelRegistry, quantize: bool) -> Vec<LoadedModel> {
    let mut players = Vec::with_capacity(models.len());
    for (idx, entry) in models.iter().enumerate() {
        let model = if entry == "base" || entry == "random" {
            LoadedModel::from_spec(entry)
        } else if let Ok(model_id) = entry.parse::<i64>() {
            if quantize {
                LoadedModel::from_db_id_quantized(registry, model_id).unwrap_or_else(|e| {
                    eprintln!("Error loading model ID {}: {}", model_id, e);
                    std::process::exit(1);
                })
            } else {
                LoadedModel::from_db_id(registry, model_id).unwrap_or_else(|e| {
                    eprintln!("Error loading model ID {}: {}", model_id, e);
                    std::process::exit(1);
                })
            }
        } else {
            eprintln!(
                "Error: unrecognized model entry '{}'. Expected a number (DB model ID), \"base\", or \"random\".",
                entry
            );
            std::process::exit(1);
        };
        println!("  [{}] {}", idx, model.label);
        players.push(model);
    }
    players
}

/// Run round-robin matches between all player pairs.
/// Returns (wins, ties) matrices where wins[i][j] = games player i won vs j.
fn run_round_robin(
    players: &[LoadedModel],
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
                matchup_idx, num_matchups, players[a].label, players[b].label
            );
            print!("  ");

            let player_a = players[a].as_player();
            let player_b = players[b].as_player();

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
    let (models, games_per_matchup, db_path, quantize) = parse_args();
    let n = models.len();

    println!("=== Elo Tournament ===");
    println!("  {} players, {} games per matchup", n, games_per_matchup);
    println!("  registry DB: {}", db_path);
    if quantize {
        println!("  quantize: ON (int8 hidden weights for .gmlp models)");
    }
    println!();

    // Always open the registry (we have a default DB path)
    let registry =
        ModelRegistry::open(&db_path).expect("Failed to open model registry DB");

    let players = load_players(&models, &registry, quantize);
    println!();

    let total_start = Instant::now();
    let (wins, ties) = run_round_robin(&players, games_per_matchup);
    let total_elapsed = total_start.elapsed();

    println!();
    println!("All matches complete in {:.1?}", total_elapsed);
    println!();

    let labels: Vec<String> = players.iter().map(|m| m.label.clone()).collect();
    let elo = compute_elo(&wins, &ties, n);
    print_results(&labels, &wins, &ties, &elo);

    // ── Record benchmarks in registry ──────────────────────────────────
    // Model IDs come directly from LoadedModel.id (set for DB-loaded models, None for base/random)
    let model_ids: Vec<Option<i64>> = players.iter().map(|m| m.id).collect();
    let has_any_db_model = model_ids.iter().any(|id| id.is_some());

    if has_any_db_model {
        println!();
        println!("Recording benchmarks in registry: {}", db_path);

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
                    opponent: labels[b].clone(),
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
                        eprintln!(
                            "  Error recording benchmark for {} vs {}: {}",
                            labels[a], labels[b], e
                        );
                    }
                }
            }
        }
    }
}
