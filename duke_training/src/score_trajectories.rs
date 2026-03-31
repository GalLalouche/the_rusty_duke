//! Score every position in a DTRJ trajectory file using LR-Cheap depth-N negamax.
//!
//! Outputs a flat binary sidecar file: one little-endian f32 per state, in the
//! same order as the trajectories. Terminal states get ±30.0 or 0.0.
//!
//! Usage:
//!   score_trajectories --trajectories <path.dtrj> \
//!                      --evaluator <path.json> \
//!                      --depth <N> \
//!                      --output <path.bin>

use std::io::{Write, BufWriter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rayon::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use duke_rust::game::state::GameResult;

use duke_training::cli::parse_flag;
use duke_training::game_setup::{negamax, TERMINAL_WIN_SCORE, TERMINAL_LOSS_SCORE};
use duke_training::learned_heuristic::CombinedWeights;
use duke_training::trajectory_io::load_trajectories;

fn main() {
    // Build a rayon thread pool with 64 MB stacks to handle deep negamax recursion.
    rayon::ThreadPoolBuilder::new()
        .stack_size(64 * 1024 * 1024)
        .build_global()
        .expect("Failed to build rayon thread pool");

    let args: Vec<String> = std::env::args().collect();

    let trajectories_path: String = parse_flag(&args, "--trajectories")
        .unwrap_or_else(|| {
            eprintln!("Usage: score_trajectories --trajectories <path> --evaluator <path> --depth <N> --output <path>");
            std::process::exit(1);
        });
    let evaluator_path: String = parse_flag(&args, "--evaluator")
        .unwrap_or_else(|| {
            eprintln!("Missing --evaluator flag");
            std::process::exit(1);
        });
    let depth: u32 = parse_flag(&args, "--depth").unwrap_or(3);
    let output_path: String = parse_flag(&args, "--output")
        .unwrap_or_else(|| {
            eprintln!("Missing --output flag");
            std::process::exit(1);
        });

    // --- Load trajectories ---
    eprintln!("Loading trajectories from {} ...", trajectories_path);
    let t0 = Instant::now();
    let games = load_trajectories(&trajectories_path)
        .expect("Failed to load trajectories");
    let total_states: usize = games.iter().map(|g| g.states.len()).sum();
    eprintln!("Loaded {} games, {} states in {:.1}s",
        games.len(), total_states, t0.elapsed().as_secs_f64());

    // --- Load evaluator ---
    eprintln!("Loading LR-Cheap evaluator from {} ...", evaluator_path);
    let evaluator = CombinedWeights::load(&evaluator_path)
        .expect("Failed to load CombinedWeights");

    // --- Score positions ---
    eprintln!("Scoring {} positions with depth {} negamax ...", total_states, depth);
    let t1 = Instant::now();
    let done = AtomicU64::new(0);
    let total = total_states as u64;

    // Parallel over games, sequential within each game
    let all_scores: Vec<Vec<f32>> = games.par_iter()
        .map(|game| {
            let mut rng = StdRng::seed_from_u64(0);
            let scores: Vec<f32> = game.states.iter().map(|gs| {
                // Clone because game_result() and negamax() take &mut GameState.
                let mut gs_clone = gs.clone();
                let result = gs_clone.game_result();
                let score = match result {
                    GameResult::Won(winner) => {
                        if winner == gs.current_player_turn() {
                            TERMINAL_WIN_SCORE as f32
                        } else {
                            TERMINAL_LOSS_SCORE as f32
                        }
                    }
                    GameResult::Tie => 0.0f32,
                    GameResult::Ongoing => {
                        negamax(&mut gs_clone, &evaluator, depth, &mut rng) as f32
                    }
                };
                let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
                if completed % 10000 == 0 || completed == total {
                    let elapsed = t1.elapsed().as_secs_f64();
                    let rate = completed as f64 / elapsed;
                    eprintln!("  Scored {}/{} ({:.1}%) [{:.0} pos/sec]",
                        completed, total,
                        completed as f64 / total as f64 * 100.0,
                        rate);
                }
                score
            }).collect();
            scores
        })
        .collect();

    let elapsed = t1.elapsed().as_secs_f64();
    let rate = total as f64 / elapsed;
    eprintln!("Scoring complete in {:.1}s ({:.0} pos/sec)", elapsed, rate);

    // --- Save scores ---
    eprintln!("Saving {} scores to {} ...", total_states, output_path);
    let f = std::fs::File::create(&output_path)
        .expect("Failed to create output file");
    let mut w = BufWriter::new(f);
    for game_scores in &all_scores {
        for &score in game_scores {
            w.write_all(&score.to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();

    let file_size = std::fs::metadata(&output_path).unwrap().len();
    let expected_size = total_states * 4;
    assert_eq!(file_size as usize, expected_size,
        "Output file size mismatch: {} bytes vs expected {}", file_size, expected_size);
    eprintln!("Saved {} scores ({:.1} MB) to {}",
        total_states,
        file_size as f64 / (1024.0 * 1024.0),
        output_path);

    // Summary stats
    let all_flat: Vec<f32> = all_scores.into_iter().flatten().collect();
    let min = all_flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = all_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = all_flat.iter().map(|v| *v as f64).sum::<f64>() / all_flat.len() as f64;
    eprintln!("Score stats: min={:.4}, max={:.4}, mean={:.4}", min, max, mean);
}
