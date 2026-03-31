//! Self-play game generation binary using LR-Cheap evaluator with depth-N negamax.
//!
//! Plays self-play games where both sides use the same CombinedWeights evaluator
//! with negamax at the specified depth. Saves full game trajectories to a DTRJ file.
//!
//! Usage:
//!   generate_games --evaluator D:/temp/combined_lr_41.json --depth 3 --games 200000 \
//!                  --output D:/temp/lr_cheap_d3_200k.dtrj --epsilon 0.05 --seed 42

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;

use duke_rust::game::ai::player::ArtificialPlayer;
use duke_rust::game::ai::stupid_sync_ai::StupidSyncAi;
use duke_rust::game::state::{GameResult, GameState};

use duke_training::cli::parse_flag;
use duke_training::game_setup::{
    create_bag, create_initial_state, greedy_move_deep_with_score, GameEvaluator, MAX_TURNS,
};
use duke_training::loaded_model::LoadedModel;
use duke_training::trajectory_io::TrajectoryWriter;

/// Play a single self-play game using depth-N negamax with epsilon-greedy exploration.
///
/// Both sides use the same evaluator. With probability `epsilon`, a random legal
/// move is played instead of the best negamax move.
///
/// Returns the sequence of all game states (including the terminal state) and the result.
/// Result of playing a single game: states, eval scores per state, and game result.
struct GameData {
    states: Vec<GameState>,
    /// Depth-N eval score for each non-terminal state (from current player's perspective).
    /// Terminal states get NaN (use game outcome instead).
    scores: Vec<f32>,
    result: GameResult,
}

fn play_depth_game(
    gs: &GameState,
    evaluator: &(dyn GameEvaluator + Sync),
    depth: u32,
    rng: &mut StdRng,
    epsilon: f64,
) -> GameData {
    let ai = StupidSyncAi {};
    let mut game = gs.clone();
    let mut states = Vec::new();
    let mut scores = Vec::new();

    loop {
        match game.game_result() {
            GameResult::Ongoing => {
                if states.len() as u32 >= MAX_TURNS {
                    states.push(game.clone());
                    scores.push(f32::NAN);
                    return GameData { states, scores, result: GameResult::Tie };
                }
                states.push(game.clone());

                if rng.gen::<f64>() < epsilon {
                    // Random move — no meaningful eval score
                    scores.push(f32::NAN);
                    ai.play_next_move(rng, &mut game);
                } else {
                    // Best negamax move at given depth — save the score
                    let (mv, score) = greedy_move_deep_with_score(&mut game, evaluator, depth, rng);
                    scores.push(score as f32);
                    let mut eval_rng = rand::rngs::SmallRng::seed_from_u64(0);
                    mv.play(&mut game, &mut eval_rng);
                }
            }
            result => {
                states.push(game.clone());
                scores.push(f32::NAN); // terminal
                return GameData { states, scores, result };
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let evaluator_path: String = parse_flag(&args, "--evaluator").unwrap_or_else(|| {
        eprintln!(
            "Usage: generate_games --evaluator <path.json> --depth <N> --games <N> \
             --output <path.dtrj> [--epsilon 0.05] [--seed 42]"
        );
        std::process::exit(1);
    });
    let depth: u32 = parse_flag(&args, "--depth").unwrap_or(3);
    let num_games: u64 = parse_flag(&args, "--games").unwrap_or(200_000);
    let output_path: String = parse_flag(&args, "--output").unwrap_or_else(|| {
        eprintln!("Missing --output flag");
        std::process::exit(1);
    });
    let epsilon: f64 = parse_flag(&args, "--epsilon").unwrap_or(0.05);
    let seed: u64 = parse_flag(&args, "--seed").unwrap_or(42);

    eprintln!("=== Self-Play Game Generation ===");
    eprintln!("  Evaluator:  {}", evaluator_path);
    eprintln!("  Depth:      {}", depth);
    eprintln!("  Games:      {}", num_games);
    eprintln!("  Output:     {}", output_path);
    eprintln!("  Epsilon:    {}", epsilon);
    eprintln!("  Seed:       {}", seed);
    eprintln!();

    // Load evaluator (supports .json with 24/41/65 weights, .gmlp, .gcnn, etc.)
    eprintln!("Loading evaluator ...");
    let loaded = LoadedModel::from_spec(&evaluator_path, false);
    eprintln!("  Loaded: {}", loaded.label);
    let evaluator = loaded.evaluator.as_ref().expect("Evaluator is None (random?)");

    // Create output directory if needed
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).expect("Failed to create output directory");
    }

    // Set up shared state
    let completed = AtomicU64::new(0);
    let total_states = AtomicU64::new(0);
    let writer = Mutex::new(TrajectoryWriter::new(&output_path).expect("Failed to create trajectory writer"));
    // Sidecar file for depth-N eval scores (one f32 per state, NaN for terminal/random)
    let scores_path = format!("{}.scores", output_path);
    let scores_writer = Mutex::new(std::io::BufWriter::new(
        std::fs::File::create(&scores_path).expect("Failed to create scores file"),
    ));
    let start = Instant::now();

    // Progress reporting interval
    let report_interval = std::cmp::max(1, num_games / 200); // ~200 progress reports

    eprintln!("Generating {} games using {} threads ...", num_games, rayon::current_num_threads());
    eprintln!();

    // Generate games in parallel using rayon
    // Each thread plays a batch of games and collects them, then writes to the shared writer.
    // We use chunks to reduce lock contention on the writer.
    let chunk_size = 50u64; // games per chunk
    let num_chunks = (num_games + chunk_size - 1) / chunk_size;

    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
        let chunk_start = chunk_idx * chunk_size;
        let chunk_end = std::cmp::min(chunk_start + chunk_size, num_games);

        // Each chunk gets a deterministic RNG derived from seed + chunk index
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(chunk_idx));
        let bag = create_bag();

        let mut chunk_games: Vec<GameData> = Vec::new();

        for _game_idx in chunk_start..chunk_end {
            let initial = create_initial_state(&bag);
            let game_data = play_depth_game(&initial, evaluator.as_ref(), depth, &mut rng, epsilon);
            chunk_games.push(game_data);
        }

        // Write the chunk to disk under the lock
        {
            let mut w = writer.lock().unwrap();
            let mut sw = scores_writer.lock().unwrap();
            for gd in &chunk_games {
                total_states.fetch_add(gd.states.len() as u64, Ordering::Relaxed);
                w.write_game(&gd.states, &gd.result).expect("Failed to write game");
                // Write scores for each state
                for &score in &gd.scores {
                    use std::io::Write;
                    (&mut *sw).write_all(&score.to_le_bytes()).expect("Failed to write score");
                }
            }
        }

        let done = completed.fetch_add(chunk_end - chunk_start, Ordering::Relaxed) + (chunk_end - chunk_start);
        if done % report_interval < (chunk_end - chunk_start) || done >= num_games {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = done as f64 / elapsed;
            let eta_secs = if rate > 0.0 { (num_games as f64 - done as f64) / rate } else { 0.0 };
            let eta_min = eta_secs / 60.0;
            eprintln!(
                "  {}/{} games ({:.1}%) {:.1} games/s, ETA {:.0}m",
                done, num_games,
                done as f64 / num_games as f64 * 100.0,
                rate,
                eta_min,
            );
        }
    });

    // Finalize files
    {
        let mut sw = scores_writer.into_inner().unwrap();
        use std::io::Write;
        sw.flush().expect("Failed to flush scores file");
    }
    let w = writer.into_inner().unwrap();
    let written = w.finish().expect("Failed to finalize trajectory file");

    let elapsed = start.elapsed();
    let total_st = total_states.load(Ordering::Relaxed);
    let file_size = std::fs::metadata(&output_path).map(|m| m.len()).unwrap_or(0);

    eprintln!();
    eprintln!("=== Generation Complete ===");
    eprintln!("  Games written:   {}", written);
    eprintln!("  Total states:    {} ({:.1} avg/game)", total_st, total_st as f64 / written as f64);
    eprintln!("  Time:            {:.1}s ({:.1}m)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / 60.0);
    eprintln!("  Rate:            {:.1} games/s", written as f64 / elapsed.as_secs_f64());
    eprintln!("  File size:       {:.1} MB", file_size as f64 / (1024.0 * 1024.0));
    let scores_size = std::fs::metadata(&scores_path).map(|m| m.len()).unwrap_or(0);
    eprintln!("  Scores file:     {} ({:.1} MB)", scores_path, scores_size as f64 / (1024.0 * 1024.0));
    eprintln!("  Output:          {}", output_path);
}
