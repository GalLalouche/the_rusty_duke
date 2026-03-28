//! Deduplicate game positions from trajectories and label them with LR-Cheap depth-N minimax.
//!
//! Usage:
//!   label_positions --trajectories D:/temp/ckpt_heuristic_1m/trajectories.dtrj \
//!                   --evaluator D:/temp/combined_lr_41.json \
//!                   --depth 2 \
//!                   --output D:/temp/labeled_positions.bin \
//!                   [--dedup-output D:/temp/labeled_positions_dedup.bin]
//!
//! The --dedup-output flag saves deduplicated positions (full GameStates + counts) BEFORE
//! labeling, so they can be re-labeled later with different evaluators/depths without
//! re-deduplicating. Default: same directory as --output with `_dedup.bin` suffix.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::{Write, BufWriter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rayon::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use duke_rust::game::state::{GameResult, GameState};
use duke_rust::game::tile::Owner;

use duke_training::cli::parse_flag;
use duke_training::encoding::{active_board_features, bag_features};
use duke_training::game_setup::negamax;
use duke_training::learned_heuristic::CombinedWeights;
use duke_training::trajectory_io::{load_trajectories, write_game_state};

// --- Labeled output format ---
// Magic: "LPOS" (4 bytes)
// Version: 1 (u32 LE)
// num_positions: u32 LE
// Per position:
//   num_active: u16 LE
//   active_indices: [u16 LE; num_active]  (board feature indices, each < 1080)
//   bag_features: [f32 LE; 26]
//   label: f32 LE
//   count: u32 LE

const OUTPUT_MAGIC: &[u8; 4] = b"LPOS";
const OUTPUT_VERSION: u32 = 1;

// --- Dedup output format ---
// Magic: "DPOS" (4 bytes)
// Version: 1 (u32 LE)
// num_positions: u32 LE
// Per position:
//   GameState serialized via trajectory_io::write_game_state
//   count: u32 LE

const DEDUP_MAGIC: &[u8; 4] = b"DPOS";
const DEDUP_VERSION: u32 = 1;

/// Compute a 64-bit hash of the board state for deduplication.
///
/// Two positions with the same hash have identical board layouts (tile types,
/// owners, sides at each cell), the same current player, and the same bag
/// contents. This is everything that matters for position evaluation.
fn position_key(gs: &GameState) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();

    // Hash board contents in canonical cell order
    for y in 0..6u8 {
        for x in 0..6u8 {
            let c = duke_rust::common::coordinates::Coordinates { x, y };
            match gs.board().get(c) {
                Some(t) => {
                    1u8.hash(&mut h);
                    t.tile_type.hash(&mut h);
                    t.owner.hash(&mut h);
                    t.current_side.hash(&mut h);
                }
                None => 0u8.hash(&mut h),
            }
        }
    }

    // Hash current player
    gs.current_player_turn().hash(&mut h);

    // Hash bag contents (sorted tile types for each player)
    let mut top_bag: Vec<u8> = gs.bag_for_owner(Owner::TopPlayer)
        .remaining().iter().map(|t| *t as u8).collect();
    top_bag.sort();
    top_bag.hash(&mut h);

    let mut bottom_bag: Vec<u8> = gs.bag_for_owner(Owner::BottomPlayer)
        .remaining().iter().map(|t| *t as u8).collect();
    bottom_bag.sort();
    bottom_bag.hash(&mut h);

    // Hash discard piles
    let mut top_disc: Vec<u8> = gs.discard_bag_for(Owner::TopPlayer)
        .existing().iter().map(|t| *t as u8).collect();
    top_disc.sort();
    top_disc.hash(&mut h);

    let mut bottom_disc: Vec<u8> = gs.discard_bag_for(Owner::BottomPlayer)
        .existing().iter().map(|t| *t as u8).collect();
    bottom_disc.sort();
    bottom_disc.hash(&mut h);

    h.finish()
}

/// Label a single position using negamax at the specified depth.
fn label_position(gs: &GameState, evaluator: &CombinedWeights, depth: u32) -> f32 {
    let mut rng = StdRng::seed_from_u64(0);
    negamax(gs, evaluator, depth, &mut rng) as f32
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let trajectories_path: String = parse_flag(&args, "--trajectories")
        .unwrap_or_else(|| {
            eprintln!("Usage: label_positions --trajectories <path> --evaluator <path> --depth <N> --output <path>");
            std::process::exit(1);
        });
    let evaluator_path: String = parse_flag(&args, "--evaluator")
        .unwrap_or_else(|| {
            eprintln!("Missing --evaluator flag");
            std::process::exit(1);
        });
    let depth: u32 = parse_flag(&args, "--depth").unwrap_or(2);
    let output_path: String = parse_flag(&args, "--output")
        .unwrap_or_else(|| {
            eprintln!("Missing --output flag");
            std::process::exit(1);
        });
    let dedup_output_path: String = parse_flag(&args, "--dedup-output")
        .unwrap_or_else(|| {
            // Default: same directory as --output, with _dedup.bin suffix
            let p = std::path::Path::new(&output_path);
            let stem = p.file_stem().unwrap_or_default().to_string_lossy();
            let parent = p.parent().unwrap_or(std::path::Path::new("."));
            parent.join(format!("{}_dedup.bin", stem)).to_string_lossy().into_owned()
        });

    // --- Step 1: Load trajectories ---
    eprintln!("Loading trajectories from {} ...", trajectories_path);
    let t0 = Instant::now();
    let games = load_trajectories(&trajectories_path)
        .expect("Failed to load trajectories");
    let total_states: usize = games.iter().map(|g| g.states.len()).sum();
    eprintln!("Loaded {} games, {} states in {:.1}s",
        games.len(), total_states, t0.elapsed().as_secs_f64());

    // --- Step 2: Deduplicate positions ---
    eprintln!("Deduplicating positions ...");
    let t1 = Instant::now();

    // Count only ongoing (non-terminal) positions
    let mut ongoing_count: usize = 0;
    let mut dedup: HashMap<u64, (GameState, u32)> = HashMap::new();
    for game in &games {
        for gs in &game.states {
            if gs.game_result() != GameResult::Ongoing {
                continue;
            }
            ongoing_count += 1;
            let key = position_key(gs);
            dedup.entry(key)
                .and_modify(|(_, count)| *count += 1)
                .or_insert_with(|| (gs.clone(), 1));
        }
    }

    let unique = dedup.len();
    let ratio = if ongoing_count > 0 {
        unique as f64 / ongoing_count as f64 * 100.0
    } else {
        0.0
    };
    eprintln!("Deduplicated {} ongoing positions to {} unique ({:.1}% unique) in {:.1}s",
        ongoing_count, unique, ratio, t1.elapsed().as_secs_f64());

    // Free trajectory memory -- we only need the dedup map from here
    drop(games);

    // --- Step 2b: Save deduplicated positions (before labeling) ---
    eprintln!("Saving {} deduped positions to {} ...", unique, dedup_output_path);
    let t_dedup = Instant::now();
    {
        let f = std::fs::File::create(&dedup_output_path)
            .expect("Failed to create dedup output file");
        let mut w = BufWriter::new(f);

        // Header
        w.write_all(DEDUP_MAGIC).unwrap();
        w.write_all(&DEDUP_VERSION.to_le_bytes()).unwrap();
        w.write_all(&(unique as u32).to_le_bytes()).unwrap();

        for (gs, count) in dedup.values() {
            write_game_state(&mut w, gs).unwrap();
            w.write_all(&count.to_le_bytes()).unwrap();
        }
        w.flush().unwrap();
    }
    let dedup_size = std::fs::metadata(&dedup_output_path).unwrap().len();
    eprintln!("Saved {} deduped positions to {} ({:.1}MB) in {:.1}s",
        unique, dedup_output_path,
        dedup_size as f64 / (1024.0 * 1024.0),
        t_dedup.elapsed().as_secs_f64());

    // --- Step 3: Load evaluator ---
    eprintln!("Loading LR-Cheap evaluator from {} ...", evaluator_path);
    let evaluator = CombinedWeights::load(&evaluator_path)
        .expect("Failed to load CombinedWeights");

    // --- Step 4: Label with depth-N negamax ---
    eprintln!("Labeling {} unique positions with depth {} ...", unique, depth);
    let t2 = Instant::now();

    let positions: Vec<(u64, GameState, u32)> = dedup.into_iter()
        .map(|(k, (gs, count))| (k, gs, count))
        .collect();

    let done = AtomicU64::new(0);
    let total = positions.len() as u64;

    // Parallel labeling
    let labeled: Vec<(GameState, f32, u32)> = positions.par_iter()
        .map(|(_, gs, count)| {
            let label = label_position(gs, &evaluator, depth);
            let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
            if completed % 10000 == 0 || completed == total {
                let elapsed = t2.elapsed().as_secs_f64();
                let rate = completed as f64 / elapsed;
                eprintln!("  Labeled {}/{} ({:.2}%) [{:.1} pos/sec]",
                    completed, total,
                    completed as f64 / total as f64 * 100.0,
                    rate);
            }
            (gs.clone(), label, *count)
        })
        .collect();

    eprintln!("Labeling complete in {:.1}s", t2.elapsed().as_secs_f64());

    // --- Step 5: Save labeled dataset ---
    eprintln!("Saving {} labeled positions to {} ...", labeled.len(), output_path);
    let t3 = Instant::now();

    let f = std::fs::File::create(&output_path)
        .expect("Failed to create output file");
    let mut w = BufWriter::new(f);

    // Header
    w.write_all(OUTPUT_MAGIC).unwrap();
    w.write_all(&OUTPUT_VERSION.to_le_bytes()).unwrap();
    w.write_all(&(labeled.len() as u32).to_le_bytes()).unwrap();

    for (gs, label, count) in &labeled {
        let board_feats = active_board_features(gs);
        let bag_feats = bag_features(gs);

        // num_active
        let num_active = board_feats.len() as u16;
        w.write_all(&num_active.to_le_bytes()).unwrap();

        // active_indices
        for &idx in board_feats.as_slice() {
            w.write_all(&(idx as u16).to_le_bytes()).unwrap();
        }

        // bag_features (26 f32)
        for &val in &bag_feats {
            w.write_all(&val.to_le_bytes()).unwrap();
        }

        // label
        w.write_all(&label.to_le_bytes()).unwrap();

        // count
        w.write_all(&count.to_le_bytes()).unwrap();
    }

    w.flush().unwrap();
    let file_size = std::fs::metadata(&output_path).unwrap().len();
    eprintln!("Saved {} labeled positions to {} ({:.1} MB) in {:.1}s",
        labeled.len(), output_path,
        file_size as f64 / (1024.0 * 1024.0),
        t3.elapsed().as_secs_f64());

    // Summary stats
    let labels: Vec<f32> = labeled.iter().map(|(_, l, _)| *l).collect();
    let min = labels.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = labels.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = labels.iter().map(|l| *l as f64).sum::<f64>() / labels.len() as f64;
    eprintln!("Label stats: min={:.4}, max={:.4}, mean={:.4}", min, max, mean);
}
