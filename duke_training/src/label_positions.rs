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
use duke_training::position_hash::position_key;
use duke_training::learned_heuristic::{CombinedWeights, extract_combined_features, NUM_COMBINED_FEATURES};
use duke_training::supervised_common::COMBINED_FEATURE_NAMES;
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

// --- Feature-Labeled output format (FLPS) ---
// Magic: "FLPS" (4 bytes)
// Version: 1 (u32 LE)
// num_positions: u32 LE
// num_labels: u32 LE  (= 41 for cheap features)
// Per position:
//   num_active: u16 LE
//   active_indices: [u16 LE; num_active]
//   bag_features: [f32 LE; 26]
//   labels: [f32 LE; num_labels]
//   count: u32 LE

const FLPS_MAGIC: &[u8; 4] = b"FLPS";
const FLPS_VERSION: u32 = 1;


// --- Dedup output format ---
// Magic: "DPOS" (4 bytes)
// Version: 1 (u32 LE)
// num_positions: u32 LE
// Per position:
//   GameState serialized via trajectory_io::write_game_state
//   count: u32 LE

const DEDUP_MAGIC: &[u8; 4] = b"DPOS";
const DEDUP_VERSION: u32 = 1;

/// Label a single position using negamax at the specified depth.
fn label_position(gs: &GameState, evaluator: &CombinedWeights, depth: u32) -> f32 {
    let mut rng = StdRng::seed_from_u64(0);
    negamax(&mut gs.clone(), evaluator, depth, &mut rng) as f32
}

/// Synthetic label: count moves for only duke + footman tiles (current player),
/// ignoring guard. Same semantics as the my_moves feature (feature 4) but
/// filtered to only the simplest piece types.
fn synthetic_label(gs: &GameState) -> f32 {
    use duke_rust::game::board::PossibleMove;
    use duke_rust::game::tile::TileType;
    let owner = gs.current_player_turn();
    let mut count = 0u32;
    for mv in gs.all_valid_game_moves_for_ignoring_guard(owner) {
        if let PossibleMove::ApplyNonCommandTileAction { src, .. } = &mv {
            if let Some(tile) = gs.board().get(*src) {
                if tile.tile_type == TileType::Duke || tile.tile_type == TileType::Footman {
                    count += 1;
                }
            }
        }
    }
    count as f32
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let trajectories_path: String = parse_flag(&args, "--trajectories")
        .unwrap_or_else(|| {
            eprintln!("Usage: label_positions --trajectories <path> --evaluator <path> --depth <N> --output <path> [--synthetic] [--all-features]");
            std::process::exit(1);
        });
    let synthetic_mode = args.contains(&"--synthetic".to_string());
    let all_features_mode = args.contains(&"--all-features".to_string());
    if synthetic_mode && all_features_mode {
        eprintln!("--synthetic and --all-features are mutually exclusive");
        std::process::exit(1);
    }
    let evaluator_path: Option<String> = parse_flag(&args, "--evaluator");
    let scores_path: Option<String> = parse_flag(&args, "--scores");
    if !synthetic_mode && !all_features_mode && scores_path.is_none() && evaluator_path.is_none() {
        eprintln!("Missing --evaluator or --scores flag");
        std::process::exit(1);
    }
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

    // --- Precomputed scores mode: skip dedup, use scores directly ---
    if let Some(ref sp) = scores_path {
        eprintln!("Loading precomputed scores from {} ...", sp);
        let score_data = std::fs::read(sp).expect("Failed to read scores file");
        let scores: Vec<f32> = score_data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(scores.len(), total_states,
            "Scores file has {} entries but trajectories have {} states", scores.len(), total_states);
        eprintln!("  Loaded {} scores", scores.len());

        // Collect non-terminal, non-NaN positions with their scores
        let mut labeled: Vec<(GameState, f32, u32)> = Vec::new();
        let mut score_idx = 0usize;
        for game in &games {
            for gs in &game.states {
                let score = scores[score_idx];
                score_idx += 1;
                if score.is_nan() { continue; }
                if gs.clone().game_result() != GameResult::Ongoing { continue; }
                labeled.push((gs.clone(), score, 1));
            }
        }
        eprintln!("  {} labeled positions (skipped NaN/terminal)", labeled.len());

        // Save as LPOS
        eprintln!("Saving {} labeled positions to {} ...", labeled.len(), output_path);
        let f = std::fs::File::create(&output_path).expect("Failed to create output file");
        let mut w = BufWriter::new(f);
        w.write_all(OUTPUT_MAGIC).unwrap();
        w.write_all(&OUTPUT_VERSION.to_le_bytes()).unwrap();
        w.write_all(&(labeled.len() as u32).to_le_bytes()).unwrap();
        for (gs, label, count) in &labeled {
            let board_feats = active_board_features(gs);
            let bag_feats = bag_features(gs);
            let num_active = board_feats.len() as u16;
            w.write_all(&num_active.to_le_bytes()).unwrap();
            for &idx in board_feats.as_slice() {
                w.write_all(&(idx as u16).to_le_bytes()).unwrap();
            }
            for &val in &bag_feats {
                w.write_all(&val.to_le_bytes()).unwrap();
            }
            w.write_all(&label.to_le_bytes()).unwrap();
            w.write_all(&count.to_le_bytes()).unwrap();
        }
        w.flush().unwrap();
        let file_size = std::fs::metadata(&output_path).map(|m| m.len()).unwrap_or(0);
        eprintln!("Saved {} positions to {} ({:.1} MB)",
            labeled.len(), output_path, file_size as f64 / (1024.0 * 1024.0));

        let labels: Vec<f32> = labeled.iter().map(|(_, l, _)| *l).collect();
        let min = labels.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = labels.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = labels.iter().map(|l| *l as f64).sum::<f64>() / labels.len() as f64;
        eprintln!("Label stats: min={:.4}, max={:.4}, mean={:.4}", min, max, mean);
        return;
    }

    // --- Step 2: Deduplicate positions ---
    eprintln!("Deduplicating positions ...");
    let t1 = Instant::now();

    // Count only ongoing (non-terminal) positions
    let mut ongoing_count: usize = 0;
    let mut dedup: HashMap<u64, (GameState, u32)> = HashMap::new();
    for game in &games {
        for gs in &game.states {
            if gs.clone().game_result() != GameResult::Ongoing {
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

    // --- Step 3: Load evaluator (unless synthetic or all-features mode) ---
    let evaluator = if synthetic_mode {
        eprintln!("*** SYNTHETIC LABEL MODE ***");
        eprintln!("  f(gs) = (num_pieces * sum_tile_values) + (turn * 100) + bag_size + (duke_x * duke_y) / max(1, flipped)");
        None
    } else if all_features_mode {
        eprintln!("*** ALL-FEATURES LABEL MODE ***");
        eprintln!("  Labeling each position with all {} combined features", NUM_COMBINED_FEATURES);
        None
    } else {
        let path = evaluator_path.as_ref().unwrap();
        eprintln!("Loading LR-Cheap evaluator from {} ...", path);
        Some(CombinedWeights::load(path)
            .expect("Failed to load CombinedWeights"))
    };

    // --- Step 4: Label positions ---
    let positions: Vec<(u64, GameState, u32)> = dedup.into_iter()
        .map(|(k, (gs, count))| (k, gs, count))
        .collect();

    let done = AtomicU64::new(0);
    let total = positions.len() as u64;

    if all_features_mode {
        // --- All-features mode: compute 41 features per position, save as FLPS ---
        eprintln!("Computing {} combined features for {} positions ...", NUM_COMBINED_FEATURES, unique);
        let t2 = Instant::now();

        let labeled: Vec<(GameState, [f32; NUM_COMBINED_FEATURES], u32)> = positions.par_iter()
            .map(|(_, gs, count)| {
                let features_f64 = extract_combined_features(gs);
                let mut features_f32 = [0.0f32; NUM_COMBINED_FEATURES];
                for i in 0..NUM_COMBINED_FEATURES {
                    features_f32[i] = features_f64[i] as f32;
                }
                let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
                if completed % 10000 == 0 || completed == total {
                    let elapsed = t2.elapsed().as_secs_f64();
                    let rate = completed as f64 / elapsed;
                    eprintln!("  Computed {}/{} ({:.2}%) [{:.1} pos/sec]",
                        completed, total,
                        completed as f64 / total as f64 * 100.0,
                        rate);
                }
                (gs.clone(), features_f32, *count)
            })
            .collect();

        eprintln!("Feature extraction complete in {:.1}s", t2.elapsed().as_secs_f64());

        // Print per-feature stats
        eprintln!("\nPer-feature statistics:");
        for fi in 0..NUM_COMBINED_FEATURES {
            let vals: Vec<f32> = labeled.iter().map(|(_, f, _)| f[fi]).collect();
            let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = vals.iter().map(|v| *v as f64).sum::<f64>() / vals.len() as f64;
            eprintln!("  [{:2}] {:30} min={:8.2} max={:8.2} mean={:8.4}",
                fi, COMBINED_FEATURE_NAMES[fi], min, max, mean);
        }

        // Normalize each feature independently to [-10, +10] range
        // so the trainer's clamp-to-+-10 mapping preserves full dynamic range.
        let mut labeled = labeled;
        eprintln!("\nNormalizing features to [-10, +10]:");
        for fi in 0..NUM_COMBINED_FEATURES {
            let min_v = labeled.iter().map(|(_, f, _)| f[fi]).fold(f32::INFINITY, f32::min);
            let max_v = labeled.iter().map(|(_, f, _)| f[fi]).fold(f32::NEG_INFINITY, f32::max);
            let half_range = ((max_v - min_v) / 2.0).max(1e-6);
            let mid = (min_v + max_v) / 2.0;
            let scale = 10.0 / half_range;
            for (_, features, _) in labeled.iter_mut() {
                features[fi] = (features[fi] - mid) * scale;
            }
            eprintln!("  [{:2}] {:30} [{:.1}, {:.1}] -> [-10, +10] (scale={:.4})",
                fi, COMBINED_FEATURE_NAMES[fi], min_v, max_v, scale);
        }

        // --- Save as FLPS ---
        eprintln!("\nSaving {} feature-labeled positions to {} ...", labeled.len(), output_path);
        let t3 = Instant::now();

        let f = std::fs::File::create(&output_path)
            .expect("Failed to create output file");
        let mut w = BufWriter::new(f);

        // Header
        w.write_all(FLPS_MAGIC).unwrap();
        w.write_all(&FLPS_VERSION.to_le_bytes()).unwrap();
        w.write_all(&(labeled.len() as u32).to_le_bytes()).unwrap();
        w.write_all(&(NUM_COMBINED_FEATURES as u32).to_le_bytes()).unwrap();

        for (gs, features, count) in &labeled {
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

            // labels (41 f32)
            for &val in features.iter() {
                w.write_all(&val.to_le_bytes()).unwrap();
            }

            // count
            w.write_all(&count.to_le_bytes()).unwrap();
        }

        w.flush().unwrap();
        let file_size = std::fs::metadata(&output_path).unwrap().len();
        eprintln!("Saved {} feature-labeled positions to {} ({:.1} MB) in {:.1}s",
            labeled.len(), output_path,
            file_size as f64 / (1024.0 * 1024.0),
            t3.elapsed().as_secs_f64());
    } else {
        // --- Standard mode: single label per position (LPOS) ---
        let label_desc = if synthetic_mode { "synthetic function" } else { &format!("depth {}", depth) };
        eprintln!("Labeling {} unique positions with {} ...", unique, label_desc);
        let t2 = Instant::now();

        // Parallel labeling
        let labeled: Vec<(GameState, f32, u32)> = positions.par_iter()
            .map(|(_, gs, count)| {
                let label = if synthetic_mode {
                    synthetic_label(gs)
                } else {
                    label_position(gs, evaluator.as_ref().unwrap(), depth)
                };
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

        // Normalize synthetic labels to +-10 range (center at mean, scale by half-range)
        // so the trainer's clamp-to-+-10 mapping gives a good spread.
        let mut labeled = labeled;
        if synthetic_mode {
            let min_l = labeled.iter().map(|(_, l, _)| *l).fold(f32::INFINITY, f32::min);
            let max_l = labeled.iter().map(|(_, l, _)| *l).fold(f32::NEG_INFINITY, f32::max);
            let mid = (min_l + max_l) / 2.0;
            let half_range = ((max_l - min_l) / 2.0).max(1e-6);
            let scale = 10.0 / half_range;
            eprintln!("  Normalizing synthetic labels: [{:.2}, {:.2}] -> [-10, +10] (center={:.2}, scale={:.4})",
                min_l, max_l, mid, scale);
            for (_, label, _) in labeled.iter_mut() {
                *label = (*label - mid) * scale;
            }
        }

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
}
