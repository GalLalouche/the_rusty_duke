//! Train LR on combined features and benchmark vs heuristic.
//!
//! Modes:
//!   --mode combined41  (default) — train on 41 combined features from a single cache
//!   --mode all65                 — merge features.bin (24) + features_combined41.bin (41) = 65 features

use std::io::BufReader;
use std::time::Instant;

use duke_training::feature_cache::{stream_feature_cache, FeatureCacheHeader, parse_header, read_one_game};
use duke_training::game_setup::{create_bag, create_initial_state, StaticHeuristicEvaluator};
use duke_training::learned_heuristic::{
    AllFeaturesWeights, CombinedWeights,
    NUM_ALL_FEATURES, NUM_COMBINED_FEATURES, NUM_FEATURES,
};
use duke_training::match_runner::{run_matches, Player};

use duke_rust::game::state::GameResult;

/// Dynamic ridge regression accumulator for arbitrary K.
struct RidgeAccumulator {
    k: usize,
    xtx: Vec<f64>,  // k*k, row-major
    xty: Vec<f64>,  // k
    n: u64,
}

impl RidgeAccumulator {
    fn new(k: usize) -> Self {
        Self {
            k,
            xtx: vec![0.0; k * k],
            xty: vec![0.0; k],
            n: 0,
        }
    }

    /// Accumulate one sample. Only fills upper triangle of xtx for speed.
    fn add(&mut self, x: &[f64], target: f64) {
        debug_assert_eq!(x.len(), self.k);
        let k = self.k;
        for i in 0..k {
            self.xty[i] += x[i] * target;
            for j in i..k {
                self.xtx[i * k + j] += x[i] * x[j];
            }
        }
        self.n += 1;
    }

    /// Mirror upper triangle to lower triangle.
    fn symmetrize(&mut self) {
        let k = self.k;
        for i in 0..k {
            for j in 0..i {
                self.xtx[i * k + j] = self.xtx[j * k + i];
            }
        }
    }

    /// Solve (X'X + lambda*I) w = X'y via Gaussian elimination with partial pivoting.
    fn solve(&self, lambda: f64) -> Vec<f64> {
        let k = self.k;
        let mut a = self.xtx.clone(); // k*k row-major
        let mut b = self.xty.clone(); // k

        // Add ridge penalty
        for i in 0..k {
            a[i * k + i] += lambda;
        }

        // Forward elimination with partial pivoting
        for col in 0..k {
            // Find pivot
            let mut max_row = col;
            let mut max_val = a[col * k + col].abs();
            for row in (col + 1)..k {
                let val = a[row * k + col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }
            // Swap rows
            if max_row != col {
                for j in 0..k {
                    a.swap(col * k + j, max_row * k + j);
                }
                b.swap(col, max_row);
            }

            let pivot = a[col * k + col];
            if pivot.abs() < 1e-30 {
                continue;
            }

            for row in (col + 1)..k {
                let factor = a[row * k + col] / pivot;
                for j in col..k {
                    a[row * k + j] -= factor * a[col * k + j];
                }
                b[row] -= factor * b[col];
            }
        }

        // Back substitution
        let mut w = vec![0.0f64; k];
        for i in (0..k).rev() {
            let mut sum = b[i];
            for j in (i + 1)..k {
                sum -= a[i * k + j] * w[j];
            }
            if a[i * k + i].abs() > 1e-30 {
                w[i] = sum / a[i * k + i];
            }
        }
        w
    }
}

// ── Feature names ─────────────────────────────────────────────────────────

const NAMES_24: &[&str] = &[
    "duke_mob_diff", "tile_count_diff", "total_mob_diff", "discard_diff",
    "duke_mob^2", "tile_cnt^2", "total_mob^2", "discard^2",
    "dukM*tilC", "dukM*totM", "dukM*disc", "tilC*totM", "tilC*disc", "totM*disc",
    "duke_mob^3", "tile_cnt^3", "total_mob^3", "discard^3",
    "guard_diff", "bag_empty_diff", "adjacency_diff", "center_diff", "duke_ratio_diff",
    "bias",
];

const NAMES_41: &[&str] = &[
    "manh_my_near_my_dk", "manh_en_near_my_dk", "manh_my_near_en_dk", "manh_en_near_en_dk",
    "my_approx_moves", "opp_approx_moves", "my_reach_sq", "opp_reach_sq", "contested_sq",
    "my_defended", "my_threatened", "opp_defended", "opp_threatened",
    "my_duke_mob", "opp_duke_mob",
    "my_disc_Duke", "my_disc_Foot", "my_disc_Pike", "my_disc_Knight", "my_disc_Champ",
    "my_disc_Drag", "my_disc_Wiz", "my_disc_Gen", "my_disc_Marsh", "my_disc_Assn",
    "my_disc_Priest", "my_disc_Bow", "my_disc_LBow",
    "op_disc_Duke", "op_disc_Foot", "op_disc_Pike", "op_disc_Knight", "op_disc_Champ",
    "op_disc_Drag", "op_disc_Wiz", "op_disc_Gen", "op_disc_Marsh", "op_disc_Assn",
    "op_disc_Priest", "op_disc_Bow", "op_disc_LBow",
];

fn feature_name(mode: &str, idx: usize) -> &'static str {
    match mode {
        "combined41" => {
            if idx < NAMES_41.len() { NAMES_41[idx] } else { "?" }
        }
        "all65" => {
            if idx < NAMES_24.len() {
                NAMES_24[idx]
            } else if idx - NUM_FEATURES < NAMES_41.len() {
                NAMES_41[idx - NUM_FEATURES]
            } else {
                "?"
            }
        }
        _ => "?",
    }
}

// ── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mode = args.iter().position(|a| a == "--mode")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("combined41");

    let bench_games: u32 = args.iter().position(|a| a == "--bench-games")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);

    match mode {
        "combined41" => run_combined41(&args, bench_games),
        "all65" => run_all65(&args, bench_games),
        other => panic!("Unknown mode '{}'. Use 'combined41' or 'all65'.", other),
    }
}

/// Original mode: train LR on 41 combined features from a single cache.
fn run_combined41(args: &[String], bench_games: u32) {
    let features_path = args.iter().position(|a| a == "--features")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("D:/temp/ckpt_heuristic_1m/features_combined41.bin");

    let k = NUM_COMBINED_FEATURES;
    let t = Instant::now();
    let mut acc = RidgeAccumulator::new(k);
    let mut games_processed: usize = 0;

    let header = stream_feature_cache(features_path, 1000, |chunk, hdr| {
        assert_eq!(hdr.num_features, k, "Expected {} features, got {}", k, hdr.num_features);
        for game in chunk {
            for state in &game.states {
                let target = match game.result {
                    GameResult::Won(winner) => {
                        if winner == state.current_player { 1.0 } else { -1.0 }
                    }
                    GameResult::Tie => 0.0,
                    GameResult::Ongoing => continue,
                };
                acc.add(&state.features, target);
            }
        }
        games_processed += chunk.len();
        if games_processed % 10000 == 0 {
            eprint!("\r  {}/{} games ({} samples, {:.1?})", games_processed, hdr.num_games, acc.n, t.elapsed());
        }
    }).expect("Failed to stream feature cache");

    acc.symmetrize();
    eprintln!();
    println!("Streamed {} games ({} samples, {} features) in {:.1?}\n",
        header.num_games, acc.n, header.num_features, t.elapsed());

    let lambda = 1e-6 * acc.n as f64;
    let w = acc.solve(lambda);

    print_weights("combined41", &w);

    // Save weights
    let mut combined = CombinedWeights::default();
    combined.weights.copy_from_slice(&w);
    let output_path = "D:/temp/combined_lr_41.json";
    combined.save(output_path).expect("Failed to save combined weights");
    println!("Weights saved to {}", output_path);

    // Benchmark
    let bag = create_bag();
    let gs = create_initial_state(&bag);
    let heuristic_eval = StaticHeuristicEvaluator::new();
    let heuristic_player = Player::Evaluator(&heuristic_eval);
    let lr_player = Player::Evaluator(&combined);

    println!("\n=== Benchmark: {} games ===\n", bench_games);
    run_matches(&gs, &lr_player, &heuristic_player, bench_games, "LR-41 vs Heuristic");
    let random = Player::Random;
    run_matches(&gs, &lr_player, &random, bench_games, "LR-41 vs Random");
    run_matches(&gs, &heuristic_player, &random, bench_games, "Heuristic vs Random (baseline)");
}

/// New mode: merge features.bin (24) + features_combined41.bin (41) into 65 features.
fn run_all65(args: &[String], bench_games: u32) {
    let features24_path = args.iter().position(|a| a == "--features24")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("D:/temp/ckpt_heuristic_1m/features.bin");

    let features41_path = args.iter().position(|a| a == "--features41")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("D:/temp/ckpt_heuristic_1m/features_combined41.bin");

    println!("=== Mode: all65 (merging 24 + 41 features) ===");
    println!("  24-feature cache: {}", features24_path);
    println!("  41-feature cache: {}", features41_path);

    let k = NUM_ALL_FEATURES; // 65
    let chunk_size: usize = 1000;
    let t = Instant::now();
    let mut acc = RidgeAccumulator::new(k);
    let mut games_processed: usize = 0;

    // Open both files and parse headers
    let f24 = std::fs::File::open(features24_path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {}", features24_path, e));
    let mut r24 = BufReader::with_capacity(1 << 20, f24);
    let hdr24 = parse_header(&mut r24)
        .unwrap_or_else(|e| panic!("Failed to read header from {}: {}", features24_path, e));

    let f41 = std::fs::File::open(features41_path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {}", features41_path, e));
    let mut r41 = BufReader::with_capacity(1 << 20, f41);
    let hdr41 = parse_header(&mut r41)
        .unwrap_or_else(|e| panic!("Failed to read header from {}: {}", features41_path, e));

    // Validate headers
    assert_eq!(hdr24.num_features, NUM_FEATURES,
        "Expected {} features in 24-feature cache, got {}", NUM_FEATURES, hdr24.num_features);
    assert_eq!(hdr41.num_features, NUM_COMBINED_FEATURES,
        "Expected {} features in 41-feature cache, got {}", NUM_COMBINED_FEATURES, hdr41.num_features);
    assert_eq!(hdr24.num_games, hdr41.num_games,
        "Game count mismatch: 24-feature cache has {} games but 41-feature cache has {} games",
        hdr24.num_games, hdr41.num_games);

    let total_games = hdr24.num_games;
    println!("  Both caches: {} games, streaming in chunks of {}...", total_games, chunk_size);

    // Stream both caches in lockstep
    let mut games_read: usize = 0;
    while games_read < total_games {
        let batch = chunk_size.min(total_games - games_read);

        for _ in 0..batch {
            let game24 = read_one_game(&mut r24, hdr24.num_features)
                .expect("Failed to read game from 24-feature cache");
            let game41 = read_one_game(&mut r41, hdr41.num_features)
                .expect("Failed to read game from 41-feature cache");

            // Validate that both caches agree on the game result and state count
            assert_eq!(game24.result, game41.result,
                "Game result mismatch at game {}: 24-cache={:?}, 41-cache={:?}",
                games_read + 1, game24.result, game41.result);
            assert_eq!(game24.states.len(), game41.states.len(),
                "State count mismatch at game {}: 24-cache has {} states, 41-cache has {} states",
                games_read + 1, game24.states.len(), game41.states.len());

            for (s24, s41) in game24.states.iter().zip(game41.states.iter()) {
                assert_eq!(s24.current_player, s41.current_player,
                    "Player mismatch at game {}: 24-cache={:?}, 41-cache={:?}",
                    games_read + 1, s24.current_player, s41.current_player);

                let target = match game24.result {
                    GameResult::Won(winner) => {
                        if winner == s24.current_player { 1.0 } else { -1.0 }
                    }
                    GameResult::Tie => 0.0,
                    GameResult::Ongoing => continue,
                };

                // Concatenate: [24 features] ++ [41 features] = 65 features
                let mut x = Vec::with_capacity(k);
                x.extend_from_slice(&s24.features);
                x.extend_from_slice(&s41.features);
                debug_assert_eq!(x.len(), k);

                acc.add(&x, target);
            }
            games_read += 1;
        }

        games_processed += batch;
        if games_processed % 10000 == 0 || games_processed == total_games {
            eprint!("\r  {}/{} games ({} samples, {:.1?})", games_processed, total_games, acc.n, t.elapsed());
        }
    }

    acc.symmetrize();
    eprintln!();
    println!("Streamed {} games ({} samples, {} features) in {:.1?}\n",
        total_games, acc.n, k, t.elapsed());

    let lambda = 1e-6 * acc.n as f64;
    let w = acc.solve(lambda);

    print_weights("all65", &w);

    // Save weights
    let mut all_weights = AllFeaturesWeights::default();
    all_weights.weights.copy_from_slice(&w);
    let output_path = "D:/temp/combined_lr_65.json";
    all_weights.save(output_path).expect("Failed to save all-features weights");
    println!("Weights saved to {}", output_path);

    // Benchmark
    let bag = create_bag();
    let gs = create_initial_state(&bag);
    let heuristic_eval = StaticHeuristicEvaluator::new();
    let heuristic_player = Player::Evaluator(&heuristic_eval);
    let lr_player = Player::Evaluator(&all_weights);

    println!("\n=== Benchmark: {} games ===\n", bench_games);
    run_matches(&gs, &lr_player, &heuristic_player, bench_games, "LR-65 vs Heuristic");
    let random = Player::Random;
    run_matches(&gs, &lr_player, &random, bench_games, "LR-65 vs Random");
    run_matches(&gs, &heuristic_player, &random, bench_games, "Heuristic vs Random (baseline)");
}

fn print_weights(mode: &str, w: &[f64]) {
    println!("=== Learned weights ({} features) ===\n", w.len());
    for (i, &val) in w.iter().enumerate() {
        let name = feature_name(mode, i);
        if val.abs() > 1e-10 {
            println!("  [{:>2}] {:>20} = {:+.6e}", i, name, val);
        }
    }
}
