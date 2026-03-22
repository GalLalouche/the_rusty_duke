//! Train LR on 41 combined features and benchmark vs heuristic.

use std::time::Instant;

use duke_training::feature_cache::load_feature_cache;
use duke_training::game_setup::{create_bag, create_initial_state, StaticHeuristicEvaluator};
use duke_training::learned_heuristic::{CombinedWeights, NUM_COMBINED_FEATURES};
use duke_training::match_runner::{run_matches, Player};

use duke_rust::game::state::GameResult;

const K: usize = NUM_COMBINED_FEATURES; // 41

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let features_path = args.iter().position(|a| a == "--features")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("D:/temp/ckpt_heuristic_1m/features_combined41.bin");

    let bench_games: u32 = args.iter().position(|a| a == "--bench-games")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);

    let t = Instant::now();
    let (header, games) = load_feature_cache(features_path).expect("Failed to load feature cache");
    let total_samples: usize = games.iter().map(|g| g.states.len()).sum();
    println!("Loaded {} games ({} samples, {} features) in {:.1?}",
        header.num_games, total_samples, header.num_features, t.elapsed());
    assert_eq!(header.num_features, K, "Expected {} features, got {}", K, header.num_features);

    // Build normal equations (X'X and X'y)
    let t = Instant::now();
    let mut xtx = [[0.0f64; K]; K];
    let mut xty = [0.0f64; K];
    let mut n: u64 = 0;

    for game in &games {
        for state in &game.states {
            let target = match game.result {
                GameResult::Won(winner) => {
                    if winner == state.current_player { 1.0 } else { -1.0 }
                }
                GameResult::Tie | GameResult::Ongoing => 0.0,
            };
            let x = &state.features;
            for i in 0..K {
                xty[i] += x[i] * target;
                for j in i..K {
                    xtx[i][j] += x[i] * x[j];
                }
            }
            n += 1;
        }
    }
    // Mirror upper triangle
    for i in 0..K {
        for j in 0..i { xtx[i][j] = xtx[j][i]; }
    }
    println!("Accumulated {} samples in {:.1?}\n", n, t.elapsed());

    let bag = create_bag();
    let gs = create_initial_state(&bag);
    let heuristic_eval = StaticHeuristicEvaluator::new();
    let heuristic_player = Player::Evaluator(&heuristic_eval);

    // Solve with a small ridge lambda for numerical stability
    let lambda = 1e-6 * n as f64;
    let w = solve_ridge(&xtx, &xty, lambda);

    // Print non-zero weights
    let names = [
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

    println!("=== Learned weights ({} features) ===\n", K);
    for (i, &val) in w.iter().enumerate() {
        let name = if i < names.len() { names[i] } else { "?" };
        if val.abs() > 1e-10 {
            println!("  [{:>2}] {:>20} = {:+.6e}", i, name, val);
        }
    }

    // Create evaluator and benchmark
    let mut combined = CombinedWeights::default();
    combined.weights = w;
    let lr_player = Player::Evaluator(&combined);

    println!("\n=== Benchmark: {} games ===\n", bench_games);
    run_matches(&gs, &lr_player, &heuristic_player, bench_games, "LR-41 vs Heuristic");
    let random = Player::Random;
    run_matches(&gs, &lr_player, &random, bench_games, "LR-41 vs Random");
    run_matches(&gs, &heuristic_player, &random, bench_games, "Heuristic vs Random (baseline)");
}

fn solve_ridge(xtx: &[[f64; K]; K], xty: &[f64; K], lambda: f64) -> [f64; K] {
    let mut a = *xtx;
    let mut b = *xty;
    for i in 0..K { a[i][i] += lambda; }
    gauss_solve(&mut a, &mut b)
}

fn gauss_solve(a: &mut [[f64; K]; K], b: &mut [f64; K]) -> [f64; K] {
    for col in 0..K {
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..K {
            let val = a[row][col].abs();
            if val > max_val { max_val = val; max_row = row; }
        }
        if max_row != col { a.swap(col, max_row); b.swap(col, max_row); }
        let pivot = a[col][col];
        if pivot.abs() < 1e-30 { continue; }
        for row in (col + 1)..K {
            let factor = a[row][col] / pivot;
            for j in col..K { a[row][j] -= factor * a[col][j]; }
            b[row] -= factor * b[col];
        }
    }
    let mut w = [0.0f64; K];
    for i in (0..K).rev() {
        let mut sum = b[i];
        for j in (i + 1)..K { sum -= a[i][j] * w[j]; }
        if a[i][i].abs() > 1e-30 { w[i] = sum / a[i][i]; }
    }
    w
}
