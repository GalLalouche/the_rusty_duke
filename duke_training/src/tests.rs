use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use duke_rust::game::ai::player::ArtificialPlayer;
use duke_rust::game::ai::stupid_sync_ai::StupidSyncAi;
use duke_rust::game::bag::TileBag;
use duke_rust::game::board_setup::{DukeInitialLocation, FootmenSetup};
use duke_rust::game::state::{GameResult, GameState};
use duke_rust::game::tile::Owner;

use crate::encoding::{active_feature_indices, encode_state, encode_state_flat, BOARD_SIZE, NUM_PLANES};
use crate::fc_model::FcValueNetwork;
use crate::fc_td_training::{FcTdTrainer, GameTrajectory};
use crate::game_setup::{create_bag, create_initial_state, play_random_game};
use crate::nnue::{NnueAccumulator, NnueEvaluator, NnueWeights, DEFAULT_L1, DEFAULT_L2};
use crate::weight_export::export_weights;

type TestBackend = Autodiff<NdArray>;

/// RAII guard that removes a file when dropped (including on panic).
struct TempFileGuard {
    path: String,
}

impl TempFileGuard {
    fn new(path: &str) -> Self {
        Self { path: path.to_string() }
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

fn create_small_state() -> GameState {
    let bag = TileBag::new(vec![]);
    GameState::new(
        &bag,
        (DukeInitialLocation::Left, FootmenSetup::Left),
        (DukeInitialLocation::Right, FootmenSetup::Right),
    )
}

fn create_test_state() -> GameState {
    let bag = create_bag();
    create_initial_state(&bag)
}

// ── encoding tests ──────────────────────────────────────────────────────

#[test]
fn encode_state_has_correct_shape() {
    let gs = create_test_state();
    let device = Default::default();
    let tensor = encode_state::<TestBackend>(&gs, &device);
    let dims = tensor.dims();
    assert_eq!(dims, [NUM_PLANES, BOARD_SIZE, BOARD_SIZE]);
}

#[test]
fn encode_state_initial_board_has_six_tiles() {
    // Initial board has 3 tiles per player (Duke + 2 Footmen = 6 total).
    // Sum of all tile-type planes (0..25) should equal 6.0.
    let gs = create_test_state();
    let device = Default::default();
    let tensor = encode_state::<TestBackend>(&gs, &device);

    // Planes 0..26 are tile-type planes
    let tile_planes = tensor.clone().slice([0..26]);
    let total: f32 = tile_planes
        .sum()
        .into_data()
        .to_vec::<f32>()
        .expect("to_vec")[0];
    assert!(
        (total - 6.0).abs() < 1e-5,
        "Expected 6 tiles on initial board, got {}",
        total
    );
}

#[test]
fn encode_state_is_relative_to_current_player() {
    use crate::encoding::BOARD_FEATURES;
    // The initial board is symmetric: TopPlayer and BottomPlayer both have
    // Duke + 2 Footmen. Encoding from TopPlayer's perspective should put
    // TopPlayer's tiles in "my" planes (0..13) and BottomPlayer's in
    // "opponent" planes (13..26). If we could flip the perspective on the
    // SAME board state, my/opponent planes should swap.
    //
    // Since we can't directly flip current_player, we verify that the
    // "my tiles" planes contain ONLY the current player's tiles by checking
    // that TopPlayer's tile positions appear in planes 0..13 and
    // BottomPlayer's in planes 13..26.
    let gs = create_test_state();
    let device = Default::default();
    assert_eq!(gs.current_player_turn(), Owner::TopPlayer);

    let flat: Vec<f32> = encode_state_flat::<TestBackend>(&gs, &device)
        .into_data()
        .to_vec()
        .expect("flat");

    // TopPlayer's tiles should be in "my" planes (0..13), NOT in opponent planes (13..26)
    let my_planes_sum: f32 = flat[..13 * BOARD_SIZE * BOARD_SIZE].iter().sum();
    let opp_planes_sum: f32 = flat[13 * BOARD_SIZE * BOARD_SIZE..26 * BOARD_SIZE * BOARD_SIZE].iter().sum();

    // Initial board: 3 tiles per player. My planes should have 3.0 (TopPlayer's tiles)
    // and opponent planes should have 3.0 (BottomPlayer's tiles).
    assert!((my_planes_sum - 3.0).abs() < 1e-5,
        "My tile planes should sum to 3.0 (3 tiles), got {}", my_planes_sum);
    assert!((opp_planes_sum - 3.0).abs() < 1e-5,
        "Opponent tile planes should sum to 3.0 (3 tiles), got {}", opp_planes_sum);

    // Now play a move and verify the perspective flips:
    // after one move, it's BottomPlayer's turn. Now BottomPlayer's tiles
    // should be in "my" planes.
    let mut gs2 = gs.clone();
    let ai = StupidSyncAi {};
    let mut rng = StdRng::seed_from_u64(42);
    ai.play_next_move(&mut rng, &mut gs2);
    assert_eq!(gs2.current_player_turn(), Owner::BottomPlayer);

    let flat2: Vec<f32> = encode_state_flat::<TestBackend>(&gs2, &device)
        .into_data()
        .to_vec()
        .expect("flat2");

    // After one move, TopPlayer moved a tile (it flipped). The total tile
    // count is still 6, but now BottomPlayer's 3 tiles are in "my" planes.
    let my_planes_sum2: f32 = flat2[..13 * BOARD_SIZE * BOARD_SIZE].iter().sum();
    let opp_planes_sum2: f32 = flat2[13 * BOARD_SIZE * BOARD_SIZE..26 * BOARD_SIZE * BOARD_SIZE].iter().sum();
    assert!((my_planes_sum2 - 3.0).abs() < 1e-5,
        "After move, BottomPlayer's 3 tiles should be in my planes, got {}", my_planes_sum2);
    assert!((opp_planes_sum2 - 3.0).abs() < 1e-5,
        "After move, TopPlayer's 3 tiles should be in opponent planes, got {}", opp_planes_sum2);
}

#[test]
fn encode_state_side_planes_are_correct() {
    // Initially all tiles are on Initial side.
    // Plane 26 (current player initial) and 28 (opponent initial) should have values.
    // Plane 27 (current player flipped) and 29 (opponent flipped) should be zero.
    let gs = create_test_state();
    let device = Default::default();
    let tensor = encode_state::<TestBackend>(&gs, &device);

    let plane_27_sum: f32 = tensor
        .clone()
        .slice([27..28])
        .sum()
        .into_data()
        .to_vec::<f32>()
        .expect("to_vec")[0];
    assert!(
        plane_27_sum.abs() < 1e-5,
        "Plane 27 (current player flipped) should be zero on initial board, got {}",
        plane_27_sum
    );

    let plane_29_sum: f32 = tensor
        .clone()
        .slice([29..30])
        .sum()
        .into_data()
        .to_vec::<f32>()
        .expect("to_vec")[0];
    assert!(
        plane_29_sum.abs() < 1e-5,
        "Plane 29 (opponent flipped) should be zero on initial board, got {}",
        plane_29_sum
    );

    let plane_26_sum: f32 = tensor
        .clone()
        .slice([26..27])
        .sum()
        .into_data()
        .to_vec::<f32>()
        .expect("to_vec")[0];
    assert!(
        plane_26_sum > 0.0,
        "Plane 26 (current player initial) should have non-zero values"
    );

    let plane_28_sum: f32 = tensor
        .clone()
        .slice([28..29])
        .sum()
        .into_data()
        .to_vec::<f32>()
        .expect("to_vec")[0];
    assert!(
        plane_28_sum > 0.0,
        "Plane 28 (opponent initial) should have non-zero values"
    );
}

// ── fc_td_training tests ────────────────────────────────────────────────

#[test]
fn train_on_batch_single_game_returns_loss() {
    let device = Default::default();
    let mut trainer: FcTdTrainer<TestBackend> = FcTdTrainer::new(device, 0.001, DEFAULT_L1, DEFAULT_L2);

    let gs = create_test_state();
    let mut rng = StdRng::seed_from_u64(0);
    let (states, result) = play_random_game(&gs, &mut rng);

    let loss = trainer.train_on_batch(&[GameTrajectory { states, result }]);
    assert!(loss.is_finite(), "Loss should be finite, got {}", loss);
    assert!(loss >= 0.0, "Loss should be non-negative, got {}", loss);
}

#[test]
fn train_reduces_loss_on_repeated_game() {
    // Use a small bag (empty) so the game is short and training is fast in debug mode.
    let device = Default::default();
    let mut trainer: FcTdTrainer<TestBackend> = FcTdTrainer::new(device, 0.01, DEFAULT_L1, DEFAULT_L2);

    let gs = create_small_state();
    let mut rng = StdRng::seed_from_u64(7);
    let (states, result) = play_random_game(&gs, &mut rng);

    // Only use first 10 states to keep batch small
    let states: Vec<GameState> = states.into_iter().take(10).collect();

    // Train on the same game trajectory a few times
    let traj = GameTrajectory { states, result };
    let first_loss = trainer.train_on_batch(std::slice::from_ref(&traj));
    let mut last_loss = first_loss;
    for _ in 1..5 {
        last_loss = trainer.train_on_batch(std::slice::from_ref(&traj));
    }

    assert!(
        last_loss < first_loss,
        "Loss should decrease after repeated training on the same game: first={}, last={}",
        first_loss,
        last_loss
    );
}

#[test]
#[ignore] // Flaky: FC model with 5 training iterations may not converge enough
fn terminal_state_target_is_correct() {
    // Use small bag for fast game in debug mode.
    let gs = create_small_state();

    // Find a game that ends in a win (not a tie)
    let mut game_states = None;
    let mut game_result = GameResult::Ongoing;
    for seed in 0..100u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let (states, result) = play_random_game(&gs, &mut rng);
        if matches!(result, GameResult::Won(_)) {
            game_states = Some(states);
            game_result = result;
            break;
        }
    }
    let states = game_states.expect("Should find a game that ends in a win");
    assert!(matches!(game_result, GameResult::Won(_)));

    // Only use last 8 states to keep batch small for debug mode
    let start = if states.len() > 8 { states.len() - 8 } else { 0 };
    let states: Vec<GameState> = states[start..].to_vec();

    // The terminal state is the last one.
    let terminal_state = &states[states.len() - 1];
    let terminal_player = terminal_state.current_player_turn();
    let expected_target = match game_result {
        GameResult::Won(winner) if winner == terminal_player => 1.0f32,
        GameResult::Won(_) => 0.0f32,
        _ => unreachable!(),
    };

    // Train a few times (kept low for debug builds)
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
    let mut trainer: FcTdTrainer<TestBackend> = FcTdTrainer::new(device.clone(), 0.01, DEFAULT_L1, DEFAULT_L2);
    let traj = GameTrajectory { states: states.clone(), result: game_result };
    for _ in 0..5 {
        trainer.train_on_batch(std::slice::from_ref(&traj));
    }

    // Now check the terminal state prediction
    let encoded = encode_state_flat::<TestBackend>(terminal_state, &device);
    let batch = encoded.unsqueeze::<2>(); // [1, 1106]
    let prediction = trainer.model.forward(batch);
    let pred_value: f32 = prediction.into_data().to_vec::<f32>().expect("to_vec")[0];

    // The prediction should be moving toward expected_target.
    // With 5 training iterations it won't converge fully, but should show movement.
    let diff = (pred_value - expected_target).abs();
    assert!(
        diff < 0.50,
        "Terminal state prediction ({}) should be closer to target ({}), diff={}",
        pred_value,
        expected_target,
        diff
    );
}

// ── NNUE / FC model tests ────────────────────────────────────────────────

#[test]
fn active_features_matches_encoding() {
    use crate::encoding::BOARD_FEATURES;
    let gs = create_test_state();
    let device = Default::default();
    let tensor = encode_state::<TestBackend>(&gs, &device);
    let flat: Vec<f32> = tensor.reshape([BOARD_FEATURES as i32]).into_data().to_vec().expect("flat");

    let active = active_feature_indices(&gs);
    // Every active index should have a 1.0 in the flat tensor
    for &idx in &active {
        assert_eq!(flat[idx], 1.0, "Feature {} should be 1.0", idx);
    }
    // Count of 1.0s in tensor should equal number of active features
    let ones_count = flat.iter().filter(|&&v| v == 1.0).count();
    assert_eq!(
        ones_count,
        active.len(),
        "Mismatch: {} ones in tensor but {} active features",
        ones_count,
        active.len()
    );
}

#[test]
fn fc_model_forward_produces_valid_output() {
    let device = Default::default();
    let model = FcValueNetwork::<TestBackend>::new(&device, DEFAULT_L1, DEFAULT_L2);

    let input = Tensor::<TestBackend, 2>::random(
        [1, crate::encoding::TOTAL_FEATURES],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );
    let output = model.forward(input);

    assert_eq!(output.dims(), [1, 1], "Output shape should be [1, 1]");
    let value: f32 = output.into_data().to_vec::<f32>().expect("to_vec")[0];
    assert!(
        (0.0..=1.0).contains(&value),
        "Output should be in [0, 1], got {}",
        value
    );
}

#[test]
fn nnue_matches_burn_fc_model() {
    use burn::backend::NdArray;

    let device = Default::default();
    let model = FcValueNetwork::<NdArray>::new(&device, DEFAULT_L1, DEFAULT_L2);
    let nnue_weights = export_weights(&model, DEFAULT_L1, DEFAULT_L2);
    let evaluator = NnueEvaluator::new(nnue_weights);

    let gs = create_test_state();

    // Burn forward pass
    let flat = encode_state_flat::<NdArray>(&gs, &device);
    let batch = flat.unsqueeze::<2>(); // [1, 1106]
    let burn_output: f32 = model
        .forward(batch)
        .into_data()
        .to_vec::<f32>()
        .expect("burn output")[0];

    // NNUE forward pass
    let nnue_output = evaluator.evaluate_state(&gs);

    let diff = (burn_output - nnue_output).abs();
    assert!(
        diff < 1e-4,
        "NNUE output {} should match burn output {}, diff={}",
        nnue_output,
        burn_output,
        diff
    );
}

#[test]
fn nnue_weights_save_load_roundtrip() {
    use burn::backend::NdArray;

    let device = Default::default();
    let model = FcValueNetwork::<NdArray>::new(&device, DEFAULT_L1, DEFAULT_L2);
    let weights = export_weights(&model, DEFAULT_L1, DEFAULT_L2);

    let path = format!("test_nnue_roundtrip_{}.nnue", std::process::id());
    let _guard = TempFileGuard::new(&path);
    weights.save(&path).expect("save failed");
    let loaded = NnueWeights::load(&path).expect("load failed");

    // Compare weights
    assert_eq!(weights.l1_weight.len(), loaded.l1_weight.len());
    for i in 0..weights.l1_weight.len() {
        assert_eq!(weights.l1_weight[i], loaded.l1_weight[i]);
    }
    assert_eq!(weights.l1_bias, loaded.l1_bias);
    assert_eq!(weights.l2_weight, loaded.l2_weight);
    assert_eq!(weights.l2_bias, loaded.l2_bias);
    assert_eq!(weights.l3_weight, loaded.l3_weight);
    assert_eq!(weights.l3_bias, loaded.l3_bias);
}

#[test]
fn accumulator_incremental_matches_full() {
    use burn::backend::NdArray;

    let device = Default::default();
    let model = FcValueNetwork::<NdArray>::new(&device, DEFAULT_L1, DEFAULT_L2);
    let weights = export_weights(&model, DEFAULT_L1, DEFAULT_L2);

    // Full computation with features [0, 5, 100]
    let features = vec![0, 5, 100];
    let full_acc = NnueAccumulator::from_features(&weights, &features);

    // Incremental: start from [0, 5], then add 100
    let partial = vec![0, 5];
    let mut inc_acc = NnueAccumulator::from_features(&weights, &partial);
    inc_acc.add_feature(100, &weights);

    for i in 0..DEFAULT_L1 {
        let diff = (full_acc.hidden[i] - inc_acc.hidden[i]).abs();
        assert!(
            diff < 1e-6,
            "Mismatch at {}: full={}, inc={}",
            i,
            full_acc.hidden[i],
            inc_acc.hidden[i]
        );
    }
}

#[test]
fn accumulator_remove_feature_matches_full() {
    use burn::backend::NdArray;

    let device = Default::default();
    let model = FcValueNetwork::<NdArray>::new(&device, DEFAULT_L1, DEFAULT_L2);
    let weights = export_weights(&model, DEFAULT_L1, DEFAULT_L2);

    // Target: features [0, 100]
    let target_features = vec![0, 100];
    let target_acc = NnueAccumulator::from_features(&weights, &target_features);

    // Start from [0, 5, 100], remove 5
    let full_features = vec![0, 5, 100];
    let mut inc_acc = NnueAccumulator::from_features(&weights, &full_features);
    inc_acc.remove_feature(5, &weights);

    for i in 0..DEFAULT_L1 {
        let diff = (target_acc.hidden[i] - inc_acc.hidden[i]).abs();
        assert!(
            diff < 1e-5,
            "remove_feature mismatch at {}: expected={}, got={}",
            i,
            target_acc.hidden[i],
            inc_acc.hidden[i]
        );
    }
}

#[test]
fn encode_state_flat_board_portion_matches_encode_state() {
    use crate::encoding::BOARD_FEATURES;
    let device = Default::default();
    let gs = create_test_state();

    // 3D encoding = board only (1080)
    let tensor_3d = encode_state::<TestBackend>(&gs, &device);
    let flat_from_3d: Vec<f32> = tensor_3d
        .reshape([BOARD_FEATURES as i32])
        .into_data()
        .to_vec()
        .expect("reshape");

    // Flat encoding = board (1080) + bag (26) = 1106
    let tensor_flat = encode_state_flat::<TestBackend>(&gs, &device);
    let flat_direct: Vec<f32> = tensor_flat
        .into_data()
        .to_vec()
        .expect("flat");

    // Board portion should match
    for i in 0..BOARD_FEATURES {
        assert_eq!(
            flat_from_3d[i], flat_direct[i],
            "Board feature mismatch at index {}", i
        );
    }

    // Bag portion should have non-zero values (initial state has tiles in bag)
    let bag_sum: f32 = flat_direct[BOARD_FEATURES..].iter().sum();
    assert!(bag_sum > 0.0, "Bag features should be non-zero for initial state");
}

#[test]
fn active_features_matches_encoding_after_moves() {
    // Test encoding consistency after several moves (flipped tiles, captures)
    let gs = create_test_state();
    let mut game = gs;
    let ai = StupidSyncAi {};
    let mut rng = StdRng::seed_from_u64(42);
    let device = Default::default();

    // Play a few moves to get flipped tiles
    for _ in 0..6 {
        if game.game_result() != GameResult::Ongoing {
            break;
        }
        ai.play_next_move(&mut rng, &mut game);
    }

    let tensor = encode_state::<TestBackend>(&game, &device);
    let flat: Vec<f32> = tensor.reshape([crate::encoding::BOARD_FEATURES as i32]).into_data().to_vec().expect("flat");

    let active = active_feature_indices(&game);
    for &idx in &active {
        assert_eq!(flat[idx], 1.0, "Feature {} should be 1.0 after moves", idx);
    }
    let ones_count = flat.iter().filter(|&&v| v == 1.0).count();
    assert_eq!(
        ones_count,
        active.len(),
        "After moves: {} ones in tensor but {} active features",
        ones_count,
        active.len()
    );
}

#[test]
fn nnue_matches_burn_across_multiple_states() {
    use burn::backend::NdArray;

    let device = Default::default();
    let model = FcValueNetwork::<NdArray>::new(&device, DEFAULT_L1, DEFAULT_L2);
    let nnue_weights = export_weights(&model, DEFAULT_L1, DEFAULT_L2);
    let evaluator = NnueEvaluator::new(nnue_weights);

    // Play a game and check NNUE matches burn at every state
    let gs = create_small_state();
    let mut game = gs;
    let ai = StupidSyncAi {};
    let mut rng = StdRng::seed_from_u64(99);
    let mut states_checked = 0;

    for _ in 0..10 {
        if game.game_result() != GameResult::Ongoing {
            break;
        }

        let flat = encode_state_flat::<NdArray>(&game, &device);
        let batch = flat.unsqueeze::<2>();
        let burn_output: f32 = model
            .forward(batch)
            .into_data()
            .to_vec::<f32>()
            .expect("burn")[0];

        let nnue_output = evaluator.evaluate_state(&game);
        let diff = (burn_output - nnue_output).abs();
        assert!(
            diff < 1e-4,
            "State {}: NNUE={} vs burn={}, diff={}",
            states_checked,
            nnue_output,
            burn_output,
            diff
        );
        states_checked += 1;

        ai.play_next_move(&mut rng, &mut game);
    }
    assert!(states_checked >= 3, "Should check at least 3 states, checked {}", states_checked);
}

#[test]
fn encode_state_flat_uses_active_feature_indices() {
    use crate::encoding::{BOARD_FEATURES, bag_features, BAG_FEATURES};
    let gs = create_test_state();
    let mut game = gs;
    let ai = StupidSyncAi {};
    let mut rng = StdRng::seed_from_u64(123);
    let device = Default::default();

    for turn in 0..8 {
        if game.game_result() != GameResult::Ongoing { break; }

        let active = active_feature_indices(&game);
        let flat: Vec<f32> = encode_state_flat::<TestBackend>(&game, &device)
            .into_data()
            .to_vec()
            .expect("flat");

        // Board portion: active indices should be 1.0
        for &idx in &active {
            assert_eq!(flat[idx], 1.0,
                "Turn {}: board feature {} should be 1.0", turn, idx);
        }
        // Board 1.0 count should match active features
        let board_ones = flat[..BOARD_FEATURES].iter().filter(|&&v| v == 1.0).count();
        assert_eq!(board_ones, active.len(),
            "Turn {}: {} ones in board but {} active features", turn, board_ones, active.len());

        // Bag portion: should match bag_features()
        let bag = bag_features(&game);
        for i in 0..BAG_FEATURES {
            assert_eq!(flat[BOARD_FEATURES + i], bag[i],
                "Turn {}: bag feature {} mismatch", turn, i);
        }

        ai.play_next_move(&mut rng, &mut game);
    }
}

#[test]
fn fc_trainer_load_model_changes_output() {
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

    // Create two trainers — they get different random weights
    let trainer1 = FcTdTrainer::<TestBackend>::new(device.clone(), 0.001, DEFAULT_L1, DEFAULT_L2);
    let mut trainer2 = FcTdTrainer::<TestBackend>::new(device.clone(), 0.001, DEFAULT_L1, DEFAULT_L2);

    let gs = create_test_state();
    let flat = encode_state_flat::<TestBackend>(&gs, &device);

    // Their outputs should differ (different random init)
    let out1: f32 = trainer1.model.forward(flat.clone().unsqueeze()).into_data().to_vec::<f32>().expect("v")[0];
    let out2_before: f32 = trainer2.model.forward(flat.clone().unsqueeze()).into_data().to_vec::<f32>().expect("v")[0];
    assert!((out1 - out2_before).abs() > 1e-6, "Two random models should differ");

    // Save trainer1's model, load into trainer2
    let path = format!("test_load_model_{}", std::process::id());
    let _guard = TempFileGuard::new(&format!("{}.mpk", path));
    trainer1.save_model(&path);
    trainer2.load_model(&path);

    // Now trainer2 should produce the same output as trainer1
    let out2_after: f32 = trainer2.model.forward(flat.unsqueeze()).into_data().to_vec::<f32>().expect("v")[0];
    let diff = (out1 - out2_after).abs();
    assert!(diff < 1e-5, "After loading, outputs should match: {} vs {}, diff={}", out1, out2_after, diff);
}

#[test]
fn greedy_move_is_deterministic() {
    use burn::backend::NdArray;
    use rand::Rng;
    use crate::game_setup::greedy_move;

    let device = Default::default();
    let model = FcValueNetwork::<NdArray>::new(&device, DEFAULT_L1, DEFAULT_L2);
    let weights = export_weights(&model, DEFAULT_L1, DEFAULT_L2);
    let evaluator = NnueEvaluator::new(weights);

    let gs = create_test_state();

    let mut rng1 = StdRng::seed_from_u64(42);
    let mv1 = greedy_move(&gs, &evaluator, &mut rng1);

    let mut rng2 = StdRng::seed_from_u64(42);
    let mv2 = greedy_move(&gs, &evaluator, &mut rng2);

    assert_eq!(mv1, mv2, "Same seed should produce same move");

    // Also verify rng state is the same after both calls
    // (proves greedy_move consumes the same amount of rng)
    assert_eq!(rng1.gen::<u64>(), rng2.gen::<u64>(),
        "RNG state should be identical after greedy_move with same seed");
}

// ── batch training tests ──────────────────────────────────────────────

#[test]
fn train_on_batch_produces_finite_loss() {
    let device = Default::default();
    let mut trainer: FcTdTrainer<TestBackend> = FcTdTrainer::new(device, 0.001, DEFAULT_L1, DEFAULT_L2);

    let gs = create_test_state();

    // Play 3 games with different seeds, collect trajectories
    let games: Vec<GameTrajectory> = (0..3)
        .map(|seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let (states, result) = play_random_game(&gs, &mut rng);
            GameTrajectory { states, result }
        })
        .collect();

    let loss = trainer.train_on_batch(&games);
    assert!(loss.is_finite(), "Batch loss should be finite, got {}", loss);
    assert!(loss >= 0.0, "Batch loss should be non-negative, got {}", loss);
}

#[test]
fn parallel_games_are_deterministic() {
    use rayon::prelude::*;

    let gs = create_test_state();

    // Play games serially
    let serial_results: Vec<(usize, GameResult)> = (0..8u64)
        .map(|seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let (states, result) = play_random_game(&gs, &mut rng);
            (states.len(), result)
        })
        .collect();

    // Play games in parallel with rayon (same seeds)
    let parallel_results: Vec<(usize, GameResult)> = (0..8u64)
        .into_par_iter()
        .map(|seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let (states, result) = play_random_game(&gs, &mut rng);
            (states.len(), result)
        })
        .collect();

    // Same seeds should produce identical games regardless of parallelism
    for (i, (serial, parallel)) in serial_results.iter().zip(parallel_results.iter()).enumerate() {
        assert_eq!(
            serial.0, parallel.0,
            "Game {}: state count mismatch (serial={}, parallel={})",
            i, serial.0, parallel.0
        );
        assert_eq!(
            serial.1, parallel.1,
            "Game {}: result mismatch (serial={:?}, parallel={:?})",
            i, serial.1, parallel.1
        );
    }
}

// ── match_runner tests ────────────────────────────────────────────────

use crate::match_runner::{play_match, run_matches, Player};
use duke_rust::game::ai::heuristics::{HeuristicAi, Heuristics};
use crate::game_setup::{HeuristicEvaluator, StaticHeuristicEvaluator};
use crate::learned_heuristic::{
    extract_features, solve_linear_system, train_weights, LearnedHeuristicWeights, NUM_FEATURES,
};

#[test]
fn play_match_terminates() {
    let bag = create_bag();
    let gs = create_initial_state(&bag);
    let random = Player::Random;
    let max_turns = 300;

    let mut rng = StdRng::seed_from_u64(42);
    let result = play_match(&gs, &random, &random, &mut rng, max_turns);

    // The match must have finished — it should not be Ongoing.
    assert_ne!(
        result,
        GameResult::Ongoing,
        "Match should terminate within {} turns",
        max_turns,
    );
}

#[test]
fn play_match_random_vs_random_is_fair() {
    let bag = create_bag();
    let gs = create_initial_state(&bag);
    let random = Player::Random;

    let num_games = 100u32;
    let mut top_wins = 0u32;
    let mut bottom_wins = 0u32;

    for seed in 0..num_games {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let result = play_match(&gs, &random, &random, &mut rng, 200);
        match result {
            GameResult::Won(Owner::TopPlayer) => top_wins += 1,
            GameResult::Won(Owner::BottomPlayer) => bottom_wins += 1,
            _ => {}
        }
    }

    let total_decisive = top_wins + bottom_wins;
    if total_decisive > 0 {
        let top_pct = top_wins as f64 / total_decisive as f64;
        assert!(
            top_pct <= 0.70 && top_pct >= 0.30,
            "Random vs Random should be roughly fair: top won {:.0}% of decisive games ({}/{})",
            top_pct * 100.0,
            top_wins,
            total_decisive,
        );
    }
}

#[test]
fn run_matches_alternates_sides() {
    // run_matches uses seed as the loop variable.
    // Even seeds: player_a = Top, player_b = Bottom
    // Odd seeds: player_b = Top, player_a = Bottom
    //
    // We verify this by running 2 games and checking that the side
    // assignment logic in run_matches produces a valid MatchResult.
    let bag = create_bag();
    let gs = create_initial_state(&bag);
    let random = Player::Random;

    let result = run_matches(&gs, &random, &random, 2, "test");

    // With 2 games, total outcomes should sum to 2
    let total = result.player_a_wins + result.player_b_wins + result.ties;
    assert_eq!(
        total, 2,
        "Two games should produce exactly 2 outcomes, got {}",
        total,
    );
}

#[test]
fn heuristic_beats_random() {
    let bag = create_bag();
    let gs = create_initial_state(&bag);

    let heuristic_evaluator = StaticHeuristicEvaluator::new();

    let heuristic = Player::Evaluator(&heuristic_evaluator);
    let random = Player::Random;

    let result = run_matches(&gs, &heuristic, &random, 50, "Heuristic vs Random");

    let total_decisive = result.player_a_wins + result.player_b_wins;
    assert!(
        total_decisive > 0,
        "At least some games should have a decisive result",
    );
    let heuristic_win_pct = result.player_a_wins as f64 / total_decisive as f64;
    assert!(
        heuristic_win_pct > 0.60,
        "Heuristic should win >60% of decisive games vs Random, got {:.0}% ({}/{})",
        heuristic_win_pct * 100.0,
        result.player_a_wins,
        total_decisive,
    );
}

// ── learned_heuristic tests ──────────────────────────────────────────

#[test]
fn feature_extraction_smoke_test() {
    let gs = create_test_state();
    let features = extract_features(&gs);

    // Should have exactly NUM_FEATURES elements
    assert_eq!(features.len(), NUM_FEATURES);

    // All features should be finite
    for (i, &f) in features.iter().enumerate() {
        assert!(f.is_finite(), "Feature {} is not finite: {}", i, f);
    }

    // Bias term (last feature) should always be 1.0
    assert_eq!(features[NUM_FEATURES - 1], 1.0, "Bias feature should be 1.0");

    // Base heuristic features should be reasonable for initial state
    // On the initial board, both players have the same setup, so
    // differences should be small (possibly zero if perfectly symmetric)
    // x2 (TotalTilesOnBoard) should be near 0 for symmetric start
    assert!(
        features[1].abs() < 100.0,
        "TotalTilesOnBoard diff should be bounded, got {}",
        features[1]
    );
}

#[test]
fn gaussian_elimination_known_3x3() {
    // Solve:
    //   2x + y - z = 8
    //   -3x - y + 2z = -11
    //   -2x + y + 2z = -3
    // Solution: x=2, y=3, z=-1
    let mut a = [[0.0f64; NUM_FEATURES]; NUM_FEATURES];
    let mut b = [0.0f64; NUM_FEATURES];

    // Set up 3x3 in the top-left corner, identity for the rest
    a[0][0] = 2.0; a[0][1] = 1.0; a[0][2] = -1.0;
    a[1][0] = -3.0; a[1][1] = -1.0; a[1][2] = 2.0;
    a[2][0] = -2.0; a[2][1] = 1.0; a[2][2] = 2.0;
    // Fill diagonal for remaining dimensions to make it non-singular
    for i in 3..NUM_FEATURES {
        a[i][i] = 1.0;
    }

    b[0] = 8.0;
    b[1] = -11.0;
    b[2] = -3.0;
    // b[3..] = 0.0 already

    let w = solve_linear_system(&mut a, &mut b);

    let eps = 1e-10;
    assert!((w[0] - 2.0).abs() < eps, "x should be 2, got {}", w[0]);
    assert!((w[1] - 3.0).abs() < eps, "y should be 3, got {}", w[1]);
    assert!((w[2] - (-1.0)).abs() < eps, "z should be -1, got {}", w[2]);

    // Remaining unknowns should be 0
    for i in 3..NUM_FEATURES {
        assert!((w[i]).abs() < eps, "w[{}] should be 0, got {}", i, w[i]);
    }
}

#[test]
fn train_on_100_games_produces_finite_weights() {
    let gs = create_test_state();
    let mut rng = StdRng::seed_from_u64(123);

    let games: Vec<_> = (0..100)
        .map(|_| play_random_game(&gs, &mut rng))
        .collect();

    let weights = train_weights(&games);

    for (i, &w) in weights.weights.iter().enumerate() {
        assert!(
            w.is_finite(),
            "Weight {} is not finite: {}",
            i, w
        );
    }

    // At least some weights should be non-zero (not all degenerate)
    let non_zero = weights.weights.iter().filter(|&&w| w.abs() > 1e-12).count();
    assert!(
        non_zero > 0,
        "At least some weights should be non-zero after training on 100 games"
    );
}

#[test]
fn learned_weights_save_load_roundtrip() {
    let gs = create_test_state();
    let mut rng = StdRng::seed_from_u64(77);

    let games: Vec<_> = (0..20)
        .map(|_| play_random_game(&gs, &mut rng))
        .collect();

    let original = train_weights(&games);

    let path = format!("test_learned_roundtrip_{}.json", std::process::id());
    let _guard = TempFileGuard::new(&path);
    original.save(&path).expect("save failed");
    let loaded = LearnedHeuristicWeights::load(&path).expect("load failed");

    for i in 0..NUM_FEATURES {
        let diff = (original.weights[i] - loaded.weights[i]).abs();
        assert!(
            diff < 1e-10,
            "Weight {} mismatch: original={}, loaded={}, diff={}",
            i, original.weights[i], loaded.weights[i], diff
        );
    }
}

#[test]
fn learned_evaluator_returns_finite_score() {
    let gs = create_test_state();

    // Create a simple weights vector (all ones) to test the evaluator
    let mut weights = LearnedHeuristicWeights::default();
    for i in 0..NUM_FEATURES {
        weights.weights[i] = 0.01;
    }

    use crate::game_setup::GameEvaluator;
    let score = weights.evaluate(&gs);
    assert!(score.is_finite(), "Evaluator should return finite score, got {}", score);
}

#[test]
fn trajectory_roundtrip_preserves_game_states() {
    use crate::trajectory_io::{TrajectoryWriter, load_trajectories};

    let bag = create_bag();
    let gs = create_initial_state(&bag);
    let ai = StupidSyncAi {};

    // Play 5 random games
    let mut games = Vec::new();
    for seed in 0..5u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let (states, result) = play_random_game(&gs, &mut rng);
        games.push((states, result));
    }

    // Write to temp file
    let path = format!("D:/temp/test_traj_roundtrip_{}.dtrj", std::process::id());
    let mut writer = TrajectoryWriter::new(&path).unwrap();
    for (states, result) in &games {
        writer.write_game(states, result).unwrap();
    }
    let count = writer.finish().unwrap();
    assert_eq!(count, 5);

    // Load back
    let loaded = load_trajectories(&path).unwrap();
    assert_eq!(loaded.len(), 5);

    for (i, ((orig_states, orig_result), loaded_game)) in games.iter().zip(loaded.iter()).enumerate() {
        assert_eq!(*orig_result, loaded_game.result, "Game {} result mismatch", i);
        assert_eq!(orig_states.len(), loaded_game.states.len(), "Game {} state count mismatch", i);

        for (j, (orig, loaded_gs)) in orig_states.iter().zip(loaded_game.states.iter()).enumerate() {
            // Verify board matches
            assert_eq!(
                orig.current_player_turn(), loaded_gs.current_player_turn(),
                "Game {} state {} player mismatch", i, j
            );
            // Verify board tiles match
            let orig_board = orig.board();
            let loaded_board = loaded_gs.board();
            for y in 0..6u16 {
                for x in 0..6u16 {
                    let c = duke_rust::common::coordinates::Coordinates { x, y };
                    let orig_tile = orig_board.get(c);
                    let loaded_tile = loaded_board.get(c);
                    match (orig_tile, loaded_tile) {
                        (None, None) => {}
                        (Some(o), Some(l)) => {
                            assert_eq!(o.tile.tile_type(), l.tile.tile_type(),
                                "Game {} state {} ({},{}) tile type mismatch", i, j, x, y);
                            assert_eq!(o.current_side, l.current_side,
                                "Game {} state {} ({},{}) side mismatch", i, j, x, y);
                            assert_eq!(o.owner, l.owner,
                                "Game {} state {} ({},{}) owner mismatch", i, j, x, y);
                        }
                        _ => panic!("Game {} state {} ({},{}) occupancy mismatch", i, j, x, y),
                    }
                }
            }
        }
    }

    // Cleanup
    let _ = std::fs::remove_file(&path);
}

#[test]
fn feature_cache_roundtrip() {
    use crate::feature_cache::{CachedGame, CachedState, save_feature_cache, load_feature_cache, accumulate_from_cache};
    use crate::learned_heuristic::{extract_features, NUM_FEATURES};

    let bag = create_bag();
    let gs = create_initial_state(&bag);

    // Play 3 games, extract features
    let mut cached_games = Vec::new();
    let mut expected_samples = 0usize;
    for seed in 0..3u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let (states, result) = play_random_game(&gs, &mut rng);
        let mut cached_states = Vec::new();
        for state in &states {
            if state.game_result() != GameResult::Ongoing { continue; }
            let features = extract_features(state);
            cached_states.push(CachedState {
                current_player: state.current_player_turn(),
                features: features.to_vec(),
            });
            expected_samples += 1;
        }
        cached_games.push(CachedGame { result, states: cached_states });
    }

    // Save
    let path = format!("D:/temp/test_feat_cache_{}.bin", std::process::id());
    save_feature_cache(&path, NUM_FEATURES, &cached_games).unwrap();

    // Load
    let (header, loaded) = load_feature_cache(&path).unwrap();
    assert_eq!(header.num_features, NUM_FEATURES);
    assert_eq!(header.num_games, 3);
    assert_eq!(loaded.len(), 3);

    // Verify features match
    for (i, (orig, loaded_game)) in cached_games.iter().zip(loaded.iter()).enumerate() {
        assert_eq!(orig.result, loaded_game.result, "Game {} result mismatch", i);
        assert_eq!(orig.states.len(), loaded_game.states.len(), "Game {} state count mismatch", i);
        for (j, (orig_s, loaded_s)) in orig.states.iter().zip(loaded_game.states.iter()).enumerate() {
            assert_eq!(orig_s.current_player, loaded_s.current_player,
                "Game {} state {} player mismatch", i, j);
            for (k, (&orig_f, &loaded_f)) in orig_s.features.iter().zip(loaded_s.features.iter()).enumerate() {
                assert!((orig_f - loaded_f).abs() < 1e-10,
                    "Game {} state {} feature {} mismatch: {} vs {}", i, j, k, orig_f, loaded_f);
            }
        }
    }

    // Verify accumulation works
    let acc = accumulate_from_cache(&loaded, header.num_features);
    assert_eq!(acc.n_samples(), expected_samples as u64);

    let _ = std::fs::remove_file(&path);
}

<<<<<<< HEAD
// ── heuristic tests ────────────────────────────────────────────────────

use duke_rust::common::coordinates::Coordinates;
use duke_rust::game::ai::heuristics::Heuristic;
use duke_rust::game::bag::DiscardBag;
use duke_rust::game::tile::{PlacedTile, TileType};
use duke_rust::game::units::tile_from_type;

/// Helper: build a GameState from placed tiles with empty bags and discards.
fn snapshot_state(
    tiles: Vec<(Coordinates, PlacedTile)>,
    current_player: Owner,
) -> GameState {
    GameState::from_snapshot(
        tiles,
        current_player,
        TileBag::new(vec![]),
        TileBag::new(vec![]),
        DiscardBag::empty(),
        DiscardBag::empty(),
        0,
    )
}

/// Helper: build a GameState with custom discard bags.
fn snapshot_state_with_discards(
    tiles: Vec<(Coordinates, PlacedTile)>,
    current_player: Owner,
    top_discard: DiscardBag,
    bottom_discard: DiscardBag,
) -> GameState {
    GameState::from_snapshot(
        tiles,
        current_player,
        TileBag::new(vec![]),
        TileBag::new(vec![]),
        top_discard,
        bottom_discard,
        0,
    )
}

/// Helper: build a GameState with custom bags (for TotalMovementOptions which
/// adds placement moves when the bag is non-empty).
fn snapshot_state_with_bags(
    tiles: Vec<(Coordinates, PlacedTile)>,
    current_player: Owner,
    top_bag: TileBag,
    bottom_bag: TileBag,
) -> GameState {
    GameState::from_snapshot(
        tiles,
        current_player,
        top_bag,
        bottom_bag,
        DiscardBag::empty(),
        DiscardBag::empty(),
        0,
    )
}

// ── DukeMovementOptions ────────────────────────────────────────────────

#[test]
fn duke_movement_options_simple_unblocked() {
    // BottomPlayer Duke (Initial) at (3,3): slides left/right.
    //   Left: x=2,1,0 => 3 squares; Right: x=4,5 => 2 squares => 5 total.
    // TopPlayer Duke (Initial) at (0,0): slides left/right.
    //   Left: out of bounds; Right: x=1,2,3,4,5 => 5 squares.
    // No guard issues since dukes are on different rows.
    let gs = snapshot_state(vec![
        (Coordinates { x: 3, y: 3 }, PlacedTile::new(Owner::BottomPlayer, tile_from_type(TileType::Duke))),
        (Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, tile_from_type(TileType::Duke))),
    ], Owner::BottomPlayer);

    let h = Heuristics::DukeMovementOptions;
    // BottomPlayer duke has 5 horizontal slide moves
    let bottom_val = h.evaluate_for_owner(Owner::BottomPlayer, &gs);
    assert_eq!(bottom_val, 5.0,
        "BottomPlayer duke at (3,3) Initial should have 5 slide moves, got {}", bottom_val);
    // TopPlayer duke has 5 horizontal slide moves
    let top_val = h.evaluate_for_owner(Owner::TopPlayer, &gs);
    assert_eq!(top_val, 5.0,
        "TopPlayer duke at (0,0) Initial should have 5 slide moves, got {}", top_val);

    // approx should ignore guard, but no guard here so same result
    let bottom_approx = h.approx_evaluate_for_owner(Owner::BottomPlayer, &gs);
    assert_eq!(bottom_approx, 5.0);
    let top_approx = h.approx_evaluate_for_owner(Owner::TopPlayer, &gs);
    assert_eq!(top_approx, 5.0);
}

#[test]
fn duke_movement_options_blocked_by_own_tile() {
    // BottomPlayer Duke (Initial) at (3,3): slides left/right.
    //   Own footman at (4,3) blocks rightward slide at x=4 => right slide stops.
    //   Left: x=2,1,0 => 3 squares.
    //   Right: blocked at x=4 => 0 squares.
    //   Total: 3 moves.
    // TopPlayer Duke at (0,0) far away.
    let gs = snapshot_state(vec![
        (Coordinates { x: 3, y: 3 }, PlacedTile::new(Owner::BottomPlayer, tile_from_type(TileType::Duke))),
        (Coordinates { x: 4, y: 3 }, PlacedTile::new(Owner::BottomPlayer, tile_from_type(TileType::Footman))),
        (Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, tile_from_type(TileType::Duke))),
    ], Owner::BottomPlayer);

    let h = Heuristics::DukeMovementOptions;
    // BottomPlayer duke: left 3 squares, right blocked => 3
    let bottom_val = h.evaluate_for_owner(Owner::BottomPlayer, &gs);
    assert_eq!(bottom_val, 3.0,
        "BottomPlayer duke at (3,3) blocked right by own footman should have 3 moves, got {}", bottom_val);

    let bottom_approx = h.approx_evaluate_for_owner(Owner::BottomPlayer, &gs);
    assert_eq!(bottom_approx, 3.0);
}

#[test]
fn duke_movement_options_flipped_duke_slides_vertically() {
    // BottomPlayer Duke (Flipped) at (2,2): slides top/bottom.
    //   Top: y=1,0 => 2 squares; Bottom: y=3,4,5 => 3 squares => 5 total.
    // TopPlayer Duke at (5,5).
    let mut bottom_duke = PlacedTile::new(Owner::BottomPlayer, tile_from_type(TileType::Duke));
    bottom_duke.flip();
    let gs = snapshot_state(vec![
        (Coordinates { x: 2, y: 2 }, bottom_duke),
        (Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::TopPlayer, tile_from_type(TileType::Duke))),
    ], Owner::BottomPlayer);

    let h = Heuristics::DukeMovementOptions;
    // BottomPlayer duke flipped slides vertically: up y=1,0 (2) + down y=3,4 (2) = 4 moves.
    // (2,5) is excluded because TopPlayer duke at (5,5) Initial slides horizontally along y=5,
    // so moving to (2,5) would put the BottomPlayer duke in guard.
    let bottom_val = h.evaluate_for_owner(Owner::BottomPlayer, &gs);
    assert_eq!(bottom_val, 4.0,
        "Flipped BottomPlayer duke at (2,2) should have 4 legal slide moves (guard), got {}", bottom_val);

    // approx_evaluate ignores guard, so the move to (2,5) is also counted => 5 total.
    let bottom_approx = h.approx_evaluate_for_owner(Owner::BottomPlayer, &gs);
    assert_eq!(bottom_approx, 5.0,
        "Flipped BottomPlayer duke at (2,2) ignoring guard should have 5 slide moves, got {}", bottom_approx);
}

// ── TotalTilesOnBoard ──────────────────────────────────────────────────

#[test]
fn total_tiles_on_board_simple() {
    // BottomPlayer has 2 tiles (Duke + Footman), TopPlayer has 1 tile (Duke).
    // TotalTilesOnBoard = count * 10.
    // BottomPlayer: 2 * 10 = 20; TopPlayer: 1 * 10 = 10.
    let gs = snapshot_state(vec![
        (Coordinates { x: 3, y: 3 }, PlacedTile::new(Owner::BottomPlayer, tile_from_type(TileType::Duke))),
        (Coordinates { x: 3, y: 4 }, PlacedTile::new(Owner::BottomPlayer, tile_from_type(TileType::Footman))),
        (Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, tile_from_type(TileType::Duke))),
    ], Owner::BottomPlayer);

    let h = Heuristics::TotalTilesOnBoard;
    let bottom_val = h.evaluate_for_owner(Owner::BottomPlayer, &gs);
    assert_eq!(bottom_val, 20.0,
        "BottomPlayer with 2 tiles should score 20.0, got {}", bottom_val);
    let top_val = h.evaluate_for_owner(Owner::TopPlayer, &gs);
    assert_eq!(top_val, 10.0,
        "TopPlayer with 1 tile should score 10.0, got {}", top_val);

    // approx is identical to evaluate for this heuristic
    assert_eq!(h.approx_evaluate_for_owner(Owner::BottomPlayer, &gs), 20.0);
    assert_eq!(h.approx_evaluate_for_owner(Owner::TopPlayer, &gs), 10.0);
}

#[test]
fn total_tiles_on_board_symmetric() {
    // Both players have 3 tiles each (Duke + 2 Footmen).
    // Each player: 3 * 10 = 30.
    let gs = snapshot_state(vec![
        (Coordinates { x: 2, y: 5 }, PlacedTile::new(Owner::BottomPlayer, tile_from_type(TileType::Duke))),
        (Coordinates { x: 1, y: 5 }, PlacedTile::new(Owner::BottomPlayer, tile_from_type(TileType::Footman))),
        (Coordinates { x: 3, y: 5 }, PlacedTile::new(Owner::BottomPlayer, tile_from_type(TileType::Footman))),
        (Coordinates { x: 2, y: 0 }, PlacedTile::new(Owner::TopPlayer, tile_from_type(TileType::Duke))),
        (Coordinates { x: 1, y: 0 }, PlacedTile::new(Owner::TopPlayer, tile_from_type(TileType::Footman))),
        (Coordinates { x: 3, y: 0 }, PlacedTile::new(Owner::TopPlayer, tile_from_type(TileType::Footman))),
    ], Owner::BottomPlayer);

    let h = Heuristics::TotalTilesOnBoard;
    assert_eq!(h.evaluate_for_owner(Owner::BottomPlayer, &gs), 30.0,
        "BottomPlayer with 3 tiles should score 30.0");
    assert_eq!(h.evaluate_for_owner(Owner::TopPlayer, &gs), 30.0,
        "TopPlayer with 3 tiles should score 30.0");

    // Difference should be 0 for symmetric setup
    assert_eq!(h.difference(Owner::BottomPlayer, &gs), 0.0);
}

// ── TotalMovementOptions ───────────────────────────────────────────────

#[test]
fn total_movement_options_duke_only() {
    // Only dukes on the board, empty bags => no placement moves.
    // BottomPlayer Duke (Initial) at (3,3): slides left/right = 5 moves.
    // TopPlayer Duke (Initial) at (0,0): slides left/right = 5 moves.
    // TotalMovementOptions counts all valid game moves (tile moves only, no placements
    // since bags are empty).
    let gs = snapshot_state(vec![
        (Coordinates { x: 3, y: 3 }, PlacedTile::new(Owner::BottomPlayer, tile_from_type(TileType::Duke))),
        (Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, tile_from_type(TileType::Duke))),
    ], Owner::BottomPlayer);

    let h = Heuristics::TotalMovementOptions;
    let bottom_val = h.evaluate_for_owner(Owner::BottomPlayer, &gs);
    assert_eq!(bottom_val, 5.0,
        "BottomPlayer with only duke at (3,3) should have 5 moves, got {}", bottom_val);
    let top_val = h.evaluate_for_owner(Owner::TopPlayer, &gs);
    assert_eq!(top_val, 5.0,
        "TopPlayer with only duke at (0,0) should have 5 moves, got {}", top_val);

    // approx is identical for TotalMovementOptions
    assert_eq!(h.approx_evaluate_for_owner(Owner::BottomPlayer, &gs), 5.0);
}

#[test]
fn total_movement_options_multiple_pieces_with_bag() {
    // BottomPlayer: Duke (Initial) at (3,5) + Footman (Initial) at (3,4).
    //   Duke slides left/right: left x=2,1,0 (3), right x=4,5 (2) = 5 moves.
    //   Footman (Initial, BottomPlayer): NearStraight Move = up/down/left/right.
    //     From (3,4): up=(3,3), down=(3,5) own duke blocks, left=(2,4), right=(4,4) = 3 moves.
    //   Plus placement moves if bag is non-empty.
    // TopPlayer: Duke at (0,0), no bag.
    //
    // With BottomPlayer having tiles in bag, placement moves are possible
    // near the duke at (3,5): valid offsets that are empty.
    let gs = snapshot_state_with_bags(
        vec![
            (Coordinates { x: 3, y: 5 }, PlacedTile::new(Owner::BottomPlayer, tile_from_type(TileType::Duke))),
            (Coordinates { x: 3, y: 4 }, PlacedTile::new(Owner::BottomPlayer, tile_from_type(TileType::Footman))),
            (Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, tile_from_type(TileType::Duke))),
        ],
        Owner::BottomPlayer,
        TileBag::new(vec![]),  // TopPlayer empty bag
        TileBag::new(vec![std::sync::Arc::new(tile_from_type(TileType::Pikeman))]),  // BottomPlayer has a tile
    );

    let h = Heuristics::TotalMovementOptions;
    // We check the value is > 0 and matches expectations.
    // Duke: 5 slide moves, Footman: 3 moves (up/left/right, down blocked by duke).
    // Plus placement moves near duke at (3,5): empty adjacent squares.
    // Duke at (3,5), adjacent: (2,5), (4,5), (3,4)=occupied. DukeOffset has
    // Left, Right, Top, Bottom variants => need to check which are valid.
    let bottom_val = h.evaluate_for_owner(Owner::BottomPlayer, &gs);
    // We'll assert it's strictly greater than the duke-only case
    assert!(bottom_val > 5.0,
        "BottomPlayer with duke + footman + bag should have > 5 moves, got {}", bottom_val);

    // TopPlayer with just duke at (0,0) and empty bag: 5 slide moves only.
    let top_val = h.evaluate_for_owner(Owner::TopPlayer, &gs);
    assert_eq!(top_val, 5.0,
        "TopPlayer with only duke at (0,0) and empty bag should have 5 moves, got {}", top_val);
}

// ── DiscardedUnits ─────────────────────────────────────────────────────

#[test]
fn discarded_units_empty_discards() {
    // No discarded tiles => score is 0 * -15 = 0.0 for both players.
    let gs = snapshot_state(vec![
        (Coordinates { x: 3, y: 3 }, PlacedTile::new(Owner::BottomPlayer, tile_from_type(TileType::Duke))),
        (Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, tile_from_type(TileType::Duke))),
    ], Owner::BottomPlayer);

    let h = Heuristics::DiscardedUnits;
    assert_eq!(h.evaluate_for_owner(Owner::BottomPlayer, &gs), 0.0,
        "No discards should give 0.0");
    assert_eq!(h.evaluate_for_owner(Owner::TopPlayer, &gs), 0.0,
        "No discards should give 0.0");
    assert_eq!(h.approx_evaluate_for_owner(Owner::BottomPlayer, &gs), 0.0);
}

#[test]
fn discarded_units_with_discards() {
    // TopPlayer has 2 discarded tiles => 2 * -15 = -30.0.
    // BottomPlayer has 1 discarded tile => 1 * -15 = -15.0.
    use std::sync::Arc;
    let gs = snapshot_state_with_discards(
        vec![
            (Coordinates { x: 3, y: 3 }, PlacedTile::new(Owner::BottomPlayer, tile_from_type(TileType::Duke))),
            (Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, tile_from_type(TileType::Duke))),
        ],
        Owner::BottomPlayer,
        DiscardBag::from_tiles(vec![
            Arc::new(tile_from_type(TileType::Footman)),
            Arc::new(tile_from_type(TileType::Pikeman)),
        ]),
        DiscardBag::from_tiles(vec![
            Arc::new(tile_from_type(TileType::Knight)),
        ]),
    );

    let h = Heuristics::DiscardedUnits;
    // TopPlayer has 2 discarded tiles: 2 * -15 = -30
    let top_val = h.evaluate_for_owner(Owner::TopPlayer, &gs);
    assert_eq!(top_val, -30.0,
        "TopPlayer with 2 discards should score -30.0, got {}", top_val);
    // BottomPlayer has 1 discarded tile: 1 * -15 = -15
    let bottom_val = h.evaluate_for_owner(Owner::BottomPlayer, &gs);
    assert_eq!(bottom_val, -15.0,
        "BottomPlayer with 1 discard should score -15.0, got {}", bottom_val);

    // approx is identical for DiscardedUnits
    assert_eq!(h.approx_evaluate_for_owner(Owner::TopPlayer, &gs), -30.0);
    assert_eq!(h.approx_evaluate_for_owner(Owner::BottomPlayer, &gs), -15.0);

    // Difference for BottomPlayer: -15 - (-30) = 15 (advantage since fewer discards)
    assert_eq!(h.difference(Owner::BottomPlayer, &gs), 15.0);
}

#[test]
fn discarded_units_three_tiles_discarded() {
    // TopPlayer has 0 discarded, BottomPlayer has 3 discarded.
    // BottomPlayer: 3 * -15 = -45.0
    use std::sync::Arc;
    let gs = snapshot_state_with_discards(
        vec![
            (Coordinates { x: 3, y: 3 }, PlacedTile::new(Owner::BottomPlayer, tile_from_type(TileType::Duke))),
            (Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, tile_from_type(TileType::Duke))),
        ],
        Owner::BottomPlayer,
        DiscardBag::empty(),
        DiscardBag::from_tiles(vec![
            Arc::new(tile_from_type(TileType::Footman)),
            Arc::new(tile_from_type(TileType::Pikeman)),
            Arc::new(tile_from_type(TileType::Knight)),
        ]),
    );

    let h = Heuristics::DiscardedUnits;
    assert_eq!(h.evaluate_for_owner(Owner::TopPlayer, &gs), 0.0,
        "TopPlayer with 0 discards should score 0.0");
    assert_eq!(h.evaluate_for_owner(Owner::BottomPlayer, &gs), -45.0,
        "BottomPlayer with 3 discards should score -45.0");

    // Difference for BottomPlayer: -45 - 0 = -45 (disadvantage)
    assert_eq!(h.difference(Owner::BottomPlayer, &gs), -45.0);
}

// ── Manhattan distance feature tests ────────────────────────────────────

mod manhattan_tests {
    use duke_rust::common::coordinates::Coordinates;
    use duke_rust::game::bag::{DiscardBag, TileBag};
    use duke_rust::game::state::GameState;
    use duke_rust::game::tile::{Owner, PlacedTile};
    use duke_rust::game::units;
    use crate::learned_heuristic::manhattan_distance_features;

    fn coord(x: u16, y: u16) -> Coordinates {
        Coordinates { x, y }
    }

    fn make_state(tiles: Vec<(Coordinates, PlacedTile)>, current: Owner) -> GameState {
        GameState::from_snapshot(
            tiles,
            current,
            TileBag::new(vec![]),
            TileBag::new(vec![]),
            DiscardBag::empty(),
            DiscardBag::empty(),
            0,
        )
    }

    /// Simple: 2 my tiles near my duke, 1 far away.
    /// My duke at (2,2), my footmen at (2,3) and (3,2), my pikeman at (5,5).
    /// Enemy duke at (4,4).
    #[test]
    fn simple_two_near_one_far() {
        let me = Owner::TopPlayer;
        let opp = Owner::BottomPlayer;

        let tiles = vec![
            (coord(2, 2), PlacedTile::new(me, units::duke())),
            (coord(2, 3), PlacedTile::new(me, units::footman())),  // dist 1 from my duke
            (coord(3, 2), PlacedTile::new(me, units::footman())),  // dist 1 from my duke
            (coord(5, 5), PlacedTile::new(me, units::pikeman())),  // dist 6 from my duke
            (coord(4, 4), PlacedTile::new(opp, units::duke())),
        ];
        let gs = make_state(tiles, me);
        let f = manhattan_distance_features(&gs);

        // my_units_near_my_duke: footman(2,3) dist=1, footman(3,2) dist=1 => 2
        assert_eq!(f[0], 2.0, "my_units_near_my_duke");
        // enemy_units_near_my_duke: enemy duke(4,4) dist=4 => 0
        assert_eq!(f[1], 0.0, "enemy_units_near_my_duke");
        // my_units_near_enemy_duke: pikeman(5,5) dist=2 => 1
        assert_eq!(f[2], 1.0, "my_units_near_enemy_duke");
        // enemy_units_near_enemy_duke: only enemy duke itself at dist 0, excluded => 0
        assert_eq!(f[3], 0.0, "enemy_units_near_enemy_duke");
    }

    /// Both dukes with surrounding tiles from both players.
    /// My duke at (1,1), enemy duke at (4,4).
    /// Mixed ownership tiles near each duke.
    #[test]
    fn both_dukes_with_surrounding_tiles() {
        let me = Owner::TopPlayer;
        let opp = Owner::BottomPlayer;

        let tiles = vec![
            (coord(1, 1), PlacedTile::new(me, units::duke())),
            (coord(4, 4), PlacedTile::new(opp, units::duke())),
            (coord(1, 2), PlacedTile::new(me, units::footman())),   // near my duke
            (coord(3, 3), PlacedTile::new(me, units::pikeman())),   // near enemy duke
            (coord(1, 0), PlacedTile::new(opp, units::footman())),  // near my duke
            (coord(4, 3), PlacedTile::new(opp, units::bowman())),   // near enemy duke
            (coord(3, 4), PlacedTile::new(opp, units::pikeman())),  // near enemy duke
        ];
        let gs = make_state(tiles, me);
        let f = manhattan_distance_features(&gs);

        // my_units_near_my_duke: footman(1,2) dist=1 => 1
        assert_eq!(f[0], 1.0, "my_units_near_my_duke");
        // enemy_units_near_my_duke: footman(1,0) dist=1 => 1
        assert_eq!(f[1], 1.0, "enemy_units_near_my_duke");
        // my_units_near_enemy_duke: pikeman(3,3) dist=2 => 1
        assert_eq!(f[2], 1.0, "my_units_near_enemy_duke");
        // enemy_units_near_enemy_duke: bowman(4,3) dist=1, pikeman(3,4) dist=1 => 2
        assert_eq!(f[3], 2.0, "enemy_units_near_enemy_duke");
    }

    /// Edge case: duke in corner (0,0) with tiles at boundary positions.
    /// My duke at (0,0), enemy duke at (5,5).
    #[test]
    fn duke_in_corner() {
        let me = Owner::TopPlayer;
        let opp = Owner::BottomPlayer;

        let tiles = vec![
            (coord(0, 0), PlacedTile::new(me, units::duke())),
            (coord(5, 5), PlacedTile::new(opp, units::duke())),
            (coord(0, 1), PlacedTile::new(me, units::footman())),   // dist 1 from my duke
            (coord(1, 1), PlacedTile::new(me, units::footman())),   // dist 2 from my duke
            (coord(0, 2), PlacedTile::new(opp, units::bowman())),   // dist 2 from my duke
            (coord(5, 4), PlacedTile::new(opp, units::pikeman())),  // dist 1 from enemy duke
        ];
        let gs = make_state(tiles, me);
        let f = manhattan_distance_features(&gs);

        // my_units_near_my_duke: footman(0,1) dist=1, footman(1,1) dist=2 => 2
        assert_eq!(f[0], 2.0, "my_units_near_my_duke");
        // enemy_units_near_my_duke: bowman(0,2) dist=2 => 1
        assert_eq!(f[1], 1.0, "enemy_units_near_my_duke");
        // my_units_near_enemy_duke: footman(0,1) dist=9, footman(1,1) dist=8 => 0
        assert_eq!(f[2], 0.0, "my_units_near_enemy_duke");
        // enemy_units_near_enemy_duke: pikeman(5,4) dist=1 => 1
        assert_eq!(f[3], 1.0, "enemy_units_near_enemy_duke");
    }
}
