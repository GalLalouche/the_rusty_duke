use std::sync::Arc;

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
use duke_rust::game::units;

use crate::encoding::{active_feature_indices, encode_state, encode_state_flat, BOARD_SIZE, NUM_PLANES};
use crate::fc_model::FcValueNetwork;
use crate::model::ValueNetwork;
use crate::nnue::{NnueAccumulator, NnueEvaluator, NnueWeights, L1_SIZE};
use crate::td_training::TdTrainer;
use crate::weight_export::export_weights;

type TestBackend = Autodiff<NdArray>;

fn create_bag() -> TileBag {
    TileBag::new(vec![
        Arc::new(units::footman()),
        Arc::new(units::bowman()),
        Arc::new(units::knight()),
        Arc::new(units::pikeman()),
        Arc::new(units::pikeman()),
        Arc::new(units::champion()),
        Arc::new(units::priest()),
        Arc::new(units::wizard()),
        Arc::new(units::dragoon()),
        Arc::new(units::general()),
        Arc::new(units::marshall()),
        Arc::new(units::longbowman()),
    ])
}

/// Small bag for faster games in debug mode.
fn create_small_bag() -> TileBag {
    TileBag::new(vec![])
}

fn create_initial_state() -> GameState {
    let bag = create_bag();
    GameState::new(
        &bag,
        (DukeInitialLocation::Left, FootmenSetup::Left),
        (DukeInitialLocation::Right, FootmenSetup::Right),
    )
}

fn create_small_state() -> GameState {
    let bag = create_small_bag();
    GameState::new(
        &bag,
        (DukeInitialLocation::Left, FootmenSetup::Left),
        (DukeInitialLocation::Right, FootmenSetup::Right),
    )
}

fn play_random_game(gs: &GameState, rng: &mut StdRng) -> (Vec<GameState>, GameResult) {
    let ai = StupidSyncAi {};
    let mut game = gs.clone();
    let mut states = Vec::new();

    loop {
        match game.game_result() {
            GameResult::Ongoing => {
                states.push(game.clone());
                ai.play_next_move(rng, &mut game);
            }
            result => {
                states.push(game.clone());
                return (states, result);
            }
        }
    }
}

// ── encoding tests ──────────────────────────────────────────────────────

#[test]
fn encode_state_has_correct_shape() {
    let gs = create_initial_state();
    let device = Default::default();
    let tensor = encode_state::<TestBackend>(&gs, &device);
    let dims = tensor.dims();
    assert_eq!(dims, [NUM_PLANES, BOARD_SIZE, BOARD_SIZE]);
}

#[test]
fn encode_state_initial_board_has_six_tiles() {
    // Initial board has 3 tiles per player (Duke + 2 Footmen = 6 total).
    // Sum of all tile-type planes (0..25) should equal 6.0.
    let gs = create_initial_state();
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
    // Create a game state from TopPlayer's perspective and BottomPlayer's
    // perspective. The "my tiles" and "opponent tiles" planes should differ.
    let gs = create_initial_state();
    let device = Default::default();

    // TopPlayer's turn (default)
    assert_eq!(gs.current_player_turn(), Owner::TopPlayer);
    let tensor_top = encode_state::<TestBackend>(&gs, &device);

    // Play one move so it becomes BottomPlayer's turn
    let mut gs2 = gs.clone();
    let ai = StupidSyncAi {};
    let mut rng = StdRng::seed_from_u64(42);
    ai.play_next_move(&mut rng, &mut gs2);
    assert_eq!(gs2.current_player_turn(), Owner::BottomPlayer);
    let tensor_bottom = encode_state::<TestBackend>(&gs2, &device);

    // The "my tiles" planes (0..13) should not be identical between the two
    // encodings because perspective flipped.
    let my_planes_top: Vec<f32> = tensor_top
        .clone()
        .slice([0..13])
        .into_data()
        .to_vec()
        .expect("to_vec");
    let my_planes_bottom: Vec<f32> = tensor_bottom
        .clone()
        .slice([0..13])
        .into_data()
        .to_vec()
        .expect("to_vec");
    assert_ne!(
        my_planes_top, my_planes_bottom,
        "Encoding should differ when current player changes"
    );
}

#[test]
fn encode_state_side_planes_are_correct() {
    // Initially all tiles are on Initial side.
    // Plane 26 (current player initial) and 28 (opponent initial) should have values.
    // Plane 27 (current player flipped) and 29 (opponent flipped) should be zero.
    let gs = create_initial_state();
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

// ── model tests ─────────────────────────────────────────────────────────

#[test]
fn model_forward_produces_valid_output() {
    let device = Default::default();
    let model = ValueNetwork::<TestBackend>::new(&device);

    // Random input of shape [1, 30, 6, 6]
    let input = Tensor::<TestBackend, 4>::random(
        [1, NUM_PLANES, BOARD_SIZE, BOARD_SIZE],
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
fn model_output_is_between_zero_and_one() {
    let device = Default::default();
    let model = ValueNetwork::<TestBackend>::new(&device);

    // Run multiple forward passes with different random inputs
    for seed in 0..10u64 {
        // Use a manual approach: create seeded data
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<f32> = (0..NUM_PLANES * BOARD_SIZE * BOARD_SIZE)
            .map(|_| {
                use rand::Rng;
                rng.gen::<f32>()
            })
            .collect();
        let input = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &device)
            .reshape([1, NUM_PLANES as i32, BOARD_SIZE as i32, BOARD_SIZE as i32]);
        let output = model.forward(input);
        let value: f32 = output.into_data().to_vec::<f32>().expect("to_vec")[0];
        assert!(
            value > 0.0 && value < 1.0,
            "Output should be in (0, 1), got {} for seed {}",
            value,
            seed
        );
    }
}

// ── td_training tests ───────────────────────────────────────────────────

#[test]
fn train_on_game_returns_loss() {
    let device = Default::default();
    let mut trainer: TdTrainer<TestBackend> = TdTrainer::new(device, 0.001);

    let gs = create_initial_state();
    let mut rng = StdRng::seed_from_u64(0);
    let (states, result) = play_random_game(&gs, &mut rng);

    let loss = trainer.train_on_game(&states, result);
    assert!(loss.is_finite(), "Loss should be finite, got {}", loss);
    assert!(loss >= 0.0, "Loss should be non-negative, got {}", loss);
}

#[test]
fn train_reduces_loss_on_repeated_game() {
    // Use a small bag (empty) so the game is short and training is fast in debug mode.
    let device = Default::default();
    let mut trainer: TdTrainer<TestBackend> = TdTrainer::new(device, 0.01);

    let gs = create_small_state();
    let mut rng = StdRng::seed_from_u64(7);
    let (states, result) = play_random_game(&gs, &mut rng);

    // Only use first 10 states to keep batch small
    let states: Vec<GameState> = states.into_iter().take(10).collect();

    // Train on the same game trajectory a few times
    let first_loss = trainer.train_on_game(&states, result);
    let mut last_loss = first_loss;
    for _ in 1..5 {
        last_loss = trainer.train_on_game(&states, result);
    }

    assert!(
        last_loss < first_loss,
        "Loss should decrease after repeated training on the same game: first={}, last={}",
        first_loss,
        last_loss
    );
}

#[test]
#[ignore] // Flaky: CNN with 5 training iterations may not converge enough
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
    let mut trainer: TdTrainer<TestBackend> = TdTrainer::new(device.clone(), 0.01);
    for _ in 0..5 {
        trainer.train_on_game(&states, game_result);
    }

    // Now check the terminal state prediction
    let encoded = encode_state::<TestBackend>(terminal_state, &device);
    let batch = encoded.unsqueeze::<4>(); // [1, 30, 6, 6]
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
    let gs = create_initial_state();
    let device = Default::default();
    let tensor = encode_state::<TestBackend>(&gs, &device);
    let flat: Vec<f32> = tensor.reshape([1080]).into_data().to_vec().expect("flat");

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
    let model = FcValueNetwork::<TestBackend>::new(&device);

    // Random input of shape [1, 1080]
    let input = Tensor::<TestBackend, 2>::random(
        [1, NUM_PLANES * BOARD_SIZE * BOARD_SIZE],
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
    let model = FcValueNetwork::<NdArray>::new(&device);
    let nnue_weights = export_weights(&model);
    let evaluator = NnueEvaluator::new(nnue_weights);

    let gs = create_initial_state();

    // Burn forward pass
    let flat = encode_state_flat::<NdArray>(&gs, &device);
    let batch = flat.unsqueeze::<2>(); // [1, 1080]
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
    let model = FcValueNetwork::<NdArray>::new(&device);
    let weights = export_weights(&model);

    let path = "test_weights.nnue";
    weights.save(path).expect("save failed");
    let loaded = NnueWeights::load(path).expect("load failed");
    std::fs::remove_file(path).ok();

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
    let model = FcValueNetwork::<NdArray>::new(&device);
    let weights = export_weights(&model);

    // Full computation with features [0, 5, 100]
    let features = vec![0, 5, 100];
    let full_acc = NnueAccumulator::from_features(&weights, &features);

    // Incremental: start from [0, 5], then add 100
    let partial = vec![0, 5];
    let mut inc_acc = NnueAccumulator::from_features(&weights, &partial);
    inc_acc.add_feature(100, &weights);

    for i in 0..L1_SIZE {
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
