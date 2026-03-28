//! Shared game setup used by training, benchmarking, and tests.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use duke_rust::game::ai::player::{AiMove, EvaluatingPlayer};
use duke_rust::game::ai::player::ArtificialPlayer;
use duke_rust::game::ai::stupid_sync_ai::StupidSyncAi;
use duke_rust::game::bag::TileBag;
use duke_rust::game::board_setup::{DukeInitialLocation, FootmenSetup};
use duke_rust::game::state::{GameResult, GameState};
use duke_rust::game::tile::TileType;

use duke_rust::game::ai::heuristics::Heuristic;

use crate::encoding::{active_board_features, bag_features};
use crate::generic_mlp::{GenericMlp, L1Accumulator};
use crate::learned_heuristic::{extract_combined_features, NUM_COMBINED_FEATURES};
use crate::nnue::NnueEvaluator;

/// Safety limit: if a game exceeds this many turns, force a draw.
/// In practice the built-in idle-move draw rule should trigger well before this.
const MAX_TURNS: u32 = 500;

/// Common trait for anything that can evaluate a game state.
/// Higher values = better for the current player.
/// The scale is arbitrary -- only relative ordering matters for move selection.
pub trait GameEvaluator {
    fn evaluate(&self, gs: &GameState) -> f32;

    /// If this evaluator wraps a `GenericMlp` with sparse NNUE-style inputs,
    /// return a reference to the network and whether combined features are
    /// included (true for 1147-input, false for 1106-input models).
    ///
    /// Used by `greedy_move` to enable the incremental L1 accumulator path.
    /// Default implementation returns `None` (no accumulator support).
    fn as_generic_mlp(&self) -> Option<(&GenericMlp, bool)> { None }
}

impl GameEvaluator for NnueEvaluator {
    fn evaluate(&self, gs: &GameState) -> f32 {
        self.evaluate_state(gs)
    }
}

/// Wrapper that adapts an `EvaluatingPlayer` (heuristic) to the `GameEvaluator` trait.
pub struct HeuristicEvaluator<'a> {
    inner: &'a dyn EvaluatingPlayer,
}

impl<'a> HeuristicEvaluator<'a> {
    pub fn new(inner: &'a dyn EvaluatingPlayer) -> Self {
        Self { inner }
    }
}

impl GameEvaluator for HeuristicEvaluator<'_> {
    fn evaluate(&self, gs: &GameState) -> f32 {
        self.inner.cheap_evaluate(gs) as f32
    }
}

/// Static heuristic evaluator using the four known heuristic enum values.
/// Fully `Send + Sync` -- no trait objects, works with rayon.
pub struct StaticHeuristicEvaluator {
    heuristics: Vec<duke_rust::game::ai::heuristics::Heuristics>,
}

impl StaticHeuristicEvaluator {
    pub fn new() -> Self {
        use duke_rust::game::ai::heuristics::Heuristics;
        Self {
            heuristics: vec![
                Heuristics::DukeMovementOptions,
                Heuristics::TotalTilesOnBoard,
                Heuristics::TotalMovementOptions,
                Heuristics::DiscardedUnits,
            ],
        }
    }
}

impl GameEvaluator for StaticHeuristicEvaluator {
    fn evaluate(&self, gs: &GameState) -> f32 {
        let owner = gs.current_player_turn();
        self.heuristics.iter()
            .map(|h| h.approx_difference(owner, gs))
            .sum::<f64>() as f32
    }
}

/// Create the standard tile bag (all tiles except Assassin).
pub fn create_bag() -> TileBag {
    TileBag::new(vec![
        TileType::Footman,
        TileType::Bowman,
        TileType::Knight,
        TileType::Pikeman,
        TileType::Pikeman,
        TileType::Champion,
        TileType::Priest,
        TileType::Wizard,
        TileType::Dragoon,
        // Assassin excluded: JumpSlide not yet fully implemented in board logic
        TileType::General,
        TileType::Marshall,
        TileType::Longbowman,
    ])
}

/// Create a standard initial game state.
pub fn create_initial_state(bag: &TileBag) -> GameState {
    GameState::new(
        bag,
        (DukeInitialLocation::Left, FootmenSetup::Left),
        (DukeInitialLocation::Right, FootmenSetup::Right),
    )
}

/// Play a game using random moves, collecting states at each turn.
pub fn play_random_game(gs: &GameState, rng: &mut StdRng) -> (Vec<GameState>, GameResult) {
    let ai = StupidSyncAi {};
    let mut game = gs.clone();
    let mut states = Vec::new();

    loop {
        match game.game_result() {
            GameResult::Ongoing => {
                if states.len() as u32 >= MAX_TURNS {
                    eprintln!("WARNING: play_random_game exceeded {} turns, forcing draw", MAX_TURNS);
                    states.push(game.clone());
                    return (states, GameResult::Tie);
                }
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

/// Play a game using self-play with epsilon-greedy exploration.
///
/// Accepts any `GameEvaluator` (NNUE, heuristic, etc.) for move selection.
/// Both players use the same evaluator.
pub fn play_selfplay_game<E: GameEvaluator + ?Sized>(
    gs: &GameState,
    evaluator: &E,
    rng: &mut StdRng,
    epsilon: f64,
) -> (Vec<GameState>, GameResult) {
    play_two_player_game(gs, evaluator, evaluator, rng, epsilon)
}

/// Play a game with two different evaluators (top_eval vs bottom_eval).
///
/// `top_eval` selects moves when it is TopPlayer's turn, `bottom_eval` when BottomPlayer's.
/// Epsilon-greedy: with probability `epsilon`, a random move is played instead.
pub fn play_two_player_game<E1: GameEvaluator + ?Sized, E2: GameEvaluator + ?Sized>(
    gs: &GameState,
    top_eval: &E1,
    bottom_eval: &E2,
    rng: &mut StdRng,
    epsilon: f64,
) -> (Vec<GameState>, GameResult) {
    use duke_rust::game::tile::Owner;

    let ai = StupidSyncAi {};
    let mut game = gs.clone();
    let mut states = Vec::new();

    loop {
        match game.game_result() {
            GameResult::Ongoing => {
                if states.len() as u32 >= MAX_TURNS {
                    states.push(game.clone());
                    return (states, GameResult::Tie);
                }
                states.push(game.clone());

                if rng.gen::<f64>() < epsilon {
                    ai.play_next_move(rng, &mut game);
                } else {
                    let current = game.current_player_turn();
                    let mv = match current {
                        Owner::TopPlayer => greedy_move(&game, top_eval, rng),
                        Owner::BottomPlayer => greedy_move(&game, bottom_eval, rng),
                    };
                    mv.play(&mut game, rng);
                }
            }
            result => {
                states.push(game.clone());
                return (states, result);
            }
        }
    }
}

/// Pick the move that minimizes the opponent's value (= maximizes our value).
///
/// Works with any `GameEvaluator` implementation (NNUE, heuristic, etc.).
/// When the evaluator wraps a sparse `GenericMlp` (1106 or 1147 inputs),
/// automatically uses the incremental L1 accumulator path for faster
/// candidate evaluation.
///
/// Panics if the game state has no legal moves.
pub fn greedy_move<E: GameEvaluator + ?Sized>(gs: &GameState, evaluator: &E, rng: &mut impl Rng) -> AiMove {
    // Check if the evaluator supports the incremental accumulator path.
    if let Some((net, include_combined)) = evaluator.as_generic_mlp() {
        return greedy_move_incremental(gs, net, include_combined, rng);
    }

    let mut moves: Vec<AiMove> = AiMove::all_moves(gs).collect();
    assert!(!moves.is_empty(), "greedy_move called with no legal moves");
    moves.shuffle(rng);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_move = None;

    // Create the deterministic rng once; clone per candidate to avoid re-seeding overhead.
    let base_eval_rng = StdRng::seed_from_u64(0);
    for mv in &moves {
        let mut clone = gs.clone();
        let mut eval_rng = base_eval_rng.clone();
        mv.play(&mut clone, &mut eval_rng);
        let prediction = evaluator.evaluate(&clone);
        // Negate: opponent's score is negative of ours.
        let score = -(prediction as f64);
        if score > best_score {
            best_score = score;
            best_move = Some(mv.clone());
        }
    }

    best_move.unwrap()
}

/// Pick the best move using incremental L1 accumulator updates.
///
/// Like `greedy_move`, but exploits the fact that most candidate moves only
/// change 2-4 features in the L1 input.  Builds the base L1 accumulator once
/// from the current position, then for each candidate:
///   1. Clone the state and play the move.
///   2. Clone the base accumulator.
///   3. Compute the feature diff (old vs new board/bag/combined features).
///   4. Patch the accumulator with the diff.
///   5. Complete the forward pass (ReLU + remaining layers).
///
/// `include_combined` should be `true` for 1147-input models, `false` for 1106.
pub fn greedy_move_incremental(
    gs: &GameState,
    net: &GenericMlp,
    include_combined: bool,
    rng: &mut impl Rng,
) -> AiMove {
    let mut moves: Vec<AiMove> = AiMove::all_moves(gs).collect();
    assert!(!moves.is_empty(), "greedy_move_incremental called with no legal moves");
    moves.shuffle(rng);

    // Build base accumulator and extract base features from the current position.
    let base_acc = L1Accumulator::from_state(net, gs, include_combined);
    let base_board = active_board_features(gs);
    let base_bag = bag_features(gs);
    let base_combined: Option<[f64; NUM_COMBINED_FEATURES]> = if include_combined {
        Some(extract_combined_features(gs))
    } else {
        None
    };

    // Evaluate each candidate move individually using incremental accumulator updates.
    let base_eval_rng = StdRng::seed_from_u64(0);
    let mut best_score = f64::NEG_INFINITY;
    let mut best_move = None;

    for mv in &moves {
        let mut clone = gs.clone();
        let mut eval_rng = base_eval_rng.clone();
        mv.play(&mut clone, &mut eval_rng);

        // Extract features from the post-move state.
        let new_board = active_board_features(&clone);
        let new_bag = bag_features(&clone);
        let new_combined: Option<[f64; NUM_COMBINED_FEATURES]> = if include_combined {
            Some(extract_combined_features(&clone))
        } else {
            None
        };

        // Clone the base accumulator and patch it with the diff.
        let mut acc = base_acc.clone();
        acc.update_features(
            net,
            &base_board,
            &new_board,
            &base_bag,
            &new_bag,
            base_combined.as_ref(),
            new_combined.as_ref(),
        );

        let prediction = acc.forward(net);
        // Negate: opponent's score is negative of ours.
        let score = -(prediction as f64);
        if score > best_score {
            best_score = score;
            best_move = Some(mv.clone());
        }
    }

    best_move.unwrap()
}
