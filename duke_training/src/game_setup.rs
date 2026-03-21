//! Shared game setup used by training, benchmarking, and tests.

use std::sync::Arc;

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use duke_rust::game::ai::player::{AiMove, EvaluatingPlayer};
use duke_rust::game::ai::player::ArtificialPlayer;
use duke_rust::game::ai::stupid_sync_ai::StupidSyncAi;
use duke_rust::game::bag::TileBag;
use duke_rust::game::board_setup::{DukeInitialLocation, FootmenSetup};
use duke_rust::game::state::{GameResult, GameState};
use duke_rust::game::units;

use crate::nnue::NnueEvaluator;

/// Common trait for anything that can evaluate a game state.
/// Higher values = better for the current player.
/// The scale is arbitrary — only relative ordering matters for move selection.
pub trait GameEvaluator {
    fn evaluate(&self, gs: &GameState) -> f32;
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

/// Create the standard tile bag (all tiles except Assassin).
pub fn create_bag() -> TileBag {
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
        // Assassin excluded: JumpSlide not yet fully implemented in board logic
        Arc::new(units::general()),
        Arc::new(units::marshall()),
        Arc::new(units::longbowman()),
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
pub fn play_nnue_game(
    gs: &GameState,
    evaluator: &NnueEvaluator,
    rng: &mut StdRng,
    epsilon: f64,
) -> (Vec<GameState>, GameResult) {
    let ai = StupidSyncAi {};
    let mut game = gs.clone();
    let mut states = Vec::new();

    loop {
        match game.game_result() {
            GameResult::Ongoing => {
                states.push(game.clone());

                if rng.gen::<f64>() < epsilon {
                    ai.play_next_move(rng, &mut game);
                } else {
                    let mv = greedy_move(&game, evaluator, rng);
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
pub fn greedy_move(gs: &GameState, evaluator: &dyn GameEvaluator, rng: &mut impl Rng) -> AiMove {
    let mut moves: Vec<AiMove> = AiMove::all_moves(gs).collect();
    moves.shuffle(rng);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_move = None;

    for mv in &moves {
        let mut clone = gs.clone();
        // Use a deterministic rng for play so candidate evaluation doesn't
        // corrupt the main rng or depend on move order.
        let mut eval_rng = StdRng::seed_from_u64(0);
        mv.play(&mut clone, &mut eval_rng);
        let prediction = evaluator.evaluate(&clone);
        // After our move it's opponent's turn. The evaluator returns a score
        // where higher = better for the current player (the opponent).
        // We want to MINIMIZE the opponent's score, so negate it.
        let score = -(prediction as f64);
        if score > best_score {
            best_score = score;
            best_move = Some(mv.clone());
        }
    }

    best_move.unwrap()
}
