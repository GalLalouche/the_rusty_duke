use std::convert::TryInto;

use crate::game::ai::my_negamax::Negamax;
use crate::game::ai::player::{AiMove, ArtificialStrategy};
use crate::game::state::GameState;
use crate::game::state::GameResult;
use crate::time_it_macro;

impl minimax::Move for AiMove {
    type G = GameState;

    fn apply(&self, state: &mut <Self::G as minimax::Game>::S) {
        time_it_macro!("minimax: apply", {
            self.play(state);
        });
    }

    fn undo(&self, state: &mut <Self::G as minimax::Game>::S) {
        time_it_macro!("undo", {
            state.undo(self.to_undo_move().unwrap());
        })
    }
}

impl minimax::Game for GameState {
    type S = GameState;
    type M = AiMove;

    fn generate_moves(state: &Self::S, moves: &mut Vec<Self::M>) {
        time_it_macro!("generate_moves", {
            moves.append(&mut AiMove::all_moves(state).collect());
        })
    }

    fn get_winner(state: &Self::S) -> Option<minimax::Winner> {
        time_it_macro!("get_winner", {
            match state.game_result() {
                GameResult::Won(o) => Some(
                    if o == state.current_player_turn() {
                        minimax::Winner::PlayerToMove
                    } else {
                        minimax::Winner::PlayerJustMoved
                    }),
                _ => None
            }
        })
    }
}

impl minimax::Evaluator for ArtificialStrategy<'_> {
    type G = GameState;

    fn evaluate(&self, s: &<Self::G as minimax::Game>::S) -> minimax::Evaluation {
        time_it_macro!("evaluate", {
            self.evaluator.evaluate(s) as i32
        })
    }
}

pub(super) fn negamax(ai: ArtificialStrategy, max_depth: usize) -> Negamax<ArtificialStrategy> {
    Negamax::new(ai, max_depth)
}
