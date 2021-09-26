use crate::game::ai::player::{AiMove, ArtificialStrategy};
use crate::game::state::{GameMove, GameState};
use crate::game::tile::Owner;
use crate::time_it_macro;

impl<'a> minimax_alpha_beta::strategy::Strategy for ArtificialStrategy<'a> {
    type Player = Owner;
    type Move = AiMove;
    type Board = GameState;

    fn evaluate(&self) -> f64 {
        time_it_macro!("alpha_beta: evaluate", {
            self.evaluator.evaluate(self.get_board())
        })
    }

    fn get_winner(&self) -> Self::Player {
        time_it_macro!("alpha_beta: get_winner", {
            self.state.winner().expect("I think(?!) this shouldn't be called if there's no winner")
        })
    }

    fn is_game_tied(&self) -> bool {
        false
    }

    fn is_game_complete(&self) -> bool {
        time_it_macro!("alpha_beta: is_game_complete", {
            self.state.is_over()
        })
    }

    fn get_available_moves(&self) -> Vec<Self::Move> {
        time_it_macro!("alpha_beta: get_available_moves", {
            self.state.all_valid_game_moves_for_current_player().iter().map(|e| e.into()).collect()
        })
    }

    fn play(&mut self, mv: &Self::Move, _maximizer: bool) {
        time_it_macro!("alpha_beta: play", {
            match &mv {
                AiMove::PullTileFormBagAndPlay(o, _) =>
                    self.state.make_a_move(GameMove::PullAndPlay(*o)),
                AiMove::ApplyNonCommandTileAction { src, dst, .. } =>
                    self.state.make_a_move(GameMove::ApplyNonCommandTileAction {
                        src: *src,
                        dst: *dst,
                    }),
                AiMove::Sentinel => {}
            }
        })
    }

    fn clear(&mut self, mv: &Self::Move) {
        time_it_macro!("alpha_beta: clear", {
            if let Some(um) = mv.to_undo_move() {
                self.state.undo(um)
            }
        });
    }

    fn get_board(&self) -> &Self::Board {
        &self.state
    }

    fn is_a_valid_move(&self, _mv: &Self::Move) -> bool {
        true // get_available moves already filters this
    }

    fn get_a_sentinel_move(&self) -> Self::Move {
        AiMove::Sentinel
    }
}
