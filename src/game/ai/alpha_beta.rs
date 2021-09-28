use std::borrow::Borrow;

use crate::common::utils::Vectors;
use crate::game::ai::player::{AiMove, ArtificialStrategy};
use crate::game::state::{GameMove, GameState};
use crate::game::tile::Owner;
use crate::time_it_macro;

struct OrdFloat(f64);

impl PartialEq for OrdFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

fn play_aux(mv: &AiMove, state: &mut GameState) {
    time_it_macro!("alpha_beta: play", {
            match &mv {
                AiMove::PullTileFormBagAndPlay(o, _) =>
                    state.make_a_move(GameMove::PullAndPlay(*o)),
                AiMove::ApplyNonCommandTileAction { src, dst, .. } =>
                    state.make_a_move(GameMove::ApplyNonCommandTileAction {
                        src: *src,
                        dst: *dst,
                    }),
                AiMove::Sentinel => {}
            }
        })
}

impl<'a> minimax_alpha_beta::strategy::Strategy for ArtificialStrategy<'a> {
    type Player = Owner;
    type Move = AiMove;
    type Board = GameState;

    fn evaluate(&self) -> f64 {
        if self.state.is_over() {
            f64::INFINITY
        } else {
            self.evaluator.evaluate(&self.state)
        }
    }

    fn get_winner(&self) -> Self::Player {
        unimplemented!("Because this lib is stupid.")
    }

    fn is_game_tied(&self) -> bool {
        unimplemented!("Because this lib is stupid.")
    }

    fn is_game_complete(&self) -> bool {
        self.state.is_over()
    }

    fn get_available_moves(&self) -> Vec<Self::Move> {
        let mut clone = self.state.clone();
        let mut result = Vec::new();
        self.state
            .all_valid_game_moves_for_current_player()
            .map(|e| e.borrow().into())
            .for_each(|mv: AiMove| {
                play_aux(&mv, &mut clone);
                let res = self.evaluator.cheap_evaluate(&clone);
                // TODO reduce duplication with below
                if let Some(um) = mv.to_undo_move() {
                    clone.undo(um)
                }
                result.push((mv, res))
            });
        result.better_sort_by_key(|e| -e.1).into_iter().map(|e| e.0).collect()
    }

    fn play(&mut self, mv: &Self::Move, _maximizer: bool) {
        play_aux(mv, &mut self.state);
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
