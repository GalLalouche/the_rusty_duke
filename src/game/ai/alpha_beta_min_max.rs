use minimax_alpha_beta::strategy::AlphaBetaMiniMaxStrategy;
use rand::{Rng, RngCore, SeedableRng, thread_rng};
use rand::rngs::StdRng;

use crate::common::utils::Vectors;
use crate::game::ai::heuristics::HeuristicAi;
use crate::game::ai::heuristics::Heuristics;
use crate::game::ai::player::{AiMove, ArtificialPlayer, EvaluatingPlayer};
use crate::game::state::{GameMove, GameState};
use crate::game::tile::Owner;
use crate::time_it_macro;

struct OrdFloat(f64);

impl PartialEq for OrdFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

fn play_aux<R: Rng>(mv: &AiMove, state: &mut GameState, rng: &mut R) {
    time_it_macro!("alpha_beta: play", {
            match &mv {
                AiMove::PullTileFormBagAndPlay(o, _) =>
                    state.make_a_move(GameMove::PullAndPlay(*o), rng),
                AiMove::ApplyNonCommandTileAction { src, dst, .. } =>
                    state.make_a_move(GameMove::ApplyNonCommandTileAction {
                        src: *src,
                        dst: *dst,
                    }, rng),
                AiMove::Sentinel => {}
            }
        })
}

pub struct HeuristicAlphaBetaPlayer {
    pub(super) evaluator: Box<dyn EvaluatingPlayer>,
    pub max_depth: u32,
}

impl HeuristicAlphaBetaPlayer {
    pub fn all_heuristics_with_max_depth(max_depth: u32) -> Self {
        HeuristicAlphaBetaPlayer {
            evaluator: Box::new(HeuristicAi::new(
                vec![
                    Box::new(Heuristics::DukeMovementOptions),
                    Box::new(Heuristics::TotalTilesOnBoard),
                    Box::new(Heuristics::TotalMovementOptions),
                    Box::new(Heuristics::DiscardedUnits),
                ]
            )),
            max_depth,
        }
    }
}

pub(super) struct HeuristicAlphaBetaPlayerStrategy<'a> {
    pub state: GameState,
    pub player: &'a HeuristicAlphaBetaPlayer,
    rng: Box<dyn RngCore>
}

impl ArtificialPlayer for HeuristicAlphaBetaPlayer {
    fn get_next_move<R: Rng>(&self, rng: &mut R, gs: &GameState) -> AiMove {
        let split = StdRng::seed_from_u64(rng.gen());
        HeuristicAlphaBetaPlayerStrategy { state: gs.clone(), player: self, rng: Box::new(split) }
            .get_best_move(self.max_depth as i64, false)
    }
}

impl<'a> minimax_alpha_beta::strategy::Strategy for HeuristicAlphaBetaPlayerStrategy<'a> {
    type Player = Owner;
    type Move = AiMove;
    type Board = GameState;

    fn evaluate(&self) -> f64 {
        if self.state.is_over() {
            f64::INFINITY
        } else {
            self.player.evaluator.evaluate(&self.state)
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
        AiMove::all_moves(&self.state)
            .for_each(|mv: AiMove| {
                play_aux(&mv, &mut clone, &mut thread_rng());
                let res = self.player.evaluator.cheap_evaluate(&clone);
                // TODO reduce duplication with below
                if let Some(um) = mv.to_undo_move() {
                    clone.undo(um)
                }
                result.push((mv, res))
            });
        result.better_sort_by_key(|e| -e.1).into_iter().map(|e| e.0).collect()
    }

    fn play(&mut self, mv: &Self::Move, _maximizer: bool) {
        play_aux(mv, &mut self.state, &mut self.rng);
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

// TODO fixtures
#[cfg(test)]
mod tests {
    use crate::game::ai::alpha_beta_min_max::HeuristicAlphaBetaPlayer;
    use crate::game::ai::test::tests::{can_find_winning_move, can_find_winning_move_with_lookahead_2};

    use super::*;

    #[test]
    fn win_in_1() {
        can_find_winning_move(HeuristicAlphaBetaPlayer::all_heuristics_with_max_depth(1))
    }

    #[test]
    fn win_in_2() {
        can_find_winning_move_with_lookahead_2(HeuristicAlphaBetaPlayer::all_heuristics_with_max_depth(2))
    }
}

use crate::game::state::GameResult;

impl minimax::Move for AiMove {
    type G = GameState;

    fn apply(&self, state: &mut <Self::G as minimax::Game>::S) {
        time_it_macro!("minimax: apply", {
            self.play(state, &mut thread_rng());
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

impl minimax::Evaluator for HeuristicAlphaBetaPlayerStrategy<'_> {
    type G = GameState;

    fn evaluate(&self, s: &<Self::G as minimax::Game>::S) -> minimax::Evaluation {
        time_it_macro!("evaluate", {
            self.player.evaluator.evaluate(s) as i32
        })
    }
}
