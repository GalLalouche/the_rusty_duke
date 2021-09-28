use minimax_alpha_beta::strategy::AlphaBetaMiniMaxStrategy;
use rand::Rng;

use crate::game::ai::heuristics::{Heuristic, Heuristics};
use crate::game::ai::player::{AiMove, ArtificialPlayer, ArtificialStrategy, EvaluatingPlayer};
use crate::game::state::GameState;

pub struct HeuristicAi {
    max_depth: u32,
    heuristics: Vec<Box<dyn Heuristic>>,
}

impl HeuristicAi {
    pub fn new(max_depth: u32, heuristics: Vec<Box<dyn Heuristic>>) -> HeuristicAi {
        HeuristicAi { max_depth, heuristics }
    }
}

impl EvaluatingPlayer for HeuristicAi {
    fn evaluate(&self, gs: &GameState) -> f64 {
        self.heuristics.iter()
            .map(|h| h.difference(gs.current_player_turn(), gs))
            .sum()
    }

    fn cheap_evaluate(&self, gs: &GameState) -> f64 {
        self.heuristics.iter()
            .map(|h| h.approx_difference(gs.current_player_turn(), gs))
            .sum()
    }
}

impl ArtificialPlayer for HeuristicAi {
    // fn play_next_move<R>(&self, _rng: &mut R, gs: &mut GameState) -> () where R: Rng {
    //     // time_it("play_next_move", || {
    //     //     assert_eq!(Owner::BottomPlayer, gs.current_player_turn());
    //     //     let mut strategy =
    //     //         ArtificialStrategy { state: gs.clone(), evaluator: Box::new(self) };
    //     //     let mut nm = negamax(strategy, self.max_depth);
    //     //     let best_move = nm.choose_move(gs).unwrap();
    //     //     // TODO implement Into to avoid duplication
    //     //     let gm = best_move
    //     //         .to_game_move()
    //     //         .expect("ASSERTION ERROR: Sentinel should not have been the best move");
    //     //     gs.make_a_move(gm);
    //     // })
    // }

    fn get_next_move<R>(&self, _rng: &mut R, gs_orig: &GameState) -> AiMove where R: Rng {
        ArtificialStrategy { state: gs_orig.clone(), evaluator: Box::new(self) }
            .get_best_move(self.max_depth as i64, false)
    }

    fn create(max_depth: u32) -> Self {
        HeuristicAi::new(
            max_depth,
            vec!(
                Box::new(Heuristics::DukeMovementOptions),
                Box::new(Heuristics::TotalTilesOnBoard),
                Box::new(Heuristics::TotalMovementOptions),
            ),
        )
    }
}

// TODO fixtures
#[cfg(test)]
mod tests {
    use crate::game::ai::test::tests::{can_find_winning_move, can_find_winning_move_with_lookahead_2};

    use super::*;

    #[test]
    fn win_in_1() {
        can_find_winning_move::<HeuristicAi>()
    }

    #[test]
    fn win_in_2() {
        can_find_winning_move_with_lookahead_2::<HeuristicAi>()
    }
}