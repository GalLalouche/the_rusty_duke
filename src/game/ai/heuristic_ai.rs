use minimax::{Negamax, Strategy};
use minimax_alpha_beta::strategy::AlphaBetaMiniMaxStrategy;
use rand::Rng;

use crate::time_it_macro;
use crate::game::ai::heuristics::Heuristic;
use crate::game::ai::minimax::negamax;
use crate::game::ai::player::{ArtificialPlayer, ArtificialStrategy, EvaluatingPlayer};
use crate::game::state::GameState;
use crate::game::tile::Owner;

pub struct HeuristicAi {
    owner: Owner,
    max_depth: usize,
    heuristics: Vec<Box<dyn Heuristic>>,
}

impl HeuristicAi {
    pub fn new(owner: Owner, max_depth: usize, heuristics: Vec<Box<dyn Heuristic>>) -> HeuristicAi {
        HeuristicAi { owner, max_depth, heuristics }
    }
}

impl EvaluatingPlayer for HeuristicAi {
    fn evaluate(&self, gs: &GameState) -> f64 {
        self.heuristics.iter().map(|h| {
            h.difference(self.owner, gs)
        }
        ).sum()
    }
}

impl ArtificialPlayer for HeuristicAi {
    fn play_next_move<R>(&self, _rng: &mut R, gs: &mut GameState) -> () where R: Rng {
        time_it_macro!("play_next_move", {
            assert_eq!(Owner::BottomPlayer, gs.current_player_turn());
            let mut strategy =
                ArtificialStrategy { state: gs.clone(), evaluator: Box::new(self) };
            let best_move = strategy.get_best_move(self.max_depth as i64, false);
            // TODO implement Into to avoid duplication
            let gm = best_move
                .to_game_move()
                .expect("ASSERTION ERROR: Sentinel should not have been the best move");
            gs.make_a_move(gm);
        })
        // time_it("play_next_move", || {
        //     assert_eq!(Owner::BottomPlayer, gs.current_player_turn());
        //     let mut strategy =
        //         ArtificialStrategy { state: gs.clone(), evaluator: Box::new(self) };
        //     let mut nm = negamax(strategy, self.max_depth);
        //     let best_move = nm.choose_move(gs).unwrap();
        //     // TODO implement Into to avoid duplication
        //     let gm = best_move
        //         .to_game_move()
        //         .expect("ASSERTION ERROR: Sentinel should not have been the best move");
        //     gs.make_a_move(gm);
        // })
    }
}
