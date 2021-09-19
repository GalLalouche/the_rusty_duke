use minimax_alpha_beta::strategy::AlphaBetaMiniMaxStrategy;
use rand::Rng;

use crate::game::ai::ai_move::{AiMove, ArtificialPlayer, ArtificialStrategy, EvaluatingPlayer};
use crate::game::ai::heuristics::Heuristic;
use crate::game::state::{GameMove, GameState};
use crate::game::tile::Owner;

pub struct HeuristicAi {
    owner: Owner,
    max_depth: usize,
    heuristics: Vec<Box<Heuristic>>,
}

impl HeuristicAi {
    pub fn new(owner: Owner, max_depth: usize, heuristics: Vec<Box<dyn Heuristic>>) -> HeuristicAi {
        HeuristicAi { owner, max_depth, heuristics }
    }
}

impl EvaluatingPlayer for HeuristicAi {
    fn evaluate(&self, gs: &GameState) -> f64 {
        self.heuristics.iter().map(|f| f.evaluate_for_owner(self.owner, gs)).sum()
    }
}

impl ArtificialPlayer for HeuristicAi {
    fn play_next_move<R>(&self, _rng: &mut R, gs: &mut GameState) -> () where R: Rng {
        assert_eq!(Owner::BottomPlayer, gs.current_player_turn);
        let mut strategy =
            ArtificialStrategy { state: gs.clone(), evaluator: Box::new(self) };
        let best_move = strategy.get_best_move(self.max_depth as i64, true);
        // TODO implement Into to avoid duplication
        let gm = best_move
            .to_game_move()
            .expect("ASSERTION ERROR: Sentinel should not have been the best move");
        gs.make_a_move(gm);
    }
}

// impl ArtificialPlayer for HeuristicAi {
//     fn play_next_move<R>(&self, rng: &mut R, gs: &mut GameState) -> () where R: Rng {
//         assert_eq!(self.max_depth, 1);
//         let mut best_gs = Vec::new();
//         let mut best_score = 0.0;
//         let mut make_move = |gm, state: &GameState| {
//             let mut clone = state.clone();
//             clone.make_a_move(gm);
//             // Swap current player so the heuristics run on the previous player, i.e., the one who
//             // made the last move.
//             clone.current_player_turn = clone.other_player();
//             let score = self.score(&clone);
//             println!("Debug");
//             println!("{}", clone.as_string());
//             println!("{}", score);
//             if score > best_score {
//                 best_score = score;
//                 best_gs = vec![clone];
//             } else if score == best_score {
//                 best_gs.push(clone);
//             }
//         };
//         for pm in gs.all_valid_game_moves() {
//             match pm {
//                 PossibleMove::PullTileFromBag => {
//                     let mut clone = gs.clone();
//                     clone.pull_tile_from_bag();
//                     for o in DukeOffset::iter() {
//                         if clone.is_valid_placement(o) {
//                             make_move(GameMove::PlaceNewTile(o), &clone);
//                         }
//                     }
//                 }
//                 PossibleMove::ApplyNonCommandTileAction { src, dst } =>
//                     make_move(GameMove::ApplyNonCommandTileAction { src, dst }, &gs),
//             }
//         }
//         *gs = best_gs.choose(rng.borrow_mut()).expect("No moves available :((").clone();
//     }
// }
