use std::fmt::{Debug};

use crate::game::state::GameState;
use crate::game::tile::Owner;

pub trait Heuristic: Debug {
    fn name(&self) -> String;
    fn evaluate_for_owner(&self, o: Owner, gs: &GameState) -> f64;
    fn approx_evaluate_for_owner(&self, o: Owner, gs: &GameState) -> f64;
    // TODO reduce duplication
    fn difference(&self, owner: Owner, gs: &GameState) -> f64 {
        let owner_score = self.evaluate_for_owner(owner, &gs);
        let other_score = self.evaluate_for_owner(owner.next_player(), &gs);
        owner_score - other_score
    }
    fn approx_difference(&self, owner: Owner, gs: &GameState) -> f64 {
        let owner_score = self.approx_evaluate_for_owner(owner, &gs);
        let other_score = self.approx_evaluate_for_owner(owner.next_player(), &gs);
        owner_score - other_score
    }
}


#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum Heuristics {
    DukeMovementOptions,
    TotalTilesOnBoard,
    TotalMovementOptions,
    DiscardedUnits,
}

impl Heuristic for Heuristics {
    fn name(&self) -> String { format!("{:?}", self) }

    // TODO reduce duplication
    fn evaluate_for_owner(&self, o: Owner, gs: &GameState) -> f64 {
        match self {
            Heuristics::DukeMovementOptions =>
                gs.get_legal_moves(gs.duke_coordinate(o)).len() as f64,
            Heuristics::TotalTilesOnBoard =>
                10.0 * gs.get_tiles_for_owner(o).len() as f64,
            Heuristics::TotalMovementOptions =>
                gs.all_valid_game_moves_for(o).count() as f64,
            Heuristics::DiscardedUnits => gs.discard_bag_for(o).len() as f64 * -10.0,
        }
    }

    fn approx_evaluate_for_owner(&self, o: Owner, gs: &GameState) -> f64 {
        match self {
            Heuristics::DukeMovementOptions =>
                gs.get_legal_moves_ignoring_guard(gs.duke_coordinate(o)).len() as f64,
            Heuristics::TotalTilesOnBoard =>
                10.0 * gs.get_tiles_for_owner(o).len() as f64,
            Heuristics::TotalMovementOptions =>
                gs.get_tiles_for_owner(o)
                    .into_iter()
                    .map(|e| e.1.get_current_side().actions().len())
                    .sum::<usize>() as f64,
            Heuristics::DiscardedUnits => gs.discard_bag_for(o).len() as f64 * -10.0,
        }
    }
}
