use std::fmt::Debug;

use crate::game::state::GameState;
use crate::game::tile::Owner;

pub trait Heuristic: Debug {
    fn evaluate_for_owner(&self, o: Owner, gs: &GameState) -> f64;
    fn difference(&self, owner: Owner, gs: &GameState) -> f64 {
        let owner_score = self.evaluate_for_owner(owner, &gs);
        let other_score = self.evaluate_for_owner(owner.next_player(), &gs);
        owner_score - other_score
    }
}


#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum Heuristics {
    DukeMovementOptions,
    TotalTilesOnBoard,
    TotalMovementOptions,
}

impl Heuristic for Heuristics {
    fn evaluate_for_owner(&self, o: Owner, gs: &GameState) -> f64 {
        match self {
            Heuristics::DukeMovementOptions =>
                gs.get_legal_moves(gs.duke_coordinate(o)).len() as f64,
            Heuristics::TotalTilesOnBoard =>
                gs.get_tiles_for_owner(o).len() as f64,
            Heuristics::TotalMovementOptions =>
                gs.all_valid_game_moves_for(o).count() as f64,
        }
    }
}
