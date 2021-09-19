use crate::game::state::GameState;
use crate::game::tile::Owner;

pub trait Heuristic {
    fn evaluate_for_owner(&self, o: Owner, gs: &GameState) -> f64;
    fn difference(&self, o: Owner, gs: &GameState) -> f64 {
        let o_score = self.evaluate_for_owner(o, &gs);
        let other_score = self.evaluate_for_owner(o.next_player(), &gs);
        o_score - other_score
    }
}

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
                gs.get_tiles_for_owner(o).len() as f64 * 10.0,
            Heuristics::TotalMovementOptions =>
                gs.all_valid_game_moves_for(o).len() as f64,
        }
    }
}
