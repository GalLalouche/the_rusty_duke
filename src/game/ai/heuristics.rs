use std::fmt::Debug;

use crate::game::ai::player::EvaluatingPlayer;
use crate::game::state::GameState;
use crate::game::tile::Owner;

pub trait Heuristic: Debug {
    fn name(&self) -> String;
    fn evaluate_for_owner(&self, o: Owner, gs: &mut GameState) -> f64;
    fn approx_evaluate_for_owner(&self, o: Owner, gs: &GameState) -> f64;
    fn difference(&self, owner: Owner, gs: &mut GameState) -> f64 {
        let owner_score = self.evaluate_for_owner(owner, gs);
        let other_score = self.evaluate_for_owner(owner.next_player(), gs);
        owner_score - other_score
    }
    fn approx_difference(&self, owner: Owner, gs: &GameState) -> f64 {
        let owner_score = self.approx_evaluate_for_owner(owner, gs);
        let other_score = self.approx_evaluate_for_owner(owner.next_player(), gs);
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
    fn name(&self) -> String {
        match self {
            Heuristics::DukeMovementOptions => "DukeMovementOptions",
            Heuristics::TotalTilesOnBoard => "TotalTilesOnBoard",
            Heuristics::TotalMovementOptions => "TotalMovementOptions",
            Heuristics::DiscardedUnits => "DiscardedUnits",
        }.to_owned()
    }

    fn evaluate_for_owner(&self, o: Owner, gs: &mut GameState) -> f64 {
        match self {
            Heuristics::DukeMovementOptions =>
                gs.get_legal_moves(gs.duke_coordinate(o)).len() as f64,
            Heuristics::TotalTilesOnBoard =>
                10.0 * gs.count_tiles_for_owner(o) as f64,
            Heuristics::TotalMovementOptions =>
                gs.all_valid_game_moves_for(o).count() as f64,
            Heuristics::DiscardedUnits => gs.discard_bag_for(o).len() as f64 * -15.0,
        }
    }

    fn approx_evaluate_for_owner(&self, o: Owner, gs: &GameState) -> f64 {
        match self {
            Heuristics::DukeMovementOptions =>
                gs.count_legal_moves_ignoring_guard(gs.duke_coordinate(o)) as f64,
            Heuristics::TotalTilesOnBoard =>
                10.0 * gs.count_tiles_for_owner(o) as f64,
            Heuristics::TotalMovementOptions =>
                gs.all_valid_game_moves_for_ignoring_guard(o).len() as f64,
            Heuristics::DiscardedUnits => gs.discard_bag_for(o).len() as f64 * -15.0,
        }
    }
}

pub struct HeuristicAi {
    heuristics: Vec<Box<dyn Heuristic>>,
}

impl HeuristicAi {
    pub fn new(heuristics: Vec<Box<dyn Heuristic>>) -> HeuristicAi { HeuristicAi { heuristics } }
}

impl EvaluatingPlayer for HeuristicAi {
    fn evaluate(&self, gs: &mut GameState) -> f64 {
        let turn = gs.current_player_turn();
        self.heuristics.iter()
            .map(|h| h.difference(turn, gs))
            .sum()
    }

    fn cheap_evaluate(&self, gs: &GameState) -> f64 {
        self.heuristics.iter()
            .map(|h| h.approx_difference(gs.current_player_turn(), gs))
            .sum()
    }
}
