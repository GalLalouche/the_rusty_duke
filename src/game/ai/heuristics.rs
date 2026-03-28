use std::fmt::Debug;

use crate::game::ai::player::EvaluatingPlayer;
use crate::game::state::GameState;
use crate::game::tile::Owner;

fn diff<F>(owner: Owner, gs: &GameState, f: F) -> f64 where F: Fn(Owner, &GameState) -> f64 {
    let owner_score = f(owner, &gs);
    let other_score = f(owner.next_player(), &gs);
    owner_score - other_score
}

pub trait Heuristic: Debug {
    fn name(&self) -> String;
    fn evaluate_for_owner(&self, o: Owner, gs: &GameState) -> f64;
    fn approx_evaluate_for_owner(&self, o: Owner, gs: &GameState) -> f64;
    fn difference(&self, owner: Owner, gs: &GameState) -> f64 {
        diff(owner, gs, |o, g| self.evaluate_for_owner(o, g))
    }
    fn approx_difference(&self, owner: Owner, gs: &GameState) -> f64 {
        diff(owner, gs, |o, g| self.approx_evaluate_for_owner(o, g))
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

    fn evaluate_for_owner(&self, o: Owner, gs: &GameState) -> f64 {
        match self {
            Heuristics::DukeMovementOptions =>
                gs.get_legal_moves(gs.duke_coordinate(o)).len() as f64,
            Heuristics::TotalTilesOnBoard =>
                10.0 * gs.get_tiles_for_owner(o).len() as f64,
            Heuristics::TotalMovementOptions =>
                gs.all_valid_game_moves_for(o).count() as f64,
            Heuristics::DiscardedUnits => gs.discard_bag_for(o).len() as f64 * -15.0,
        }
    }

    fn approx_evaluate_for_owner(&self, o: Owner, gs: &GameState) -> f64 {
        match self {
            Heuristics::DukeMovementOptions =>
                gs.get_legal_moves_ignoring_guard(gs.duke_coordinate(o)).len() as f64,
            e => e.evaluate_for_owner(o, gs),
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
