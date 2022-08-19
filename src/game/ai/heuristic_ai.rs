use crate::game::ai::heuristics::Heuristic;
use crate::game::ai::player::EvaluatingPlayer;
use crate::game::state::GameState;

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