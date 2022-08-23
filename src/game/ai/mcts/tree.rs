use std::borrow::Borrow;
use rand::{Rng, RngCore};
use crate::{assert_not, debug_assert_not, time_it_macro};
use crate::common::percentage::Percentage;
use crate::common::utils::split_rng;
use crate::game::ai::mcts::Playouts;
use crate::game::ai::mcts::tree::Node::{Explored, FiniteState, SinglePassSimulation, Unexplored};
use crate::game::ai::player::AiMove;
use crate::game::state::{GameResult, GameState};
use crate::game::tile::Owner;

const PLACE_TILE_PERCENTAGE: f64 = 0.5;

type Depth = u64;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default, Hash)]
struct Ratio {
    wins: Playouts,
    total_games: Playouts,
}

struct GameRoot {
    root_state: GameState,
    ratio: Ratio,
    tree: Vec<Node>,
    rng: Box<dyn RngCore>,
}

#[derive(Debug, Clone, PartialEq)]
enum Node {
    Unexplored(AiMove, GameState),
    FiniteState(AiMove, TestResult),
    SinglePassSimulation(AiMove, TestResult, Box<Node>),
    Explored(AiMove, Ratio, Vec<Node>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TestResult {
    CurrentPlayerWon,
    CurrentPlayerLost,
    Tie,
    MaxDepthReached { final_state_score: f64 },
}

impl Node {
    pub fn ai_move(&self) -> &AiMove {
        match self {
            Unexplored(mv, _) => mv,
            FiniteState(mv, _) => mv,
            SinglePassSimulation(mv, _, _) => mv,
            Explored(mv, _, _) => mv,
        }
    }
    pub fn unexplored<R: Rng>(mv: AiMove, gs: GameState, rng: &mut R) -> Self {
        let mut gs = gs;
        mv.play(&mut gs, rng);
        Unexplored(mv, gs)
    }
    pub fn is_explored(&self) -> bool {
        match self {
            Unexplored { .. } => false,
            Explored { .. } => true,
            FiniteState { .. } => todo!(),
            SinglePassSimulation { .. } => todo!(),
        }
    }

    pub fn simulate<R: Rng>(&mut self, rng: &mut R) -> Self {
        match self {
            Unexplored(ai_move, gs) => Node::simulate_unexplored(ai_move, gs, rng),
            _ => todo!(),
        }
    }

    fn simulate_unexplored<R: Rng>(ai_move: &AiMove, gs: &GameState, rng: &mut R) -> Self {
        let mv = random_move(gs, rng);
        let mut gs = gs.clone();
        let result = Node::test_move(
            gs.current_player_turn(),
            &mv,
            &mut gs,
            rng,
            0,
        );
        SinglePassSimulation(
            ai_move.clone(),
            result.1,
            Box::new(result.0),
        )
    }


    fn test_move<R: Rng>(
        current_player: Owner,
        mv: &AiMove,
        gs: &mut GameState,
        rng: &mut R,
        depth: Depth,
    ) -> (Node, TestResult) {
        debug_assert_not!(gs.is_over());
        time_it_macro!("play", {mv.play(gs, rng)});
        macro_rules! finite {
            ($test_result: expr) => {(FiniteState(mv.clone(), $test_result), $test_result)}
        }
        match time_it_macro!("game_result", {gs.game_result()}) {
            GameResult::Tie => finite!(TestResult::Tie),
            GameResult::Won(o) =>
                if o == current_player {
                    finite!(TestResult::CurrentPlayerWon)
                } else {
                    finite!(TestResult::CurrentPlayerLost)
                },
            GameResult::Ongoing => {
                let next_move: AiMove = random_move(gs, rng);
                let (node, res) = Node::test_move(current_player, &next_move, gs, rng, depth + 1);
                (SinglePassSimulation(mv.clone(), res, Box::new(node)), res)
            }
        }
    }

    // TODO this should return a Vec<String>
    fn print_single_pass_simulation_plays<R: Rng>(&self, gs: GameState, rng: &mut R) -> () {
        match self {
            SinglePassSimulation(mv, _, child) => {
                let mut gs = gs;
                mv.play(&mut gs, rng);
                println!("{}", gs.current_player_turn());
                println!("{}", gs.as_double_string());
                Node::print_single_pass_simulation_plays(child, gs, rng);
            }
            FiniteState(mv, result) => {
                let mut gs = gs;
                mv.play(&mut gs, rng);
                println!("FINAL STATE: '{:?}'", result);
                println!("{}", gs.as_double_string());
            }
            e => panic!("Unsupported node type for single pass: '{:?}'", e),
        }
    }
}

fn random_move<R: Rng>(gs: &GameState, rng: &mut R) -> AiMove {
    AiMove::from(
        gs
            .get_random_move_for_current_player(rng, Percentage::new(PLACE_TILE_PERCENTAGE))
            .unwrap()
            .borrow()
    )
}

impl GameRoot {
    pub fn initialize<R: Rng>(gs: GameState, rng: &mut R) -> Self {
        let moves: Vec<Node> =
            AiMove::all_moves(&gs).map(|e| Node::unexplored(e, gs.clone(), rng)).collect();
        assert_not!(moves.is_empty());
        GameRoot {
            root_state: gs,
            tree: moves,
            ratio: Ratio::default(),
            rng: Box::new(split_rng(rng)),
        }
    }

    pub fn select(&mut self) -> () {
        let mut initial = false;
        for unexplored_node in self.tree.iter_mut().filter(|e| !e.is_explored()) {
            *unexplored_node = unexplored_node.simulate(&mut self.rng);
            initial = true;
        }
        if initial {
            // self.update_ratio();
            return;
        }
        todo!()
    }

    fn update_ratio(&mut self) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::StdRng;
    use rand::SeedableRng;
    use crate::common::utils::test_rng;
    use crate::game::bag::TileBag;
    use crate::game::board_setup::{DukeInitialLocation, FootmenSetup};
    use crate::game::tile::TileRef;
    use crate::game::units;
    use super::*;

    #[test]
    #[ignore]
    fn my_main() {
        let mut gs = GameState::new(
            &TileBag::new(vec!(
                TileRef::new(units::footman()),
                TileRef::new(units::footman()),
                TileRef::new(units::pikeman()),
                TileRef::new(units::pikeman()),
            )),
            (DukeInitialLocation::Left, FootmenSetup::Left),
            (DukeInitialLocation::Right, FootmenSetup::Right),
        );
        let mut tree = GameRoot::initialize(gs.clone(), &mut test_rng());
        let mut rng: StdRng = StdRng::seed_from_u64(42);
        tree.select();
        tree.tree.get(0).unwrap().print_single_pass_simulation_plays(gs, &mut test_rng());
    }
}