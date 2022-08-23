use std::borrow::Borrow;
use std::collections::HashMap;

use rand::Rng;
use rand::seq::SliceRandom;

use crate::common::percentage::Percentage;
use crate::game::ai::player::{AiMove, ArtificialPlayer};
use crate::game::state::{GameResult, GameState};
use crate::game::tile::Owner;
use crate::time_it_macro;

pub struct MctsPlayer {
    pub playouts: u64,
    pub max_depth: Option<usize>,
}

impl MctsPlayer {
    fn test_move<R: Rng>(
        &self,
        current_player: Owner,
        mv: &AiMove,
        gs: &mut GameState,
        rng: &mut R,
        depth: usize,
    ) -> (Depth, TestResult) {
        debug_assert!(!gs.is_over());
        if self.max_depth.contains(&depth) {
            panic!();
        }
        time_it_macro!("play", {mv.play(gs, rng)});
        match time_it_macro!("game_result", {gs.game_result()}) {
            GameResult::Tie => (depth, TestResult::Tie),
            GameResult::Ongoing => {
                let next_move = random_move(gs, rng);
                self.test_move(current_player, &next_move, gs, rng, depth + 1)
            }
            GameResult::Won(o) =>
                if o == current_player {
                    (depth, TestResult::CurrentPlayerWon)
                } else {
                    (depth, TestResult::CurrentPlayerLost)
                }
        }
    }
}

impl ArtificialPlayer for MctsPlayer {
    fn get_next_move<R: Rng>(&self, rng: &mut R, gs: &GameState) -> AiMove {
        let all_moves: Vec<AiMove> = AiMove::all_moves(gs).collect();
        let mut scores: HashMap<&AiMove, i64> = HashMap::new();
        for _ in 0..self.playouts {
            let move_to_test = all_moves.choose(rng).unwrap();
            let mut temp_gs = gs.clone();
            let result = time_it_macro!(
                "rollout",
                {self.test_move(gs.current_player_turn(), move_to_test, &mut temp_gs, rng, 0)}
            );
            // println!("depth: {}, rollout time: {:?}", result.0, get_time_in_nanos("rollout").map(|e| e as f64 / 1e9));
            let score = match result.1 {
                TestResult::CurrentPlayerWon => 1,
                TestResult::CurrentPlayerLost => -1,
                TestResult::Tie => 0,
                TestResult::MaxDepthReached { .. } => todo!()
            };
            *scores.entry(move_to_test).or_insert(0) += score;
        }
        // println!("{:?}", scores);
        // println!(
        //     "play: {:?}, game_result: {:?}, random_move: {:?}, is_guard: {:?}",
        //     get_time_in_seconds("play"),
        //     get_time_in_seconds("game_result"),
        //     get_time_in_seconds("random_move"),
        //     get_time_in_seconds("is_guard"),
        // );
        scores.iter()
            .max_by(|a, b| a.1.cmp(&b.1))
            .map(|(k, _v)| k)
            .unwrap()
            .clone()
            .clone()
    }
}

enum TestResult {
    CurrentPlayerWon,
    CurrentPlayerLost,
    Tie,
    MaxDepthReached { final_state_score: f64 },
}

fn random_move<R: Rng>(gs: &GameState, rng: &mut R) -> AiMove {
    time_it_macro!("random_move",{
    gs.get_random_move_for_current_player(rng, Percentage::new(0.5)).unwrap().borrow().into()
    })
}

type Depth = usize;
