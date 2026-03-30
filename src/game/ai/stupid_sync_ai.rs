use rand::Rng;
use rand::seq::IteratorRandom;

use crate::game::ai::player::{AiMove, ArtificialPlayer};
use crate::game::state::GameState;

pub struct StupidSyncAi {}

impl ArtificialPlayer for StupidSyncAi {
    fn get_next_move<R: Rng>(&self, rng: &mut R, gs: &GameState) -> AiMove {
        // Use reservoir sampling from iterator to avoid collecting all moves into a Vec.
        gs.all_valid_game_moves_for_current_player()
            .choose(rng)
            .map(|pm| (&pm).into())
            .unwrap()
    }
}