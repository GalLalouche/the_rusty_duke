//! Match-playing logic extracted from the benchmark binary for testability.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use duke_rust::game::ai::player::ArtificialPlayer;
use duke_rust::game::ai::stupid_sync_ai::StupidSyncAi;
use duke_rust::game::state::{GameResult, GameState};
use duke_rust::game::tile::Owner;

use crate::game_setup::{greedy_move, GameEvaluator};

/// Represents a player strategy in a benchmark match.
pub enum Player<'a> {
    Random,
    Evaluator(&'a (dyn GameEvaluator + Sync)),
}

// Player is Sync because &(dyn GameEvaluator + Sync) is Sync
unsafe impl Sync for Player<'_> {}

/// Play a single match between a top player and a bottom player.
///
/// Returns the `GameResult` when the game ends or a `Tie` if `max_turns` is exceeded.
pub fn play_match(
    gs: &GameState,
    top_player: &Player,
    bottom_player: &Player,
    rng: &mut StdRng,
    max_turns: u32,
) -> GameResult {
    let ai = StupidSyncAi {};
    let mut game = gs.clone();
    let mut turns = 0u32;

    loop {
        match game.game_result() {
            GameResult::Ongoing => {
                if turns >= max_turns {
                    return GameResult::Tie;
                }
                let current = game.current_player_turn();
                let player = match current {
                    Owner::TopPlayer => top_player,
                    Owner::BottomPlayer => bottom_player,
                };
                match player {
                    Player::Random => {
                        ai.play_next_move(rng, &mut game);
                    }
                    Player::Evaluator(eval) => {
                        let mv = greedy_move(&game, *eval, rng);
                        mv.play(&mut game, rng);
                    }
                }
                turns += 1;
            }
            result => return result,
        }
    }
}

/// Aggregated result of running multiple matches between two players.
pub struct MatchResult {
    pub player_a_wins: u32,
    pub player_b_wins: u32,
    pub ties: u32,
}

/// Run `num_games` matches between `player_a` and `player_b`, alternating sides.
/// Games are played in parallel using rayon.
///
/// Even seeds: player_a plays as Top, player_b plays as Bottom.
/// Odd seeds: player_b plays as Top, player_a plays as Bottom.
///
/// Prints a summary line with the label and timing information.
pub fn run_matches(
    gs: &GameState,
    player_a: &Player,
    player_b: &Player,
    num_games: u32,
    label: &str,
) -> MatchResult {
    let start = Instant::now();

    let results: Vec<GameResult> = (0..num_games)
        .into_par_iter()
        .map(|seed| {
            let mut rng = StdRng::seed_from_u64(seed as u64);
            if seed % 2 == 0 {
                play_match(gs, player_a, player_b, &mut rng, 200)
            } else {
                let r = play_match(gs, player_b, player_a, &mut rng, 200);
                match r {
                    GameResult::Won(Owner::TopPlayer) => GameResult::Won(Owner::BottomPlayer),
                    GameResult::Won(Owner::BottomPlayer) => GameResult::Won(Owner::TopPlayer),
                    other => other,
                }
            }
        })
        .collect();

    let mut result = MatchResult {
        player_a_wins: 0,
        player_b_wins: 0,
        ties: 0,
    };
    for game_result in &results {
        match game_result {
            GameResult::Won(Owner::TopPlayer) => result.player_a_wins += 1,
            GameResult::Won(Owner::BottomPlayer) => result.player_b_wins += 1,
            _ => result.ties += 1,
        }
    }

    let elapsed = start.elapsed();
    let total = num_games as f64;
    println!(
        "{}: A={:.1}% B={:.1}% Tie={:.1}% ({} games in {:.1?})",
        label,
        result.player_a_wins as f64 / total * 100.0,
        result.player_b_wins as f64 / total * 100.0,
        result.ties as f64 / total * 100.0,
        num_games,
        elapsed,
    );

    result
}
