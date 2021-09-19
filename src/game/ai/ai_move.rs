use minimax_alpha_beta::strategy::Strategy;
use rand::Rng;

use crate::common::coordinates::Coordinates;
use crate::game::board::{DukeOffset, PossibleMove};
use crate::game::state::{GameMove, GameState};
use crate::game::tile::{Owner, PlacedTile};

pub trait ArtificialPlayer {
    fn play_next_move<R>(&self, rng: &mut R, gs: &mut GameState) -> () where R: Rng;
}

pub trait EvaluatingPlayer {
    fn evaluate(&self, gs: &GameState) -> f64;
}

pub(super) struct ArtificialStrategy<'a> {
    pub state: GameState,
    pub evaluator: Box<&'a dyn EvaluatingPlayer>,
}


#[derive(Debug, Clone)]
pub(super) enum AiMove {
    // FIXME Pulling is random, but the library doesn't suppose that stuff yet...
    // so just take the random value pulled.
    PullTileFormBagAndPlay(DukeOffset),
    ApplyNonCommandTileAction { src: Coordinates, dst: Coordinates, capturing: Option<PlacedTile> },
    Sentinel,
}

impl AiMove {
    pub fn to_game_move(&self) -> Option<GameMove> {
        match self {
            AiMove::PullTileFormBagAndPlay(o) => Some(GameMove::PullAndPlay(*o)),
            AiMove::ApplyNonCommandTileAction { src, dst, .. } =>
                Some(GameMove::ApplyNonCommandTileAction {
                    src: *src,
                    dst: *dst,
                }),
            AiMove::Sentinel => None,
        }
    }

    pub fn to_undo_move(&self) -> Option<PossibleMove> {
        match self {
            AiMove::PullTileFormBagAndPlay(o) => Some(PossibleMove::PlaceNewTile(*o)),
            AiMove::ApplyNonCommandTileAction { src, dst, capturing } =>
                Some(PossibleMove::ApplyNonCommandTileAction {
                    src: *src,
                    dst: *dst,
                    capturing: capturing.clone(),
                }),
            AiMove::Sentinel => None,
        }
    }
}

impl Into<AiMove> for &PossibleMove {
    fn into(self) -> AiMove {
        match self {
            PossibleMove::PlaceNewTile(o) =>
                AiMove::PullTileFormBagAndPlay(*o),
            PossibleMove::ApplyNonCommandTileAction { src, dst, capturing } =>
                AiMove::ApplyNonCommandTileAction { src: *src, dst: *dst, capturing: capturing.clone() },
        }
    }
}

impl<'a> Strategy for ArtificialStrategy<'a> {
    type Player = Owner;
    // A hack to make undoing a bit easier
    type Move = AiMove;
    type Board = GameState;

    fn evaluate(&self) -> f64 {
        self.evaluator.evaluate(self.get_board())
    }

    fn get_winner(&self) -> Self::Player {
        self.state.winner().expect("I think(?!) this shouldn't be called if there's no winner")
    }

    fn is_game_tied(&self) -> bool {
        false
    }

    fn is_game_complete(&self) -> bool {
        self.state.is_over()
    }

    fn get_available_moves(&self) -> Vec<Self::Move> {
        self.state.all_valid_game_moves_for_current_player().iter().map(|e| e.into()).collect()
    }

    fn play(&mut self, mv: &Self::Move, maximizer: bool) {
        match &mv {
            AiMove::PullTileFormBagAndPlay(o) =>
                self.state.make_a_move(GameMove::PullAndPlay(*o)),
            AiMove::ApplyNonCommandTileAction { src, dst, .. } =>
                self.state.make_a_move(GameMove::ApplyNonCommandTileAction {
                    src: *src,
                    dst: *dst,
                }),
            AiMove::Sentinel => {}
        }
    }

    fn clear(&mut self, mv: &Self::Move) {
        if let Some(um) = mv.to_undo_move() {
            self.state.undo(um)
        }
    }

    fn get_board(&self) -> &Self::Board {
        &self.state
    }

    fn is_a_valid_move(&self, _mv: &Self::Move) -> bool {
        true // get_available moves already filters this
    }

    fn get_a_sentinel_move(&self) -> Self::Move {
        AiMove::Sentinel
    }
}

