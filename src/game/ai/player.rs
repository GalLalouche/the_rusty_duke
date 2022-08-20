use std::borrow::Borrow;
use std::convert::{TryFrom, TryInto};
use std::fmt::{Display, Formatter};

use rand::Rng;

use crate::common::coordinates::Coordinates;
use crate::game::board::{DukeOffset, PossibleMove};
use crate::game::state::{GameMove, GameState};
use crate::game::tile::{Owner, PlacedTile};

pub trait ArtificialPlayer {
    fn play_next_move<R: Rng>(&self, rng: &mut R, gs: &mut GameState) -> PossibleMove {
        let mv = self.get_next_move(rng, gs);
        gs.make_a_move(mv.borrow().try_into().unwrap(), rng);
        mv.to_undo_move().expect("AI moved should have been playable")
    }
    fn get_next_move<R: Rng>(&self, rng: &mut R, gs: &GameState) -> AiMove;
}

pub trait EvaluatingPlayer {
    fn evaluate(&self, gs: &GameState) -> f64;
    // A faster version of the above, that might not be entirely accurate.
    // For example, it might consider illegal moves.
    fn cheap_evaluate(&self, gs: &GameState) -> f64;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AiMove {
    PullTileFormBagAndPlay(DukeOffset, Owner),
    ApplyNonCommandTileAction { src: Coordinates, dst: Coordinates, capturing: Option<PlacedTile> },
    Sentinel,
}

impl Display for AiMove {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AiMove::Sentinel | AiMove::PullTileFormBagAndPlay { .. } => write!(f, "{:?}", self),
            AiMove::ApplyNonCommandTileAction { src, dst, capturing } =>
                write!(f, "ApplyNonCommandTileAction {{ src: {:?}, dst: {:?}{}}}",
                       src,
                       dst,
                       match &capturing {
                           None => "".to_owned(),
                           Some(t) => format!("capturing: {}", t.tile.get_name()),
                       }
                )
        }
    }
}

impl TryFrom<&AiMove> for GameMove {
    type Error = ();

    fn try_from(value: &AiMove) -> Result<Self, Self::Error> {
        match value {
            AiMove::PullTileFormBagAndPlay(o, _) => Ok(GameMove::PullAndPlay(*o)),
            AiMove::ApplyNonCommandTileAction { src, dst, .. } =>
                Ok(GameMove::ApplyNonCommandTileAction {
                    src: *src,
                    dst: *dst,
                }),
            AiMove::Sentinel => Err(()),
        }
    }
}

impl AiMove {
    pub fn to_undo_move(&self) -> Option<PossibleMove> {
        match self {
            AiMove::PullTileFormBagAndPlay(o, owner) => Some(PossibleMove::PlaceNewTile(*o, *owner)),
            AiMove::ApplyNonCommandTileAction { src, dst, capturing } =>
                Some(PossibleMove::ApplyNonCommandTileAction {
                    src: *src,
                    dst: *dst,
                    capturing: capturing.clone(),
                }),
            AiMove::Sentinel => None,
        }
    }


    pub fn play<R: Rng>(&self, state: &mut GameState, rng: &mut R) {
        match &self {
            AiMove::PullTileFormBagAndPlay(o, _) =>
                state.make_a_move(GameMove::PullAndPlay(*o), rng),
            AiMove::ApplyNonCommandTileAction { src, dst, .. } =>
                state.make_a_move(GameMove::ApplyNonCommandTileAction {
                    src: *src,
                    dst: *dst,
                }, rng),
            AiMove::Sentinel => {}
        }
    }

    pub fn all_moves(gs: &GameState) -> impl Iterator<Item=AiMove> + '_ {
        gs.all_valid_game_moves_for_current_player().map(|e| e.borrow().into())
    }
}

impl From<&PossibleMove> for AiMove {
    fn from(pm: &PossibleMove) -> Self {
        match pm {
            PossibleMove::PlaceNewTile(o, owner) =>
                AiMove::PullTileFormBagAndPlay(*o, *owner),
            PossibleMove::ApplyNonCommandTileAction { src, dst, capturing } =>
                AiMove::ApplyNonCommandTileAction { src: *src, dst: *dst, capturing: capturing.clone() },
        }
    }
}