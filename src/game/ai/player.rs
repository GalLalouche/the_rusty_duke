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
pub enum AiMove {
    // FIXME Pulling is random, but the library doesn't suppose that stuff yet...
    // so just take the random value pulled.
    PullTileFormBagAndPlay(DukeOffset, Owner),
    ApplyNonCommandTileAction { src: Coordinates, dst: Coordinates, capturing: Option<PlacedTile> },
    Sentinel,
}

impl AiMove {
    pub fn to_game_move(&self) -> Option<GameMove> {
        match self {
            AiMove::PullTileFormBagAndPlay(o, _) => Some(GameMove::PullAndPlay(*o)),
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
}

impl Into<AiMove> for &PossibleMove {
    fn into(self) -> AiMove {
        match self {
            PossibleMove::PlaceNewTile(o, owner) =>
                AiMove::PullTileFormBagAndPlay(*o, *owner),
            PossibleMove::ApplyNonCommandTileAction { src, dst, capturing } =>
                AiMove::ApplyNonCommandTileAction { src: *src, dst: *dst, capturing: capturing.clone() },
        }
    }
}