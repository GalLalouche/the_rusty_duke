use std::collections::HashMap;

use rand::Rng;

use crate::common::board::Board;
use crate::game::offset::Coordinate;
use crate::assert_not;

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum CurrentSide {
    Initial,
    Flipped,
}

impl CurrentSide {
    pub fn flip(&self) -> CurrentSide {
        match self {
            CurrentSide::Initial => CurrentSide::Flipped,
            CurrentSide::Flipped => CurrentSide::Initial,
        }
    }
}


#[derive(PartialEq, Eq, Clone, Copy)]
pub enum TokenAction {
    Move,
    Jump,
    Slide,
    Command,
    JumpSlide,
    Strike,
}

#[derive(Clone)]
pub struct TokenSide {
    board: Board<TokenAction>,
}

impl TokenSide {
    const SIDE: usize = 5;

    pub(in crate::game) fn new(map: HashMap<Coordinate, TokenAction>) -> TokenSide {
        for (c, a) in &map {
            match a {
                TokenAction::Jump =>
                    assert_not!(c.near_center(), "Jumps near the center should be moves"),
                TokenAction::Slide =>
                    assert!(c.near_center(), "Slides should be near the center"),
                TokenAction::JumpSlide =>
                    assert!(c.near_center(), "Jump slides should be near the center (?)"),
                TokenAction::Move =>
                    assert!(c.is_linear_from_center(), "Moves can't be L shaped"),
                // All combinations are valid.
                TokenAction::Strike => {}
                TokenAction::Command => {}
            }
        }
        let mut res = TokenSide { board: Board::square(TokenSide::SIDE) };
        for (k, v) in map {
            let result = res.board.put(k.into(), v);
            assert!(result.is_none());
        }
        res
    }

    pub fn actions(&self) -> Vec<(Coordinate, &TokenAction)> {
        self.board.active_coordinates()
            .iter()
            .map(|e| (e.0.into(), e.1))
            .collect()
    }
}

#[derive(Clone)]
pub struct GameToken {
    pub side_a: TokenSide,
    pub side_b: TokenSide,
    pub current_side: CurrentSide,
}

impl GameToken {
    pub fn new(side_a: TokenSide, side_b: TokenSide) -> GameToken {
        GameToken {
            side_a,
            side_b,
            current_side: CurrentSide::Initial,
        }
    }
    pub fn flip(&mut self) -> () {
        self.current_side = self.current_side.flip()
    }
    pub fn get_current_side(&self) -> &TokenSide {
        match self.current_side {
            CurrentSide::Initial => &self.side_a,
            CurrentSide::Flipped => &self.side_b,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Owner {
    Player1,
    Player2,
}

impl Owner {
    fn same_team(&self, other: &Self) -> bool {
        self == other
    }
    fn difference_team(&self, other: &Self) -> bool {
        self != other
    }
}

#[derive(Clone)]
pub struct OwnedToken {
    pub token: GameToken,
    pub owner: Owner,
}

pub trait Ownership {
    fn same_team(&self, other: &Self) -> bool;
    fn different_team(&self, other: &Self) -> bool {
        !self.same_team(other)
    }
}

impl Ownership for &Owner {
    fn same_team(&self, other: &Self) -> bool {
        self == other
    }
}

impl Ownership for OwnedToken {
    fn same_team(&self, other: &Self) -> bool {
        self.owner == other.owner
    }
}


#[derive(Clone)]
pub struct TokenBag {
    bag: Vec<GameToken>,
}

impl TokenBag {
    pub fn new(bag: Vec<GameToken>) -> TokenBag {
        TokenBag { bag }
    }

    pub fn pull(&mut self) -> Option<GameToken> {
        if self.bag.is_empty() {
            None
        } else {
            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..self.bag.len());
            Some(self.bag.remove(index))
        }
    }

    pub fn remaining(&self) -> &Vec<GameToken> {
        &self.bag
    }
}

#[derive(Clone)]
pub struct DiscardBag {
    bag: Vec<GameToken>,
}

impl DiscardBag {
    pub fn empty() -> DiscardBag {
        DiscardBag { bag: Vec::new() }
    }

    pub fn add(&mut self, t: GameToken) -> () {
        self.bag.push(t);
    }

    pub fn existing(&self) -> &Vec<GameToken> {
        &self.bag
    }
}