use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};

use rand::Rng;

use crate::assert_not;
use crate::common::board::Board;
use crate::game::offset::{Offsetable, Offsets};

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


#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum TileAction {
    Move,
    Jump,
    Slide,
    Command,
    JumpSlide,
    Strike,
}

#[derive(Debug, Clone)]
pub struct TileSide {
    board: Board<TileAction>,
}

impl TileSide {
    pub const SIDE: u16 = 5;

    pub(in crate::game) fn new(
        vec: Vec<(&dyn Offsetable, TileAction)>) -> TileSide {
        let map: HashMap<Offsets, TileAction> = vec
            .iter()
            .flat_map(|(tso, ta)|
                tso
                    .offsets()
                    .iter()
                    .map(|o| (*o, *ta))
                    .collect::<Vec<_>>()
            ).collect();
        TileSide::verify_actions(&map);
        TileSide::verify_no_illegal_repeats(&map);
        let mut res = TileSide { board: Board::square(TileSide::SIDE) };
        for (k, v) in map.borrow() {
            let result = res.board.put((*k).into(), *v);
            assert!(result.is_none());
        }
        res
    }

    fn verify_actions(map: &HashMap<Offsets, TileAction>) -> () {
        for (c, a) in map.borrow() {
            match a {
                TileAction::Jump =>
                    assert_not!(c.near_center(), "Jumps near the center should be moves"),
                TileAction::Slide =>
                    assert!(c.near_center(), "Slides should be near the center"),
                TileAction::JumpSlide =>
                    assert_not!(c.near_center(), "Jump slides not should be near the center"),
                TileAction::Move =>
                    assert!(c.is_linear_from_center(), "Moves can't be L shaped"),
                // All combinations are valid.
                TileAction::Strike => {}
                TileAction::Command => {}
            }
        }
    }

    fn verify_no_illegal_repeats(map: &HashMap<Offsets, TileAction>) -> () {
        let mut commands = HashSet::new();
        let mut non_command_actions = HashSet::new();
        let mut unit_icon = 0; // TODO: unused for now

        for (c, a) in map.borrow() {
            match a {
                TileAction::Command => {
                    assert!(!commands.contains(c), "Command already exists for {:?}", c);
                    commands.insert(c);
                }
                _ => {
                    assert_not!(commands.contains(c), "Non-Command already exists for {:?}", c);
                    non_command_actions.insert(c);
                }
            }
        }
    }

    pub fn actions(&self) -> Vec<(Offsets, &TileAction)> {
        self.board.active_coordinates()
            .iter()
            .map(|e| (e.0.into(), e.1))
            .collect()
    }

    pub fn get_board(&self) -> &Board<TileAction> {
        &self.board
    }
}

#[derive(Clone)]
pub struct Tile {
    pub side_a: TileSide,
    pub side_b: TileSide,
    pub current_side: CurrentSide,
    pub name: String,
}

impl Tile {
    pub fn new(side_a: TileSide, side_b: TileSide, name: &str) -> Tile {
        Tile {
            side_a,
            side_b,
            current_side: CurrentSide::Initial,
            name: name.to_owned(),
        }
    }
    pub fn flip(&mut self) -> () {
        self.current_side = self.current_side.flip()
    }
    pub fn get_current_side(&self) -> &TileSide {
        match self.current_side {
            CurrentSide::Initial => &self.side_a,
            CurrentSide::Flipped => &self.side_b,
        }
    }
    pub fn single_char_token(&self) -> char {
        let c = self.name.chars().next().unwrap();
        match self.current_side {
            CurrentSide::Initial => c.to_ascii_lowercase(),
            CurrentSide::Flipped => c.to_ascii_uppercase(),
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Owner {
    Player1,
    Player2,
}

#[derive(Clone)]
pub struct OwnedTile {
    pub tile: Tile,
    pub owner: Owner,
}

impl OwnedTile {
    pub fn single_char_token(&self) -> char {
        self.tile.single_char_token()
    }
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

impl Ownership for OwnedTile {
    fn same_team(&self, other: &Self) -> bool {
        self.owner == other.owner
    }
}


#[derive(Clone)]
pub struct TileBag {
    bag: Vec<Tile>,
}

impl TileBag {
    pub fn new(bag: Vec<Tile>) -> TileBag {
        TileBag { bag }
    }

    pub fn pull(&mut self) -> Option<Tile> {
        if self.bag.is_empty() {
            None
        } else {
            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..self.bag.len());
            Some(self.bag.remove(index))
        }
    }

    pub fn remaining(&self) -> &Vec<Tile> {
        &self.bag
    }
}

#[derive(Clone)]
pub struct DiscardBag {
    bag: Vec<Tile>,
}

impl DiscardBag {
    pub fn empty() -> DiscardBag {
        DiscardBag { bag: Vec::new() }
    }

    pub fn add(&mut self, t: Tile) -> () {
        self.bag.push(t);
    }

    pub fn existing(&self) -> &Vec<Tile> {
        &self.bag
    }
}
