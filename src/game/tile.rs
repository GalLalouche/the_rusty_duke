use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use crate::common::coordinates::Coordinates;
use crate::game::tile_side::{TileAction, TileSide};

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum CurrentSide {
    Initial,
    Flipped,
}

#[derive(Debug)]
pub struct Tile {
    side_a: TileSide,
    side_b: TileSide,
    name: String,
}

impl PartialEq for Tile {
    fn eq(&self, other: &Self) -> bool {
        self.name_compare(other)
    }
}

impl Eq for Tile {}

impl Hash for Tile {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}

impl Tile {
    pub fn get_side_a(&self) -> &TileSide {
        &self.side_a
    }

    pub fn get_side_b(&self) -> &TileSide {
        &self.side_b
    }

    pub fn get_name(&self) -> &String {
        &self.name
    }

    pub fn name_compare(&self, other: &Tile) -> bool {
        self.name == other.name
    }
    pub fn new(side_a: TileSide, side_b: TileSide, name: &str) -> Tile {
        Tile {
            side_a,
            side_b,
            name: name.to_owned(),
        }
    }

    pub(super) fn flip_vertical(&self) -> Tile {
        Tile {
            side_a: self.side_a.flip_vertical(),
            side_b: self.side_b.flip_vertical(),
            name: self.name.clone(),
        }
    }
}

// #[derive(Debug, PartialEq, Eq, Clone)]
// pub struct TileRef {
//     pub tile: Rc<Tile>,
// }

pub type TileRef = Rc<Tile>;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum Owner {
    TopPlayer,
    BottomPlayer,
}

impl Display for Owner {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Owner {
    pub fn next_player(self) -> Owner {
        match self {
            Owner::TopPlayer => Owner::BottomPlayer,
            Owner::BottomPlayer => Owner::TopPlayer,
        }
    }
}

impl CurrentSide {
    pub fn flip(&self) -> CurrentSide {
        match self {
            CurrentSide::Initial => CurrentSide::Flipped,
            CurrentSide::Flipped => CurrentSide::Initial,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct PlacedTile {
    pub tile: TileRef,
    // TODO: this should be private
    pub current_side: CurrentSide,
    pub owner: Owner,
}

impl Clone for PlacedTile {
    fn clone(&self) -> Self {
        PlacedTile {
            tile: self.tile.clone(),
            current_side: self.current_side,
            owner: self.owner,
        }
    }
}

impl PlacedTile {
    pub fn new(owner: Owner, tile: Tile) -> PlacedTile {
        let maybe_flipped_tile = match owner {
            Owner::TopPlayer => tile.flip_vertical(),
            Owner::BottomPlayer => tile,
        };
        PlacedTile { owner, tile: Rc::new(maybe_flipped_tile), current_side: CurrentSide::Initial }
    }
    pub fn new_from_ref(owner: Owner, tile: TileRef) -> PlacedTile {
        let maybe_flipped_tile = match owner {
            Owner::TopPlayer => Rc::new(tile.flip_vertical()),
            Owner::BottomPlayer => tile,
        };
        PlacedTile { owner, tile: maybe_flipped_tile, current_side: CurrentSide::Initial }
    }
    pub fn get_current_side(&self) -> &TileSide {
        match self.current_side {
            CurrentSide::Initial => &self.tile.side_a,
            CurrentSide::Flipped => &self.tile.side_b,
        }
    }
    pub fn flip(&mut self) -> () {
        self.current_side = self.current_side.flip();
    }
    pub fn single_char_token(&self) -> char {
        let c = self.tile.name.chars().next().unwrap();
        match self.current_side {
            CurrentSide::Initial => c.to_ascii_lowercase(),
            CurrentSide::Flipped => c.to_ascii_uppercase(),
        }
    }
    pub fn get_action_from_coordinates(&self, src: Coordinates, dst: Coordinates) -> Option<TileAction> {
        self.get_current_side().get_action_from_coordinates(src, dst)
    }
}

pub trait Ownership: Sized {
    fn same_team(self, other: Self) -> bool;
    fn different_team(self, other: Self) -> bool {
        !self.same_team(other)
    }
}

impl Ownership for Owner {
    fn same_team(self, other: Self) -> bool {
        self == other
    }
}

impl Ownership for &PlacedTile {
    fn same_team(self, other: Self) -> bool {
        self.owner == other.owner
    }
}
