use std::convert::TryFrom;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

use crate::common::coordinates::Coordinates;
use crate::game::tile_side::{TileAction, TileSide};

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum CurrentSide {
    Initial,
    Flipped,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash,
    strum_macros::EnumCount, strum_macros::IntoStaticStr, strum_macros::Display)]
#[repr(u8)]
pub enum TileType {
    Duke = 0,
    Footman = 1,
    Pikeman = 2,
    Knight = 3,
    Champion = 4,
    Dragoon = 5,
    Wizard = 6,
    General = 7,
    Marshall = 8,
    Assassin = 9,
    Priest = 10,
    Bowman = 11,
    Longbowman = 12,
}

impl TileType {
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub fn is_duke(self) -> bool {
        self == TileType::Duke
    }

    #[inline]
    pub fn get_name(self) -> &'static str {
        self.into()
    }
}

impl TryFrom<u8> for TileType {
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(TileType::Duke),
            1 => Ok(TileType::Footman),
            2 => Ok(TileType::Pikeman),
            3 => Ok(TileType::Knight),
            4 => Ok(TileType::Champion),
            5 => Ok(TileType::Dragoon),
            6 => Ok(TileType::Wizard),
            7 => Ok(TileType::General),
            8 => Ok(TileType::Marshall),
            9 => Ok(TileType::Assassin),
            10 => Ok(TileType::Priest),
            11 => Ok(TileType::Bowman),
            12 => Ok(TileType::Longbowman),
            other => Err(other),
        }
    }
}

#[derive(Debug)]
pub struct Tile {
    side_a: TileSide,
    side_b: TileSide,
    tile_type: TileType,
}

impl PartialEq for Tile {
    fn eq(&self, other: &Self) -> bool {
        self.tile_type == other.tile_type
    }
}

impl Eq for Tile {}

impl Hash for Tile {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tile_type.hash(state)
    }
}

impl Tile {
    pub fn get_side_a(&self) -> &TileSide {
        &self.side_a
    }

    pub fn get_side_b(&self) -> &TileSide {
        &self.side_b
    }

    pub fn tile_type(&self) -> TileType {
        self.tile_type
    }

    pub fn get_name(&self) -> &'static str {
        self.tile_type.into()
    }

    pub fn new(side_a: TileSide, side_b: TileSide, tile_type: TileType) -> Tile {
        Tile {
            side_a,
            side_b,
            tile_type,
        }
    }

    pub(super) fn flip_vertical(&self) -> Tile {
        Tile {
            side_a: self.side_a.flip_vertical(),
            side_b: self.side_b.flip_vertical(),
            tile_type: self.tile_type,
        }
    }
}

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

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct PlacedTile {
    pub tile_type: TileType,
    pub current_side: CurrentSide,
    pub owner: Owner,
}

impl Display for PlacedTile {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({}, {:?})", self.tile_type, self.owner, self.current_side)
    }
}

impl PlacedTile {
    pub fn new(owner: Owner, tile_type: TileType) -> PlacedTile {
        PlacedTile { owner, tile_type, current_side: CurrentSide::Initial }
    }
    /// Look up the full Tile data from the static tile table.
    #[inline]
    pub fn tile(&self) -> &'static Tile {
        crate::game::units::tile_table_lookup(self.tile_type, self.owner)
    }
    pub fn get_current_side(&self) -> &TileSide {
        let tile = self.tile();
        match self.current_side {
            CurrentSide::Initial => &tile.side_a,
            CurrentSide::Flipped => &tile.side_b,
        }
    }
    pub fn flip(&mut self) -> () {
        self.current_side = self.current_side.flip();
    }
    pub fn single_char_token(&self) -> char {
        let c = self.tile_type.get_name().chars().next().unwrap();
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
    fn same_team(&self, other: &Self) -> bool;
    fn different_team(&self, other: &Self) -> bool {
        !self.same_team(other)
    }
}

impl Ownership for Owner {
    fn same_team(&self, other: &Self) -> bool {
        self == other
    }
}

impl Ownership for &PlacedTile {
    fn same_team(&self, other: &Self) -> bool {
        self.owner == other.owner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strum::EnumCount;

    #[test]
    fn tiletype_roundtrips_through_index_and_try_from() {
        for i in 0..TileType::COUNT {
            let tt = TileType::try_from(i as u8).expect(&format!("TryFrom failed for {}", i));
            assert_eq!(tt.index(), i, "index() mismatch for variant {:?}", tt);
        }
    }

    #[test]
    fn tiletype_try_from_succeeds_for_all_valid_values() {
        for i in 0u8..TileType::COUNT as u8 {
            assert!(
                TileType::try_from(i).is_ok(),
                "TryFrom should succeed for value {}",
                i,
            );
        }
    }

    #[test]
    fn tiletype_try_from_fails_for_out_of_range_values() {
        for i in TileType::COUNT as u8..=255 {
            assert_eq!(
                TileType::try_from(i),
                Err(i),
                "TryFrom should fail for value {}",
                i,
            );
        }
    }
}
