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

    /// First character of the tile name, looked up via a const table
    /// instead of going through the string conversion.
    #[inline]
    pub fn first_char(self) -> char {
        const CHARS: [char; 13] = [
            'D', // Duke
            'F', // Footman
            'P', // Pikeman
            'K', // Knight
            'C', // Champion
            'D', // Dragoon
            'W', // Wizard
            'G', // General
            'M', // Marshall
            'A', // Assassin
            'P', // Priest
            'B', // Bowman
            'L', // Longbowman
        ];
        CHARS[self as usize]
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
        let c = self.tile_type.first_char();
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

    // ── TileType::is_duke ─────────────────────────────────────────────
    #[test]
    fn is_duke_returns_true_for_duke() {
        assert!(TileType::Duke.is_duke());
    }

    #[test]
    fn is_duke_returns_false_for_non_duke() {
        assert!(!TileType::Footman.is_duke());
        assert!(!TileType::Knight.is_duke());
        assert!(!TileType::Champion.is_duke());
    }

    // ── PlacedTile construction and fields ────────────────────────────
    #[test]
    fn placed_tile_new_sets_initial_side() {
        let pt = PlacedTile::new(Owner::TopPlayer, TileType::Footman);
        assert_eq!(pt.tile_type, TileType::Footman);
        assert_eq!(pt.owner, Owner::TopPlayer);
        assert_eq!(pt.current_side, CurrentSide::Initial);
    }

    #[test]
    fn placed_tile_new_bottom_player() {
        let pt = PlacedTile::new(Owner::BottomPlayer, TileType::Duke);
        assert_eq!(pt.owner, Owner::BottomPlayer);
        assert_eq!(pt.tile_type, TileType::Duke);
        assert_eq!(pt.current_side, CurrentSide::Initial);
    }

    // ── PlacedTile flip ───────────────────────────────────────────────
    #[test]
    fn placed_tile_flip_changes_to_flipped() {
        let mut pt = PlacedTile::new(Owner::TopPlayer, TileType::Footman);
        assert_eq!(pt.current_side, CurrentSide::Initial);
        pt.flip();
        assert_eq!(pt.current_side, CurrentSide::Flipped);
    }

    #[test]
    fn placed_tile_flip_twice_returns_to_initial() {
        let mut pt = PlacedTile::new(Owner::TopPlayer, TileType::Footman);
        pt.flip();
        pt.flip();
        assert_eq!(pt.current_side, CurrentSide::Initial);
    }

    // ── CurrentSide flip ──────────────────────────────────────────────
    #[test]
    fn current_side_flip_initial_to_flipped() {
        assert_eq!(CurrentSide::Initial.flip(), CurrentSide::Flipped);
    }

    #[test]
    fn current_side_flip_flipped_to_initial() {
        assert_eq!(CurrentSide::Flipped.flip(), CurrentSide::Initial);
    }

    // ── Owner ─────────────────────────────────────────────────────────
    #[test]
    fn owner_next_player_top_to_bottom() {
        assert_eq!(Owner::TopPlayer.next_player(), Owner::BottomPlayer);
    }

    #[test]
    fn owner_next_player_bottom_to_top() {
        assert_eq!(Owner::BottomPlayer.next_player(), Owner::TopPlayer);
    }

    #[test]
    fn owner_next_player_twice_returns_same() {
        assert_eq!(Owner::TopPlayer.next_player().next_player(), Owner::TopPlayer);
    }

    // ── Ownership trait ───────────────────────────────────────────────
    #[test]
    fn same_team_for_same_owner() {
        assert!(Owner::TopPlayer.same_team(&Owner::TopPlayer));
        assert!(Owner::BottomPlayer.same_team(&Owner::BottomPlayer));
    }

    #[test]
    fn different_team_for_different_owners() {
        assert!(Owner::TopPlayer.different_team(&Owner::BottomPlayer));
        assert!(Owner::BottomPlayer.different_team(&Owner::TopPlayer));
    }

    #[test]
    fn placed_tile_same_team_checks_owner() {
        let top1 = PlacedTile::new(Owner::TopPlayer, TileType::Footman);
        let top2 = PlacedTile::new(Owner::TopPlayer, TileType::Knight);
        let bot1 = PlacedTile::new(Owner::BottomPlayer, TileType::Footman);
        assert!((&top1).same_team(&&top2));
        assert!((&top1).different_team(&&bot1));
    }

    // ── PlacedTile::get_current_side ──────────────────────────────────
    #[test]
    fn get_current_side_returns_side_a_for_initial() {
        let pt = PlacedTile::new(Owner::BottomPlayer, TileType::Footman);
        let side = pt.get_current_side();
        // The tile should have a Unit action on initial side
        let has_unit = side.actions().iter().any(|(_, a)| *a == crate::game::tile_side::TileAction::Unit);
        assert!(has_unit);
    }

    #[test]
    fn get_current_side_changes_after_flip() {
        let mut pt = PlacedTile::new(Owner::BottomPlayer, TileType::Footman);
        let side_a_actions: Vec<_> = pt.get_current_side().actions().clone();
        pt.flip();
        let side_b_actions: Vec<_> = pt.get_current_side().actions().clone();
        // side_a and side_b should be different for Footman
        assert_ne!(side_a_actions, side_b_actions);
    }

    // ── TileType::get_name ────────────────────────────────────────────
    #[test]
    fn get_name_returns_display_name() {
        assert_eq!(TileType::Duke.get_name(), "Duke");
        assert_eq!(TileType::Footman.get_name(), "Footman");
        assert_eq!(TileType::Longbowman.get_name(), "Longbowman");
    }
}
