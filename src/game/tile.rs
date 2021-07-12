use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};

use rand::Rng;

use crate::assert_not;
use crate::common::board::Board;
use crate::common::coordinates::Coordinates;
use crate::common::utils::Folding;
use crate::game::offset::{Centerable, HorizontalOffset, Indexable, Offsetable, Offsets, VerticalOffset};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CurrentSide {
    Initial,
    Flipped,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum TileAction {
    Unit,
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
        let mut map: HashMap<Offsets, TileAction> = vec
            .iter()
            .flat_map(|(tso, ta)|
                tso
                    .offsets()
                    .iter()
                    .map(|o| (*o, *ta))
                    .collect::<Vec<_>>()
            ).collect();
        if map.iter().all(|(_, v)| *v != TileAction::Unit) {
            map.insert(Offsets::center(), TileAction::Unit);
        }
        TileSide::verify_no_illegal_repeats(&map);
        TileSide::verify_actions(&map);
        let mut res = TileSide { board: Board::square(TileSide::SIDE) };
        for (k, v) in map.borrow() {
            let result = res.board.put(&k.into(), *v);
            assert!(result.is_none());
        }
        res
    }

    // TODO: Verify that there is nothing after slides
    fn verify_actions(map: &HashMap<Offsets, TileAction>) -> () {
        let center =
            map.iter().find(|(_, a)| **a == TileAction::Unit).expect("No Unit action found").0;
        let is_near_center = |o: &Offsets| {
            o.is_near(center)
        };
        let is_linear_from_center = |o: &Offsets| {
            o.is_linear_from(center)
        };
        for (c, a) in map.borrow() {
            match a {
                TileAction::Unit =>
                    assert!(c.x.is_centered(), "The tile should always be horizontally centered"),
                TileAction::Jump =>
                    assert_not!(is_near_center(c), "Jumps near the center should be moves"),
                TileAction::Slide => {
                    assert!(is_near_center(c), "Slides should be near the center");
                    let after_slide_offset = match (c.x, c.y) {
                        (HorizontalOffset::Center, VerticalOffset::Top) =>
                            VerticalOffset::FarTop.center(),
                        (HorizontalOffset::Center, VerticalOffset::Bottom) =>
                            VerticalOffset::FarBottom.center(),
                        (HorizontalOffset::Left, VerticalOffset::Center) =>
                            HorizontalOffset::FarLeft.center(),
                        (HorizontalOffset::Right, VerticalOffset::Center) =>
                            HorizontalOffset::FarRight.center(),
                        (HorizontalOffset::Left, VerticalOffset::Top) =>
                            Offsets::new(HorizontalOffset::FarLeft, VerticalOffset::FarTop),
                        (HorizontalOffset::Right, VerticalOffset::Top) =>
                            Offsets::new(HorizontalOffset::FarRight, VerticalOffset::FarTop),
                        (HorizontalOffset::Left, VerticalOffset::Bottom) =>
                            Offsets::new(HorizontalOffset::FarLeft, VerticalOffset::FarBottom),
                        (HorizontalOffset::Right, VerticalOffset::Bottom) =>
                            Offsets::new(HorizontalOffset::FarRight, VerticalOffset::FarBottom),
                        _ => panic!("Assertion Error: near center should already have been verified"),
                    };
                    // In theory, jumps and other operations can appear after a slide, but they don't.
                    assert_not!(map.contains_key(&after_slide_offset))
                }
                TileAction::JumpSlide =>
                    assert_not!(is_near_center(c), "Jump slides not should be near the center"),
                TileAction::Move =>
                    assert!(is_linear_from_center(c), "Moves can't be L shaped"),
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
                TileAction::Unit =>
                    unit_icon += 1,
                _ => {
                    assert_not!(commands.contains(c), "Non-Command already exists for {:?}", c);
                    non_command_actions.insert(c);
                }
            }
        }
        assert_eq!(unit_icon, 1, "Unit action should have been 1, was {}", unit_icon);
    }

    pub fn actions(&self) -> Vec<(Offsets, TileAction)> {
        self.board.active_coordinates()
            .iter()
            .map(|e| (e.0.borrow().into(), e.1.clone()))
            .collect()
    }

    pub fn get_board(&self) -> &Board<TileAction> {
        &self.board
    }

    pub fn center_offset(&self) -> VerticalOffset {
        let center_horizontal_offset = TileSide::SIDE / 2;
        for y in 0..5 {
            if self.board.get(&Coordinates { x: center_horizontal_offset, y }) == Some(&TileAction::Unit) {
                return VerticalOffset::from_index(y);
            };
        }
        panic!("No Unit action found in the center columns;\n{:?}", self);
    }

    fn has_horizontal_slide(&self) -> bool {
        let left_offset = &Offsets::new(HorizontalOffset::Left, self.center_offset());
        let right_offset = &Offsets::new(HorizontalOffset::Right, self.center_offset());
        assert_eq!(self.board.get(&left_offset.into()), self.board.get(&right_offset.into()));
        self.board.get(&left_offset.into()).has(&&TileAction::Slide)
    }

    /// Panics if dst is out of bounds.
    // TODO: Should this really panic?
    // TODO: Handle jump slides
    pub fn get_action_from_coordinates(&self, src: &Coordinates, dst: &Coordinates) -> Option<&TileAction> {
        let x_diff = i32::from(dst.x) - i32::from(src.x);
        let y_base = i32::from(self.center_offset().to_index() - 2);
        let y_diff = y_base + i32::from(dst.y) - i32::from(src.y);
        let x_offset =
            if y_diff == 0 && self.has_horizontal_slide() {
                assert!(self.board.get(&HorizontalOffset::FarLeft.center().borrow().into()).is_none());
                assert!(self.board.get(&HorizontalOffset::FarRight.center().borrow().into()).is_none());
                if x_diff > 0 { HorizontalOffset::Right } else { HorizontalOffset::Left }
            } else {
                match x_diff {
                    -2 => HorizontalOffset::FarLeft,
                    -1 => HorizontalOffset::Left,
                    0 => HorizontalOffset::Center,
                    1 => HorizontalOffset::Right,
                    2 => HorizontalOffset::FarRight,
                    _ => panic!("Out of bounds"),
                }
            };
        // TODO extract this out to a helper function as well
        let get_vertical_slide =
            (if y_diff > 0 {
                self.board.get(&VerticalOffset::Bottom.center().borrow().into())
            } else if y_diff < 0 {
                self.board.get(&VerticalOffset::Top.center().borrow().into())
            } else {
                None
            }).filter(|a| *a == &TileAction::Slide);

        let y_offset = get_vertical_slide
            .filter(|_| x_diff == 0)
            .map(|_| if y_diff > 0 { VerticalOffset::Bottom } else { VerticalOffset::Top })
            .unwrap_or_else(|| match y_diff {
                -2 => VerticalOffset::FarTop,
                -1 => VerticalOffset::Top,
                0 => VerticalOffset::Center,
                1 => VerticalOffset::Bottom,
                2 => VerticalOffset::FarBottom,
                _ => panic!("Out of bounds"),
            });
        self.board.get(&Offsets::new(x_offset, y_offset).borrow().into())
    }

    pub fn flip_vertical(self) -> TileSide {
        TileSide {
            board: self.board.flip_vertical()
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tile {
    pub side_a: TileSide,
    pub side_b: TileSide,
    pub name: String,
}

impl Tile {
    pub fn new(side_a: TileSide, side_b: TileSide, name: &str) -> Tile {
        Tile {
            side_a,
            side_b,
            name: name.to_owned(),
        }
    }

    pub fn flip_vertical(self) -> Tile {
        Tile {
            side_a: self.side_a.flip_vertical(),
            side_b: self.side_b.flip_vertical(),
            name: self.name.clone(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Owner {
    TopPlayer,
    BottomPlayer,
}

impl Owner {
    pub fn next_player(&self) -> Owner {
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

#[derive(Debug, Clone)]
pub struct PlacedTile {
    pub tile: Tile,
    // TODO: this should be private
    pub current_side: CurrentSide,
    pub owner: Owner,
}

impl PlacedTile {
    pub fn new(owner: Owner, tile: Tile) -> PlacedTile {
        let maybe_flipped_tile = match owner {
            Owner::TopPlayer => tile.flip_vertical(),
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
    pub fn get_action_from_coordinates(&self, src: &Coordinates, dst: &Coordinates) -> Option<&TileAction> {
        self.get_current_side().get_action_from_coordinates(src, dst)
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

impl Ownership for PlacedTile {
    fn same_team(&self, other: &Self) -> bool {
        self.owner == other.owner
    }
}

#[derive(Debug, Clone)]
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
    pub fn is_empty(&self) -> bool {
        self.remaining().len() == 0
    }
    pub fn non_empty(&self) -> bool {
        !self.is_empty()
    }
}

#[derive(Debug, Clone)]
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

#[cfg(test)]
mod test {
    use crate::*;
    use crate::game::offset::{FourWaySymmetric, HorizontalSymmetricOffset};

    use super::*;

    #[test]
    #[should_panic]
    fn get_action_panics_if_dst_if_out_of_bounds() {
        let tile = TileSide::new(vec![
            (&FourWaySymmetric::NearStraight, TileAction::Move)
        ]);
        tile.get_action_from_coordinates(&Coordinates { x: 0, y: 0 }, &Coordinates { x: 0, y: 3 });
    }

    #[test]
    fn get_action_returns_none_if_no_action() {
        let tile = TileSide::new(vec![
            (&FourWaySymmetric::NearStraight, TileAction::Move)
        ]);
        assert_none!(tile.get_action_from_coordinates(&Coordinates{x: 2, y:4}, &Coordinates{x: 3, y:5}))
    }

    #[test]
    fn get_action_returns_some_if_there_is_an_action() {
        let tile = TileSide::new(vec![
            (&(HorizontalSymmetricOffset::Far, VerticalOffset::Top), TileAction::Jump)
        ]);
        assert_some!(
            &TileAction::Jump,
            tile.get_action_from_coordinates(&Coordinates{x: 2, y:4}, &Coordinates{x: 4, y:3}),
        )
    }

    #[test]
    fn get_action_returns_some_if_there_is_an_action_non_vertical_center() {
        let tile = TileSide::new(vec![
            (&(VerticalOffset::FarTop), TileAction::Strike),
            (&(VerticalOffset::Bottom), TileAction::Unit),
        ]);
        assert_some!(
            &TileAction::Strike,
            tile.get_action_from_coordinates(&Coordinates{x: 2, y:4}, &Coordinates{x: 2, y:1}),
        )
    }

    #[test]
    fn get_action_returns_some_slides_if_there_is_a_horizontal_slide_action_1_spaces() {
        let tile = TileSide::new(vec![
            (&HorizontalSymmetricOffset::Near, TileAction::Slide)
        ]);
        assert_some!(
            &TileAction::Slide,
            tile.get_action_from_coordinates(&Coordinates{x: 0, y:2}, &Coordinates{x: 1, y:2}),
        )
    }

    #[test]
    fn get_action_returns_some_slides_if_there_is_a_horizontal_slide_action_2_spaces() {
        let tile = TileSide::new(vec![
            (&HorizontalSymmetricOffset::Near, TileAction::Slide)
        ]);
        assert_some!(
            &TileAction::Slide,
            tile.get_action_from_coordinates(&Coordinates{x: 0, y:2}, &Coordinates{x: 2, y:2}),
        )
    }

    #[test]
    fn get_action_returns_some_slides_if_there_is_a_horizontal_slide_mult_spaces_action() {
        let tile = TileSide::new(vec![
            (&HorizontalSymmetricOffset::Near, TileAction::Slide)
        ]);
        assert_some!(
            &TileAction::Slide,
            tile.get_action_from_coordinates(&Coordinates{x: 0, y:2}, &Coordinates{x: 5, y:2}),
        )
    }

    #[test]
    fn get_action_returns_some_slides_if_there_is_a_top_slide_action() {
        let tile = TileSide::new(vec![
            (&VerticalOffset::Top, TileAction::Slide)
        ]);
        assert_some!(
            &TileAction::Slide,
            tile.get_action_from_coordinates(&Coordinates{x: 0, y:5}, &Coordinates{x: 0, y:0}),
        )
    }

    #[test]
    fn get_action_returns_some_slides_if_there_is_a_top_slide_2_spaces_action() {
        let tile = TileSide::new(vec![
            (&VerticalOffset::Top, TileAction::Slide)
        ]);
        assert_some!(
            &TileAction::Slide,
            tile.get_action_from_coordinates(&Coordinates{x: 0, y:5}, &Coordinates{x: 0, y:3}),
        )
    }

    #[test]
    fn get_action_returns_some_slides_if_there_is_a_bottom_slide_action() {
        let tile = TileSide::new(vec![
            (&VerticalOffset::Bottom, TileAction::Slide)
        ]);
        assert_some!(
            &TileAction::Slide,
            tile.get_action_from_coordinates(&Coordinates{x: 0, y:0}, &Coordinates{x: 0, y:5}),
        )
    }

    #[test]
    fn get_action_returns_some_slides_if_there_is_a_bottom_slide_2_spaces_action() {
        let tile = TileSide::new(vec![
            (&VerticalOffset::Bottom, TileAction::Slide)
        ]);
        assert_some!(
            &TileAction::Slide,
            tile.get_action_from_coordinates(&Coordinates{x: 0, y:3}, &Coordinates{x: 0, y:5}),
        )
    }
}
