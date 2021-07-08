use std::convert::TryFrom;
use std::hash::Hash;

use crate::common::coordinates;
use crate::game::offset::HorizontalOffset::{FarLeft, FarRight, Left, Right};
use crate::game::offset::VerticalOffset::{Bottom, FarBottom, FarTop, Top};

pub trait Offsetable {
    fn offsets(&self) -> Vec<Offsets>;
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum HorizontalOffset {
    FarLeft,
    Left,
    Center,
    Right,
    FarRight,
}

impl HorizontalOffset {
    pub fn flipped(&self) -> HorizontalOffset {
        match self {
            HorizontalOffset::FarLeft => FarRight,
            HorizontalOffset::Left => Right,
            HorizontalOffset::Center => HorizontalOffset::Center,
            HorizontalOffset::Right => Left,
            HorizontalOffset::FarRight => FarLeft,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum HorizontalSymmetricOffset {
    Far,
    Near,
    Center,
}

impl Offsetable for HorizontalSymmetricOffset {
    fn offsets(&self) -> Vec<Offsets> {
        assert_ne!(*self, HorizontalSymmetricOffset::Center);
        (*self, VerticalOffset::Center).offsets()
    }
}

impl Offsetable for (HorizontalSymmetricOffset, VerticalOffset) {
    fn offsets(&self) -> Vec<Offsets> {
        (match self.0 {
            HorizontalSymmetricOffset::Far => vec![FarLeft, FarRight],
            HorizontalSymmetricOffset::Near => vec![Left, Right],
            HorizontalSymmetricOffset::Center => vec![HorizontalOffset::Center],
        }).iter().map(|ho| Offsets::new(*ho, self.1)).collect()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum FourWaySymmetric {
    NearLinear,
    FarLinear,
    NearDiagonal,
    FarDiagonal,
}

impl Offsetable for FourWaySymmetric {
    fn offsets(&self) -> Vec<Offsets> {
        match self {
            FourWaySymmetric::NearLinear =>
                vec![Left.center(), Right.center()],
            FourWaySymmetric::FarLinear =>
                vec![FarLeft.center(), FarRight.center()],
            FourWaySymmetric::NearDiagonal =>
                vec![
                    Offsets::new(Left, Top),
                    Offsets::new(Right, Top),
                    Offsets::new(Left, Bottom),
                    Offsets::new(Right, Bottom),
                ],
            FourWaySymmetric::FarDiagonal =>
                vec![
                    Offsets::new(FarLeft, FarTop),
                    Offsets::new(FarRight, FarTop),
                    Offsets::new(FarLeft, FarBottom),
                    Offsets::new(FarRight, FarBottom),
                ],
        }
    }
}


#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum VerticalOffset {
    FarTop,
    Top,
    Center,
    Bottom,
    FarBottom,
}

impl Offsetable for VerticalOffset {
    fn offsets(&self) -> Vec<Offsets> {
        vec![Offsets::new(HorizontalOffset::Center, *self)]
    }
}

impl VerticalOffset {
    pub fn symmetric_centered(&self) -> (HorizontalSymmetricOffset, VerticalOffset) {
        assert_ne!(*self, VerticalOffset::Center);
        (HorizontalSymmetricOffset::Center, *self)
    }
    pub fn flipped(&self) -> VerticalOffset {
        match self {
            VerticalOffset::FarTop => FarBottom,
            VerticalOffset::Top => Bottom,
            VerticalOffset::Center => VerticalOffset::Center,
            VerticalOffset::Bottom => Top,
            VerticalOffset::FarBottom => FarTop,
        }
    }
}


#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct Offsets {
    pub x: HorizontalOffset,
    pub y: VerticalOffset,
}

impl Offsets {
    pub fn center() -> Offsets {
        Offsets::new(HorizontalOffset::Center, VerticalOffset::Center)
    }
    pub fn horizontal_flipped(&self) -> Offsets {
        Offsets {
            x: self.x.flipped(),
            y: self.y,
        }
    }
    pub fn vertical_flipped(&self) -> Offsets {
        Offsets {
            x: self.x,
            y: self.y.flipped(),
        }
    }
    pub fn flipped(&self) -> Offsets {
        Offsets {
            x: self.x.flipped(),
            y: self.y.flipped(),
        }
    }

    pub fn new(x: HorizontalOffset, y: VerticalOffset) -> Offsets {
        Offsets { x, y }
    }
    pub fn is_near(&self, other: &Self) -> bool {
        self.x.distance_from(&other.x) <= 1 && self.y.distance_from(&other.y) <= 1
    }
    pub fn is_linear_from(&self, other: &Self) -> bool {
        self.x == other.x || self.y == other.y ||
            self.y.is_centered() ||
            // Covers the linear diagonals
            self.x.distance_from(&other.x) == self.y.distance_from(&other.y)
    }
}

impl From<coordinates::Coordinates> for Offsets {
    fn from(other: coordinates::Coordinates) -> Self {
        Offsets::new(Indexable::from_index(other.x), Indexable::from_index(other.y))
    }
}

impl From<Offsets> for coordinates::Coordinates {
    fn from(other: Offsets) -> coordinates::Coordinates {
        coordinates::Coordinates { x: other.x.to_index(), y: other.y.to_index() }
    }
}

pub trait Centerable {
    fn center(&self) -> Offsets;
    fn distance_from_center(&self) -> u16;
    fn is_centered(&self) -> bool {
        self.distance_from_center() == 0
    }
}

impl Centerable for HorizontalOffset {
    fn center(&self) -> Offsets {
        Offsets {
            x: self.clone(),
            y: VerticalOffset::Center,
        }
    }

    fn distance_from_center(&self) -> u16 {
        match self {
            HorizontalOffset::FarLeft => 2,
            HorizontalOffset::Left => 1,
            HorizontalOffset::Center => 0,
            HorizontalOffset::Right => 1,
            HorizontalOffset::FarRight => 2,
        }
    }
}

impl Centerable for VerticalOffset {
    fn center(&self) -> Offsets {
        Offsets {
            x: HorizontalOffset::Center,
            y: self.clone(),
        }
    }

    fn distance_from_center(&self) -> u16 {
        match self {
            FarTop => 2,
            Top => 1,
            VerticalOffset::Center => 0,
            Bottom => 1,
            FarBottom => 2,
        }
    }
}


pub trait Indexable {
    fn to_index(&self) -> u16;
    fn from_index(i: u16) -> Self;

    fn distance_from(&self, other: &Self) -> u16 {
        u16::try_from((i32::from(self.to_index()) - i32::from(other.to_index())).abs()).unwrap()
    }
}

impl Indexable for HorizontalOffset {
    fn to_index(&self) -> u16 {
        match self {
            HorizontalOffset::FarLeft => 0,
            HorizontalOffset::Left => 1,
            HorizontalOffset::Center => 2,
            HorizontalOffset::Right => 3,
            HorizontalOffset::FarRight => 4,
        }
    }

    fn from_index(i: u16) -> Self {
        match i {
            0 => FarLeft,
            1 => Left,
            2 => HorizontalOffset::Center,
            3 => Right,
            4 => FarRight,
            x => panic!("Unsupported integer <{}>", x)
        }
    }
}

impl Indexable for VerticalOffset {
    fn to_index(&self) -> u16 {
        match self {
            VerticalOffset::FarTop => 0,
            VerticalOffset::Top => 1,
            VerticalOffset::Center => 2,
            VerticalOffset::Bottom => 3,
            VerticalOffset::FarBottom => 4,
        }
    }

    fn from_index(i: u16) -> Self {
        match i {
            0 => FarTop,
            1 => Top,
            2 => VerticalOffset::Center,
            3 => Bottom,
            4 => FarBottom,
            x => panic!("Unsupported integer <{}>", x)
        }
    }
}

mod test {
    use super::*;

    #[test]
    fn coordinate_to_offsets() {
        let c = coordinates::Coordinates { x: 0, y: 2 };
        assert_eq!(
            Offsets::new(FarLeft, VerticalOffset::Center),
            c.into(),
        )
    }

    #[test]
    fn offsets_to_coordinates() {
        let os = Offsets::new(FarLeft, VerticalOffset::Center);
        assert_eq!(
            coordinates::Coordinates { x: 0, y: 2 },
            os.into(),
        )
    }
}
