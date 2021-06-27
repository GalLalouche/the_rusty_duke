use crate::common::board;

#[derive(PartialEq, Eq, Clone, Copy, Hash)]
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
            HorizontalOffset::FarLeft => HorizontalOffset::FarRight,
            HorizontalOffset::Left => HorizontalOffset::Right,
            HorizontalOffset::Center => HorizontalOffset::Center,
            HorizontalOffset::Right => HorizontalOffset::Left,
            HorizontalOffset::FarRight => HorizontalOffset::FarLeft,
        }
    }
}


#[derive(PartialEq, Eq, Clone, Copy, Hash)]
pub enum VerticalOffset {
    FarTop,
    Top,
    Center,
    Bottom,
    FarBottom,
}

impl VerticalOffset {
    pub fn flipped(&self) -> VerticalOffset {
        match self {
            VerticalOffset::FarTop => VerticalOffset::FarBottom,
            VerticalOffset::Top => VerticalOffset::Bottom,
            VerticalOffset::Center => VerticalOffset::Center,
            VerticalOffset::Bottom => VerticalOffset::Top,
            VerticalOffset::FarBottom => VerticalOffset::FarTop,
        }
    }
}


#[derive(PartialEq, Eq, Clone, Copy, Hash)]
pub struct Coordinate {
    pub x: HorizontalOffset,
    pub y: VerticalOffset,
}

impl Coordinate {
    pub fn horizontal_flipped(&self) -> Coordinate {
        Coordinate {
            x: self.x.flipped(),
            y: self.y,
        }
    }
    pub fn vertical_flipped(&self) -> Coordinate {
        Coordinate {
            x: self.x,
            y: self.y.flipped(),
        }
    }
    pub fn flipped(&self) -> Coordinate {
        Coordinate {
            x: self.x.flipped(),
            y: self.y.flipped(),
        }
    }

    pub fn new(x: HorizontalOffset, y: VerticalOffset) -> Coordinate {
        if x == HorizontalOffset::Center && y == VerticalOffset::Center {
            panic!("Cannot create a coordinate of two Centers!");
        }
        Coordinate { x, y }
    }
    pub fn centered<C: Centerable>(c: C) -> Coordinate {
        c.center()
    }
}

impl From<board::Coordinate> for Coordinate {
    fn from(other: board::Coordinate) -> Self {
        Coordinate::new(Indexable::from_index(other.x), Indexable::from_index(other.x))
    }
}

impl From<Coordinate> for board::Coordinate {
    fn from(other: Coordinate) -> board::Coordinate {
        board::Coordinate { x: other.x.to_index(), y: other.y.to_index() }
    }
}

pub trait Centerable {
    fn center(&self) -> Coordinate;
}

impl Centerable for HorizontalOffset {
    fn center(&self) -> Coordinate {
        Coordinate {
            x: self.clone(),
            y: VerticalOffset::Center,
        }
    }
}

impl Centerable for VerticalOffset {
    fn center(&self) -> Coordinate {
        Coordinate {
            x: HorizontalOffset::Center,
            y: self.clone(),
        }
    }
}


pub trait Indexable {
    fn to_index(&self) -> usize;
    fn from_index(i: usize) -> Self;
}

impl Indexable for HorizontalOffset {
    fn to_index(&self) -> usize {
        match self {
            HorizontalOffset::FarLeft => 0,
            HorizontalOffset::Left => 1,
            HorizontalOffset::Center => 2,
            HorizontalOffset::Right => 3,
            HorizontalOffset::FarRight => 4,
        }
    }

    fn from_index(i: usize) -> Self {
        match i {
            0 => HorizontalOffset::FarLeft,
            1 => HorizontalOffset::Left,
            2 => HorizontalOffset::Center,
            3 => HorizontalOffset::Right,
            4 => HorizontalOffset::FarRight,
            x => panic!("Unsupported integer <{}>", x)
        }
    }
}

impl Indexable for VerticalOffset {
    fn to_index(&self) -> usize {
        match self {
            VerticalOffset::FarTop => 0,
            VerticalOffset::Top => 1,
            VerticalOffset::Center => 2,
            VerticalOffset::Bottom => 3,
            VerticalOffset::FarBottom => 4,
        }
    }

    fn from_index(i: usize) -> Self {
        match i {
            0 => VerticalOffset::FarTop,
            1 => VerticalOffset::Top,
            2 => VerticalOffset::Center,
            3 => VerticalOffset::Bottom,
            4 => VerticalOffset::FarBottom,
            x => panic!("Unsupported integer <{}>", x)
        }
    }
}
