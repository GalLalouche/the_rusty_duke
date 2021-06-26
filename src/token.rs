use std::collections::HashMap;

struct TokenSide {}

#[derive(PartialEq, Eq)]
enum CurrentSide {
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
    x: HorizontalOffset,
    y: VerticalOffset,
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
}

impl Coordinate {
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

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum TokenAction {
    Move,
    Jump,
    Slide,
    Command,
    JumpSlide,
    Strike,
    Dread,
}

trait Indexable {
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

pub struct Side {
    board: Vec<Vec<Option<TokenAction>>>,
}

impl Side {
    const WIDTH: usize = 5;
    const HEIGHT: usize = 5;

    pub fn new(map: HashMap<Coordinate, TokenAction>) -> Side {
        let mut columns = Vec::with_capacity(Side::WIDTH);
        for x in 0..Side::WIDTH {
            let x_offset = Indexable::from_index(x);
            columns.push(Vec::with_capacity(Side::HEIGHT));
            let mut row = &columns[x];
            for y in 0..Side::HEIGHT {
                let y_offset = Indexable::from_index(y);
                row.push(if x_offset == HorizontalOffset::Center || y_offset == VerticalOffset::Center {
                    None
                } else {
                    let c = Coordinate::new(x_offset, y_offset);
                    map.get(&c).cloned()
                });
            }
        }
        Side { board: columns }
    }
    pub fn action(&self, c: Coordinate) -> Option<TokenAction> {
        self.board[c.x.to_index()][c.y.to_index()]
    }
    pub fn actions(&self) -> Vec<(Coordinate, TokenAction)> {
        let mut result = Vec::new();
        for (x, v) in self.board.iter().enumerate() {
            let x_offset = Indexable::from_index(x);
            for (y, e) in v.iter().enumerate() {
                let y_offset = Indexable::from_index(y);
                if x_offset == HorizontalOffset::Center || y_offset == VerticalOffset::Center {
                    continue;
                }
                let c = Coordinate::new(x_offset, y_offset);
                match e {
                    None => (),
                    Some(a) => result.push((c, a.clone())),
                }
            }
        }
        result
    }
}

pub struct GameToken {
    pub side_a: Side,
    pub side_b: Side,
    pub current_side: CurrentSide,
}

impl GameToken {
    pub fn new(side_a: Side, side_b: Side) -> GameToken {
        GameToken {
            side_a,
            side_b,
            current_side: CurrentSide::Initial,
        }
    }
    pub fn flip(&mut self) -> () {
        self.current_side = self.current_side.flip()
    }
    pub fn get_current_side(&self) -> &Side {
        match self.current_side {
            CurrentSide::Initial => &self.side_a,
            CurrentSide::Flipped => &self.side_b,
        }
    }
}