use crate::common::coordinates::Coordinates;

pub trait Rectangular {
    fn width(&self) -> u8;
    fn height(&self) -> u8;
    fn area(&self) -> u8 { self.height() * self.width() }
    fn is_in_bounds(&self, c: Coordinates) -> bool {
        c.x < self.width() && c.y < self.height()
    }
    fn is_out_of_bounds(&self, c: Coordinates) -> bool {
        !self.is_in_bounds(c)
    }
}

pub struct Rectangle {
    height: u8,
    width: u8,
}

impl Rectangle {
    pub fn with_width_and_height(width: u8, height: u8) -> Rectangle { Rectangle { width, height } }
}

impl Rectangular for Rectangle {
    fn width(&self) -> u8 { self.width }
    fn height(&self) -> u8 { self.height }
}

pub struct Square(u8);

impl Square {
    pub fn new(side: u8) -> Square { Square(side) }
    pub fn side(&self) -> u8 { self.0 }
}

impl Rectangular for Square {
    fn width(&self) -> u8 { self.side() }
    fn height(&self) -> u8 { self.side() }
}
