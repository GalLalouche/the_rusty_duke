use crate::common::coordinates::Coordinates;

pub trait Rectangular {
    fn width(&self) -> u16;
    fn height(&self) -> u16;
    fn area(&self) -> u16 { self.height() * self.width() }
    fn is_in_bounds(&self, c: Coordinates) -> bool {
        c.x < self.width() && c.y < self.height()
    }
    fn is_out_of_bounds(&self, c: Coordinates) -> bool {
        !self.is_in_bounds(c)
    }
}

pub struct Rectangle {
    height: u16,
    width: u16,
}

impl Rectangle {
    pub fn with_width_and_height(width: u16, height: u16) -> Rectangle { Rectangle { width, height } }
}

impl Rectangular for Rectangle {
    fn width(&self) -> u16 { self.width }
    fn height(&self) -> u16 { self.height }
}

pub struct Square(u16);

impl Square {
    pub fn new(side: u16) -> Square { Square(side) }
    pub fn side(&self) -> u16 { self.0 }
}

impl Rectangular for Square {
    fn width(&self) -> u16 { self.side() }
    fn height(&self) -> u16 { self.side() }
}
