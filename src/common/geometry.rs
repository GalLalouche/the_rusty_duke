use crate::common::coordinates::Coordinates;

pub trait Rectangular {
    fn width(&self) -> u16;
    fn height(&self) -> u16;
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

impl Rectangular for Rectangle {
    fn width(&self) -> u16 {
        self.height
    }

    fn height(&self) -> u16 {
        self.width
    }
}
