use crate::common::coordinates::Coordinates;
use crate::common::geometry::Rectangular;
use crate::game::board::DukeOffset;

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum MoveView {
    Up,
    Down,
    Left,
    Right,
}

impl MoveView {
    // For two adjacent coordinates, returns the direction src has to go to reach dst.
    // Return None if not adjacent.
    pub fn relative_direction(src: Coordinates, dst: Coordinates) -> Option<MoveView> {
        match ((src.x as i32) - (dst.x as i32), (src.y as i32) - (dst.y as i32)) {
            (-1, 0) => Some(MoveView::Right),
            (1, 0) => Some(MoveView::Left),
            (0, -1) => Some(MoveView::Down),
            (0, 1) => Some(MoveView::Up),
            _ => None,
        }
    }

    pub fn mv(&self, c: Coordinates, r: &impl Rectangular) -> Option<Coordinates> {
        (match self {
            MoveView::Up => if c.y > 0 { Some(Coordinates { x: c.x, y: c.y - 1 }) } else { None },
            MoveView::Down => if c.y < r.height() - 1 { Some(Coordinates { x: c.x, y: c.y + 1 }) } else { None },
            MoveView::Left => if c.x > 0 { Some(Coordinates { x: c.x - 1, y: c.y }) } else { None },
            MoveView::Right => if c.x < r.width() - 1 { Some(Coordinates { x: c.x + 1, y: c.y }) } else { None },
        }).filter(|e| r.is_in_bounds(*e))
    }
}

impl From<MoveView> for DukeOffset {
    fn from(mv: MoveView) -> Self {
        match mv {
            MoveView::Up => DukeOffset::Top,
            MoveView::Down => DukeOffset::Bottom,
            MoveView::Left => DukeOffset::Left,
            MoveView::Right => DukeOffset::Right,
        }
    }
}

#[cfg(test)]
mod move_view_tests {
    use crate::{assert_none, assert_some};
    use crate::common::coordinates::Coordinates;

    use super::*;

    #[test]
    fn relative_direction_some() {
        assert_some!(
            MoveView::Right,
            MoveView::relative_direction(Coordinates{x: 0, y: 1}, Coordinates{x: 1, y:1}),
        );
        assert_some!(
            MoveView::Left,
            MoveView::relative_direction(Coordinates{x: 1, y: 1}, Coordinates{x: 0, y:1}),
        );
        assert_some!(
            MoveView::Up,
            MoveView::relative_direction(Coordinates{x: 0, y: 1}, Coordinates{x: 0, y:0}),
        );
        assert_some!(
            MoveView::Down,
            MoveView::relative_direction(Coordinates{x: 0, y: 0}, Coordinates{x: 0, y:1}),
        );
    }

    #[test]
    fn relative_none() {
        assert_none!(MoveView::relative_direction(Coordinates{x: 0, y: 0}, Coordinates{x: 1, y:1}))
    }
}
