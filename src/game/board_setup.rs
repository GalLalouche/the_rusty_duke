use crate::common::board::Board;
use crate::common::coordinates::Coordinates;
use crate::common::geometry::Rectangular;
use crate::game::board::GameBoard;
use crate::game::tile::Owner;
use crate::game::units;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FootmenSetup {
    // Footmen are to the sides of the Duke
    Sides,
    // One Footman is above the Duke, and one is to its player's left
    Left,
    // One Footman is above the Duke, and one is to its player's right
    Right,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DukeInitialLocation { Left, Right }

pub(super) fn setup(
    player_1_setup: (DukeInitialLocation, FootmenSetup),
    player_2_setup: (DukeInitialLocation, FootmenSetup),
) -> GameBoard {
    let mut result = GameBoard::new(Board::square(GameBoard::BOARD_SIZE));

    { // First player
        let duke_1_x = match player_1_setup.0 {
            DukeInitialLocation::Left => 3,
            DukeInitialLocation::Right => 2,
        };
        result.place(
            Coordinates { y: 0, x: duke_1_x },
            units::place_tile(Owner::TopPlayer, units::duke),
        );
        let (f1_1, f1_2) = match player_1_setup.1 {
            FootmenSetup::Sides =>
                (Coordinates { x: duke_1_x + 1, y: 0 }, Coordinates { x: duke_1_x - 1, y: 0 }),
            FootmenSetup::Left =>
                (Coordinates { x: duke_1_x + 1, y: 0 }, Coordinates { x: duke_1_x, y: 1 }),
            FootmenSetup::Right =>
                (Coordinates { x: duke_1_x - 1, y: 0 }, Coordinates { x: duke_1_x, y: 1 }),
        };
        result.place(f1_1, units::place_tile(Owner::TopPlayer, units::footman));
        result.place(f1_2, units::place_tile(Owner::TopPlayer, units::footman));
    }
    { // Second player
        let last_row = result.height() - 1;
        let duke_2_x = match player_2_setup.0 {
            DukeInitialLocation::Left => 2,
            DukeInitialLocation::Right => 3,
        };
        result.place(
            Coordinates { y: last_row, x: duke_2_x }, units::place_tile(Owner::BottomPlayer, units::duke));
        let (f2_1, f2_2) = match player_2_setup.1 {
            FootmenSetup::Sides =>
                (Coordinates { x: duke_2_x + 1, y: last_row }, Coordinates { x: duke_2_x - 1, y: last_row }),
            FootmenSetup::Left =>
                (Coordinates { x: duke_2_x - 1, y: last_row }, Coordinates { x: duke_2_x, y: last_row - 1 }),
            FootmenSetup::Right =>
                (Coordinates { x: duke_2_x + 1, y: last_row }, Coordinates { x: duke_2_x, y: last_row - 1 }),
        };
        result.place(f2_1, units::place_tile(Owner::BottomPlayer, units::footman));
        result.place(f2_2, units::place_tile(Owner::BottomPlayer, units::footman));
    }

    result
}
