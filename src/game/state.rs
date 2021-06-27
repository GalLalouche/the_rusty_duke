use crate::common::board::Coordinate;
use crate::game::board::GameBoard;
use crate::game::token::{DiscardBag, TokenBag};
use crate::game::units;
use crate::game::units::footman;

enum CurrentTurn {
    Player1,
    Player2,
}

enum FootmenSetup {
    // Footmen are to the sides of the Duke
    Sides,
    // One Footman is above the Duke, and one is to its player's left
    Left,
    // One Footman is above the Duke, and one is to its player's right
    Right,
}

enum DukeInitialLocation {
    Left,
    Right,
}

struct GameState {
    board: GameBoard,
    current_player_turn: CurrentTurn,
    player_1_bag: TokenBag,
    player_1_discard: DiscardBag,
    player_2_bag: TokenBag,
    player_2_discard: DiscardBag,
}

impl GameState {
    pub fn new(
        base_bag: &TokenBag,
        player_1_setup: (DukeInitialLocation, FootmenSetup),
        player_2_setup: (DukeInitialLocation, FootmenSetup),
    ) -> GameState {
        let mut board = GameBoard::empty();
        let duke_1_x = match player_1_setup.0 {
            DukeInitialLocation::Left => 3,
            DukeInitialLocation::Right => 2,
        };
        board.place(Coordinate { y: 0, x: duke_1_x }, units::duke());
        let (f1_1, f1_2) = match player_1_setup.1 {
            FootmenSetup::Sides =>
                (Coordinate { x: duke_1_x + 1, y: 0 }, Coordinate { x: duke_1_x - 1, y: 0 }),
            FootmenSetup::Left =>
                (Coordinate { x: duke_1_x + 1, y: 0 }, Coordinate { x: duke_1_x, y: 1 }),
            FootmenSetup::Right =>
                (Coordinate { x: duke_1_x - 1, y: 0 }, Coordinate { x: duke_1_x, y: 1 }),
        };
        board.place(f1_1, footman());
        board.place(f1_2, footman());

        let last_row = board.height() - 1;
        let duke_2_x = match player_2_setup.0 {
            DukeInitialLocation::Left => 2,
            DukeInitialLocation::Right => 3,
        };
        board.place(Coordinate { y: last_row, x: duke_2_x }, units::duke());
        let (f2_1, f2_2) = match player_2_setup.1 {
            FootmenSetup::Sides =>
                (Coordinate { x: duke_2_x + 1, y: last_row }, Coordinate { x: duke_2_x - 1, y: last_row }),
            FootmenSetup::Left =>
                (Coordinate { x: duke_2_x - 1, y: last_row }, Coordinate { x: duke_2_x, y: last_row - 1 }),
            FootmenSetup::Right =>
                (Coordinate { x: duke_2_x + 1, y: last_row }, Coordinate { x: duke_2_x, y: last_row - 1 }),
        };
        board.place(f2_1, footman());
        board.place(f2_2, footman());

        GameState {
            board,
            current_player_turn: CurrentTurn::Player1,
            player_1_bag: base_bag.clone(),
            player_1_discard: DiscardBag::empty(),
            player_2_bag: base_bag.clone(),
            player_2_discard: DiscardBag::empty(),
        }
    }
}
