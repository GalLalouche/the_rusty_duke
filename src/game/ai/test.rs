#[cfg(test)]
pub mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use crate::common::coordinates::Coordinates;
    use crate::game::ai::player::{AiMove, ArtificialPlayer};
    use crate::game::board::GameBoard;
    use crate::game::state::GameState;
    use crate::game::tile::{Owner, PlacedTile};
    use crate::game::units;

    pub fn can_find_winning_move<A: ArtificialPlayer>(a: A) {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, units::duke()));
        board.place(Coordinates { x: 1, y: 0 }, PlacedTile::new(Owner::TopPlayer, units::footman()));
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::BottomPlayer, units::duke()));
        board.place(Coordinates { x: 4, y: 4 }, PlacedTile::new(Owner::BottomPlayer, units::footman()));
        let gs = GameState::from_board(board, Owner::BottomPlayer);
        let mv = a.get_next_move(&mut StdRng::seed_from_u64(0), &gs);

        assert_eq!(
            AiMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 5, y: 5 },
                dst: Coordinates { x: 0, y: 5 },
                capturing: None,
            },
            mv,
        );
    }

    pub fn can_find_winning_move_with_lookahead_2<A: ArtificialPlayer>(a: A) {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::BottomPlayer, units::duke()));
        // Extra tile to allow an empty move
        board.place(Coordinates { x: 3, y: 3 }, PlacedTile::new(Owner::TopPlayer, units::footman()));
        // This footman can move forward to block the duke
        board.place(Coordinates { x: 1, y: 1 }, PlacedTile::new(Owner::TopPlayer, units::footman()));
        // This tile guards the above footman when it moves
        let mut tile = PlacedTile::new(Owner::TopPlayer, units::pikeman());
        tile.flip();
        board.place(Coordinates { x: 2, y: 2 }, tile);
        // After moving the above footman, the second player will have to move its footman, and then
        // the top duke will have a move in 1.
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::TopPlayer, units::duke()));
        let gs = GameState::from_board(board, Owner::TopPlayer);
        let mv = a.get_next_move(&mut StdRng::seed_from_u64(0), &gs);

        assert_eq!(
            AiMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 1, y: 1 },
                dst: Coordinates { x: 1, y: 0 },
                capturing: None,
            },
            mv,
        );
    }
}