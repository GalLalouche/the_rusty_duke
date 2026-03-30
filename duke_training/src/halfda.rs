//! HalfDA (Half-Duke-Accumulator) feature encoding for NNUE-style networks.
//!
//! Each non-duke piece on the board is encoded as a sparse feature indexed by:
//!   duke_square * (36 * 13 * 2 * 2) + piece_square * (13 * 2 * 2) + piece_type * (2 * 2) + piece_color * 2 + piece_side
//!
//! Where:
//! - `duke_square`: 0-35 (current player's duke position, y*6+x)
//! - `piece_square`: 0-35 (the other piece's position)
//! - `piece_type`: 0-12 (TileType enum, 13 types)
//! - `piece_color`: 0=my piece, 1=opponent piece
//! - `piece_side`: 0=Initial, 1=Flipped
//!
//! Total features: 36 * 36 * 13 * 2 * 2 = 67,392
//! Active features per position: ~4-10 (one per non-duke piece on board)

use duke_rust::game::state::GameState;
use duke_rust::game::tile::{CurrentSide, TileType};

/// Total number of HalfDA features.
pub const HALFDA_FEATURES: usize = 36 * 36 * 13 * 2 * 2; // 67,392

/// Stride constants for the feature index computation.
const PIECE_SIDE_STRIDE: usize = 1;
const PIECE_COLOR_STRIDE: usize = 2 * PIECE_SIDE_STRIDE;           // 2
const PIECE_TYPE_STRIDE: usize = 2 * PIECE_COLOR_STRIDE;           // 4
const PIECE_SQUARE_STRIDE: usize = 13 * PIECE_TYPE_STRIDE;         // 52
const DUKE_SQUARE_STRIDE: usize = 36 * PIECE_SQUARE_STRIDE;        // 1872

/// Maximum number of non-duke pieces on the board.
/// In The Duke, the board is 6x6=36 squares. With 2 dukes, at most 34 non-duke pieces.
/// In practice, typical positions have 4-10 non-duke pieces.
const MAX_PIECES: usize = 34;

/// Stack-allocated buffer for HalfDA feature indices.
pub struct HalfDABuffer {
    pub indices: [u32; MAX_PIECES],
    pub len: usize,
}

impl HalfDABuffer {
    #[inline]
    fn new() -> Self {
        Self {
            indices: [0; MAX_PIECES],
            len: 0,
        }
    }

    #[inline]
    fn push(&mut self, idx: u32) {
        debug_assert!(self.len < MAX_PIECES,
            "HalfDABuffer overflow: tried to push {} features (max {})",
            self.len + 1, MAX_PIECES);
        debug_assert!((idx as usize) < HALFDA_FEATURES,
            "HalfDA feature index {} >= {}", idx, HALFDA_FEATURES);
        self.indices[self.len] = idx;
        self.len += 1;
    }

    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        &self.indices[..self.len]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
}

/// Encode a game state using HalfDA features.
///
/// The encoding is from the current player's perspective:
/// - The current player's duke position is `duke_square`
/// - `piece_color` is 0 for current player's pieces, 1 for opponent's
pub fn encode_halfda(gs: &GameState) -> HalfDABuffer {
    let current_player = gs.current_player_turn();
    let board = gs.board();

    // Find current player's duke position
    let duke_coords = board
        .find(|t| t.owner == current_player && t.tile_type == TileType::Duke)
        .expect("Current player's duke must be on the board");
    let duke_square = duke_coords.y as usize * 6 + duke_coords.x as usize;

    let mut buf = HalfDABuffer::new();

    for (coords, placed_tile) in board.active_coordinates() {
        // Skip the current player's duke (it's the reference point)
        if placed_tile.tile_type == TileType::Duke && placed_tile.owner == current_player {
            continue;
        }

        let piece_square = coords.y as usize * 6 + coords.x as usize;
        let piece_type = placed_tile.tile_type.index();
        let piece_color = if placed_tile.owner == current_player { 0 } else { 1 };
        let piece_side = match placed_tile.current_side {
            CurrentSide::Initial => 0,
            CurrentSide::Flipped => 1,
        };

        let idx = duke_square * DUKE_SQUARE_STRIDE
            + piece_square * PIECE_SQUARE_STRIDE
            + piece_type * PIECE_TYPE_STRIDE
            + piece_color * PIECE_COLOR_STRIDE
            + piece_side * PIECE_SIDE_STRIDE;

        buf.push(idx as u32);
    }

    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use duke_rust::common::coordinates::Coordinates;
    use duke_rust::game::bag::{DiscardBag, TileBag};
    use duke_rust::game::state::{GameSnapshot, GameState};
    use duke_rust::game::tile::{CurrentSide, Owner, PlacedTile, TileType};

    /// Helper: build a GameState from a list of tiles, bags, and current player.
    fn make_state(
        tiles: Vec<(Coordinates, PlacedTile)>,
        current_turn: Owner,
    ) -> GameState {
        GameState::from_snapshot(GameSnapshot {
            tiles,
            current_turn,
            top_bag: TileBag::new(vec![]),
            bottom_bag: TileBag::new(vec![]),
            top_discard: DiscardBag::empty(),
            bottom_discard: DiscardBag::empty(),
            idle_move_count: 0,
        })
    }

    fn make_placed(owner: Owner, tile_type: TileType, flipped: bool) -> PlacedTile {
        let mut p = PlacedTile::new(owner, tile_type);
        if flipped {
            p.flip();
        }
        p
    }

    #[test]
    fn test_initial_board_feature_count() {
        // Initial board: 2 dukes + 4 footmen = 6 tiles total.
        // HalfDA encodes non-duke pieces only, so 5 features (opponent duke + 4 footmen).
        use duke_rust::game::board_setup::{DukeInitialLocation, FootmenSetup};
        let bag = TileBag::new(vec![
            TileType::Footman, TileType::Pikeman, TileType::Knight,
            TileType::Champion, TileType::Dragoon, TileType::Wizard,
            TileType::General, TileType::Marshall, TileType::Bowman,
            TileType::Longbowman, TileType::Priest,
        ]);
        let gs = GameState::new(
            &bag,
            (DukeInitialLocation::Left, FootmenSetup::Left),
            (DukeInitialLocation::Right, FootmenSetup::Right),
        );

        let features = encode_halfda(&gs);
        // 6 tiles on board, minus 1 (current player's duke) = 5 features
        assert_eq!(features.len(), 5,
            "Initial board should have 5 HalfDA features (opponent duke + 4 footmen)");
    }

    #[test]
    fn test_feature_indices_within_bounds() {
        // Create a position with several tiles
        let tiles = vec![
            (Coordinates { x: 0, y: 0 }, make_placed(Owner::TopPlayer, TileType::Duke, false)),
            (Coordinates { x: 5, y: 5 }, make_placed(Owner::BottomPlayer, TileType::Duke, false)),
            (Coordinates { x: 1, y: 0 }, make_placed(Owner::TopPlayer, TileType::Footman, false)),
            (Coordinates { x: 2, y: 3 }, make_placed(Owner::BottomPlayer, TileType::Knight, true)),
            (Coordinates { x: 4, y: 4 }, make_placed(Owner::TopPlayer, TileType::Wizard, true)),
        ];
        let gs = make_state(tiles, Owner::TopPlayer);
        let features = encode_halfda(&gs);

        for &idx in features.as_slice() {
            assert!((idx as usize) < HALFDA_FEATURES,
                "Feature index {} is out of bounds (max {})", idx, HALFDA_FEATURES);
        }
    }

    #[test]
    fn test_no_duplicate_indices() {
        let tiles = vec![
            (Coordinates { x: 0, y: 0 }, make_placed(Owner::TopPlayer, TileType::Duke, false)),
            (Coordinates { x: 5, y: 5 }, make_placed(Owner::BottomPlayer, TileType::Duke, false)),
            (Coordinates { x: 1, y: 0 }, make_placed(Owner::TopPlayer, TileType::Footman, false)),
            (Coordinates { x: 2, y: 0 }, make_placed(Owner::TopPlayer, TileType::Footman, true)),
            (Coordinates { x: 3, y: 3 }, make_placed(Owner::BottomPlayer, TileType::Knight, false)),
            (Coordinates { x: 4, y: 4 }, make_placed(Owner::BottomPlayer, TileType::Pikeman, true)),
        ];
        let gs = make_state(tiles, Owner::TopPlayer);
        let features = encode_halfda(&gs);

        let mut seen = std::collections::HashSet::new();
        for &idx in features.as_slice() {
            assert!(seen.insert(idx), "Duplicate HalfDA feature index: {}", idx);
        }
    }

    #[test]
    fn test_encoding_flips_with_player_perspective() {
        // Same board, but different current player -> different encoding
        let tiles = vec![
            (Coordinates { x: 0, y: 0 }, make_placed(Owner::TopPlayer, TileType::Duke, false)),
            (Coordinates { x: 5, y: 5 }, make_placed(Owner::BottomPlayer, TileType::Duke, false)),
            (Coordinates { x: 2, y: 2 }, make_placed(Owner::TopPlayer, TileType::Footman, false)),
        ];

        let gs_top = make_state(tiles.clone(), Owner::TopPlayer);
        let gs_bottom = make_state(tiles, Owner::BottomPlayer);

        let feat_top = encode_halfda(&gs_top);
        let feat_bottom = encode_halfda(&gs_bottom);

        // When TopPlayer is current: duke at (0,0), encode bottom duke + footman
        // When BottomPlayer is current: duke at (5,5), encode top duke + footman
        // The features must differ because duke_square and piece_color both change
        let top_set: std::collections::HashSet<u32> = feat_top.as_slice().iter().copied().collect();
        let bottom_set: std::collections::HashSet<u32> = feat_bottom.as_slice().iter().copied().collect();
        assert_ne!(top_set, bottom_set,
            "HalfDA encoding should differ when current player changes");
    }

    #[test]
    fn test_known_feature_index() {
        // Manually compute a known feature index:
        // Duke at (0,0) = square 0
        // Footman at (1,0) = square 1, type=1 (Footman), color=0 (mine), side=0 (Initial)
        // idx = 0 * 1872 + 1 * 52 + 1 * 4 + 0 * 2 + 0 = 56
        let tiles = vec![
            (Coordinates { x: 0, y: 0 }, make_placed(Owner::TopPlayer, TileType::Duke, false)),
            (Coordinates { x: 5, y: 5 }, make_placed(Owner::BottomPlayer, TileType::Duke, false)),
            (Coordinates { x: 1, y: 0 }, make_placed(Owner::TopPlayer, TileType::Footman, false)),
        ];
        let gs = make_state(tiles, Owner::TopPlayer);
        let features = encode_halfda(&gs);

        // Should have 2 features: opponent duke and my footman
        assert_eq!(features.len(), 2);

        let indices: Vec<u32> = features.as_slice().to_vec();
        // My footman: duke_sq=0, piece_sq=1, type=1, color=0, side=0
        // idx = 0*1872 + 1*52 + 1*4 + 0*2 + 0 = 56
        assert!(indices.contains(&56),
            "Expected feature index 56 for own Footman at (1,0), got {:?}", indices);

        // Opponent duke: duke_sq=0, piece_sq=35, type=0(Duke), color=1(opp), side=0(Initial)
        // idx = 0*1872 + 35*52 + 0*4 + 1*2 + 0 = 1822
        assert!(indices.contains(&1822),
            "Expected feature index 1822 for opponent Duke at (5,5), got {:?}", indices);
    }
}
