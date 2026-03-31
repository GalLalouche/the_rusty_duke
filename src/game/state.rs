use std::hash::{Hash, Hasher};
use rand::seq::SliceRandom;
use rand::Rng;
use strum::IntoEnumIterator;

use crate::assert_not;
use crate::common::board::Board;
use crate::common::coordinates::Coordinates;
use crate::common::geometry::Rectangular;
use crate::common::percentage::Percentage;
use crate::game::board_setup;
use crate::game::bag::{DiscardBag, TileBag};
use crate::game::board::{BoardMove, DukeOffset, GameBoard, PossibleMove, WithNewTiles};
use crate::game::board_setup::{DukeInitialLocation, FootmenSetup};
use crate::game::dumb_printer::{double_char_print_state, single_char_print_state};
use crate::game::tile::{CurrentSide, Owner, PlacedTile, TileType};
use crate::game::tile_side::TileAction;

// Technically not part of the base game rules, but it makes it easier for the AI
pub const MAX_MOVES_WITHOUT_CAPTURE_OR_PLACEMENT: usize = 10;

/// Maximum depth of the idle-move stack (captures + placements reset it).
/// Each game turn can push 1-2 entries (PullAndPlay pushes twice).
/// Reduced from 1536 to 512 to shrink GameState by ~1KB for better cache
/// locality. 512 entries is still sufficient for MAX_TURNS (500) games
/// plus search depth overhead.
/// Uses u8 values (max idle count is 10) to keep the array compact.
const IDLE_STACK_CAP: usize = 128;

#[derive(Debug, Clone, Eq)]
pub struct GameState {
    board: GameBoard,
    pulled_tile: Option<TileType>,
    current_player_turn: Owner,
    top_player_bag: TileBag,
    top_player_discard: DiscardBag,
    bottom_player_bag: TileBag,
    bottom_player_discard: DiscardBag,
    /// Stack tracking the number of consecutive moves without a capture or
    /// placement.  Uses a fixed-size array instead of Vec to avoid heap
    /// allocation and improve cache locality (this struct is cloned and
    /// make/undo'd on every negamax node).
    /// Values are u8 since the max idle count is MAX_MOVES_WITHOUT_CAPTURE_OR_PLACEMENT (10).
    idle_stack: [u8; IDLE_STACK_CAP],
    idle_stack_len: u16,
}

impl PartialEq for GameState {
    fn eq(&self, other: &Self) -> bool {
        self.board == other.board
            && self.pulled_tile == other.pulled_tile
            && self.current_player_turn == other.current_player_turn
            && self.top_player_bag == other.top_player_bag
            && self.top_player_discard == other.top_player_discard
            && self.bottom_player_bag == other.bottom_player_bag
            && self.bottom_player_discard == other.bottom_player_discard
            && self.idle_stack_len == other.idle_stack_len
            && self.idle_stack[..self.idle_stack_len as usize]
                == other.idle_stack[..other.idle_stack_len as usize]
    }
}

#[derive(Debug, Clone)]
pub enum GameMove {
    // TODO document how you can place without pulling...
    PlaceNewTile(DukeOffset),
    PullAndPlay(DukeOffset),
    ApplyNonCommandTileAction { src: Coordinates, dst: Coordinates },
}

impl Into<GameMove> for &PossibleMove {
    fn into(self) -> GameMove {
        match self {
            PossibleMove::PlaceNewTile(o, _) => GameMove::PullAndPlay(*o),
            PossibleMove::ApplyNonCommandTileAction { src, dst, .. } =>
                GameMove::ApplyNonCommandTileAction {
                    src: *src,
                    dst: *dst,
                }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CanPullNewTileResult {
    EmptyBag,
    NoSpaceNearDuke,
    DukeAlwaysInGuard,
    OK,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum GameResult {
    Tie,
    Ongoing,
    Won(Owner),
}

/// Parameter struct for [`GameState::from_snapshot`], avoiding positional-argument transposition bugs.
pub struct GameSnapshot {
    pub tiles: Vec<(Coordinates, PlacedTile)>,
    pub top_bag: TileBag,
    pub bottom_bag: TileBag,
    pub top_discard: DiscardBag,
    pub bottom_discard: DiscardBag,
    pub current_turn: Owner,
    pub idle_move_count: usize,
}

impl GameState {
    pub fn board(&self) -> &Board<PlacedTile> { self.board.get_board() }

    pub fn pulled_tile(&self) -> &Option<TileType> { &self.pulled_tile }
    #[inline]
    pub fn current_player_turn(&self) -> Owner { self.current_player_turn }
    #[inline]
    pub fn idle_move_count(&self) -> usize {
        debug_assert!(self.idle_stack_len > 0, "idle_stack should never be empty");
        self.idle_stack[self.idle_stack_len as usize - 1] as usize
    }

    #[inline]
    fn idle_stack_push(&mut self, val: u8) {
        let idx = self.idle_stack_len as usize;
        debug_assert!(idx < IDLE_STACK_CAP, "idle_stack overflow");
        self.idle_stack[idx] = val;
        self.idle_stack_len += 1;
    }

    #[inline]
    fn idle_stack_pop(&mut self) {
        debug_assert!(self.idle_stack_len > 0, "idle_stack underflow");
        self.idle_stack_len -= 1;
    }

    #[inline]
    fn idle_stack_last_mut(&mut self) -> &mut u8 {
        debug_assert!(self.idle_stack_len > 0, "idle_stack empty");
        &mut self.idle_stack[self.idle_stack_len as usize - 1]
    }
    pub fn top_player_bag(&self) -> &TileBag { &self.top_player_bag }
    pub fn player_1_discard(&self) -> &DiscardBag { &self.top_player_discard }
    pub fn bottom_player_bag(&self) -> &TileBag { &self.bottom_player_bag }
    pub fn player_2_discard(&self) -> &DiscardBag { &self.bottom_player_discard }

    #[cfg(test)]
    pub(super) fn from_board(board: GameBoard, current_player: Owner) -> GameState {
        GameState::from_board_with_bag(board, current_player, TileBag::empty())
    }
    #[cfg(test)]
    pub(super) fn from_board_with_bag(
        board: GameBoard, current_player_turn: Owner, bag: TileBag) -> GameState {
        GameState {
            board,
            current_player_turn,
            pulled_tile: None,
            top_player_bag: bag.clone(),
            top_player_discard: DiscardBag::empty(),
            bottom_player_bag: bag,
            bottom_player_discard: DiscardBag::empty(),
            idle_stack: [0u8; IDLE_STACK_CAP],
            idle_stack_len: 1,
        }
    }

    /// Reconstruct a GameState from raw snapshot data (for deserialization).
    ///
    /// Panics if either player's duke is missing from the board, since many
    /// downstream methods (e.g., `duke_coordinate`, guard checking) assume both
    /// dukes are present.
    pub fn from_snapshot(snap: GameSnapshot) -> GameState {
        let mut board = GameBoard::empty();
        for (coords, placed) in snap.tiles {
            board.place(coords, placed);
        }
        // Validate that both dukes exist on the board -- downstream methods
        // (duke_coordinate, guard checking) will panic if a duke is missing.
        assert!(
            board.get_board().find(|t: &PlacedTile| t.owner == Owner::TopPlayer && t.tile_type.is_duke()).is_some(),
            "from_snapshot: TopPlayer duke is missing from the board"
        );
        assert!(
            board.get_board().find(|t: &PlacedTile| t.owner == Owner::BottomPlayer && t.tile_type.is_duke()).is_some(),
            "from_snapshot: BottomPlayer duke is missing from the board"
        );
        let mut idle_stack = [0u8; IDLE_STACK_CAP];
        idle_stack[0] = snap.idle_move_count as u8;
        GameState {
            board,
            current_player_turn: snap.current_turn,
            pulled_tile: None,
            top_player_bag: snap.top_bag,
            top_player_discard: snap.top_discard,
            bottom_player_bag: snap.bottom_bag,
            bottom_player_discard: snap.bottom_discard,
            idle_stack,
            idle_stack_len: 1,
        }
    }
    pub fn new(
        base_bag: &TileBag,
        player_1_setup: (DukeInitialLocation, FootmenSetup),
        player_2_setup: (DukeInitialLocation, FootmenSetup),
    ) -> GameState {
        let board = board_setup::setup(player_1_setup, player_2_setup);

        GameState {
            board,
            current_player_turn: Owner::TopPlayer,
            pulled_tile: None,
            top_player_bag: base_bag.clone(),
            top_player_discard: DiscardBag::empty(),
            bottom_player_bag: base_bag.clone(),
            bottom_player_discard: DiscardBag::empty(),
            idle_stack: [0u8; IDLE_STACK_CAP],
            idle_stack_len: 1,
        }
    }

    fn can_pull_tile_from_bag(&mut self) -> CanPullNewTileResult {
        let bag = match self.current_player_turn {
            Owner::TopPlayer => &self.top_player_bag,
            Owner::BottomPlayer => &self.bottom_player_bag,
        };
        if bag.is_empty() {
            CanPullNewTileResult::EmptyBag
        } else if self.board.can_place_new_tile_near_duke(self.current_player_turn) {
            if DukeOffset::iter().any(|offset| self.is_valid_placement(offset)) {
                CanPullNewTileResult::OK
            } else {
                CanPullNewTileResult::DukeAlwaysInGuard
            }
        } else {
            CanPullNewTileResult::NoSpaceNearDuke
        }
    }

    pub fn can_pull_tile_from_bag_bool(&mut self) -> bool {
        self.can_pull_tile_from_bag() == CanPullNewTileResult::OK
    }

    pub fn pull_tile_from_bag<R: Rng>(&mut self, rng: &mut R) -> () {
        let can_pull = self.can_pull_tile_from_bag();
        assert_eq!(
            can_pull, CanPullNewTileResult::OK, "Cannot pull new tile from bag: {:?}", can_pull);
        let bag = match self.current_player_turn {
            Owner::TopPlayer => &mut self.top_player_bag,
            Owner::BottomPlayer => &mut self.bottom_player_bag,
        };
        let result =
            bag.pull(rng).expect("Assertion Error: bag should not have been empty by this point");
        self.pulled_tile = Some(result);
    }

    /// Remove a specific tile type from the current player's bag and set it
    /// as the pulled tile.  Panics if the tile is not in the bag.
    /// After calling this, use `make_a_move(GameMove::PlaceNewTile(offset))` to place it.
    pub fn pull_specific_tile_from_bag(&mut self, tile: TileType) {
        let bag = match self.current_player_turn {
            Owner::TopPlayer => &mut self.top_player_bag,
            Owner::BottomPlayer => &mut self.bottom_player_bag,
        };
        assert!(
            bag.remove_specific(tile),
            "pull_specific_tile_from_bag: tile {:?} not found in bag", tile,
        );
        self.pulled_tile = Some(tile);
        // Push a sentinel entry for the "pull" half.  make_a_move(PlaceNewTile)
        // will push a second entry for the "place" half.  undo(PlaceNewTile)
        // pops both entries, mirroring the make_a_move(PullAndPlay) path which
        // also pushes twice (once for PullAndPlay, once for the recursive
        // PlaceNewTile).
        self.idle_stack_push(0);
    }

    fn is_waiting_for_tile_placement(&self) -> bool {
        self.pulled_tile.is_some()
    }

    pub fn can_make_a_move(&mut self, game_move: &GameMove) -> bool {
        match game_move {
            GameMove::PlaceNewTile(o) =>
                self.is_waiting_for_tile_placement() && self.is_valid_placement(*o),
            GameMove::PullAndPlay(o) =>
                self.can_pull_tile_from_bag_bool() && self.is_valid_placement(*o),
            GameMove::ApplyNonCommandTileAction { src, dst } =>
                self.board.can_move(*src, *dst) &&
                    self.board.does_not_put_in_guard(
                        self.game_move_to_board_move(&game_move), self.current_player_turn),
        }
    }
    #[inline]
    pub fn make_a_move<R: Rng>(&mut self, game_move: GameMove, rng: &mut R) -> () {
        match game_move {
            GameMove::PlaceNewTile(_) | GameMove::PullAndPlay(_) =>
                self.idle_stack_push(0),
            GameMove::ApplyNonCommandTileAction { src: _, dst } =>
                if self.board.get(dst).is_some() {
                    // Capture: reset idle counter (push new 0).
                    self.idle_stack_push(0)
                } else {
                    // Non-capture: increment current idle counter.
                    *self.idle_stack_last_mut() += 1
                },
        };
        if let GameMove::PlaceNewTile(_) = game_move {
            debug_assert!(self.is_waiting_for_tile_placement(), "Invalid state for placing a new tile");
        } else {
            debug_assert!(!self.is_waiting_for_tile_placement(), "Waiting for a new tile placement");
        }
        if let GameMove::PullAndPlay(o) = &game_move {
            self.pull_tile_from_bag(rng);
            self.make_a_move(GameMove::PlaceNewTile(*o), rng);
            return;
        }
        if let GameMove::ApplyNonCommandTileAction { src, dst } = game_move {
            let tile = self.board.get(src).expect("Cannot move from an empty tile");
            debug_assert_eq!(
                tile.owner,
                self.current_player_turn,
                "Cannot move unowned tile in {:?}",
                src
            );
            debug_assert!(self.board.can_move(src, dst), "Can't move from {} to {}", src, dst)
        }
        let board_move = self.game_move_to_board_move(&game_move);
        let captured = self.board.make_a_move(board_move);
        if let Some(captured_tile) = captured {
            // Captured tile goes to its owner's discard pile (not the captor's)
            self.discard_bag_for_mut(captured_tile.owner)
                .add(captured_tile.tile_type);
        }
        debug_assert!(!self.board.is_guard(self.current_player_turn),
            "make_a_move left current player in guard");
        self.current_player_turn = self.current_player_turn.next_player();
        if self.is_waiting_for_tile_placement() {
            self.pulled_tile = None
        }
        debug_assert!(self.idle_stack_len > 0,
            "idle_stack should never be empty after make_a_move");
    }

    fn game_move_to_board_move(&self, gm: &GameMove) -> BoardMove {
        match gm {
            GameMove::PlaceNewTile(offset) =>
                BoardMove::PlaceNewTile(
                    self.pulled_tile.expect("No pulled tile"),
                    *offset,
                    self.current_player_turn,
                ),
            GameMove::ApplyNonCommandTileAction { src, dst } =>
                BoardMove::ApplyNonCommandTileAction { src: *src, dst: *dst },
            // GameMove::CommandAnotherTile { commander_src, unit_src, unit_dst } =>
            //     BoardMove::CommandAnotherTile { commander_src, unit_src, unit_dst },
            GameMove::PullAndPlay(_) => todo!(),
        }
    }

    fn possible_move_to_board_move(&self, pm: &PossibleMove) -> BoardMove {
        match pm {
            PossibleMove::PlaceNewTile(offset, owner) => BoardMove::PlaceNewTile(
                TileType::Footman, // stub tile for guard checking
                *offset,
                *owner,
            ),
            PossibleMove::ApplyNonCommandTileAction { src, dst, .. } =>
                BoardMove::ApplyNonCommandTileAction { src: *src, dst: *dst },
        }
    }


    pub fn get_tiles_for_current_owner(&self) -> Vec<(Coordinates, &PlacedTile)> {
        self.get_tiles_for_owner(self.current_player_turn)
    }

    pub fn get_tiles_for_owner(&self, o: Owner) -> Vec<(Coordinates, &PlacedTile)> {
        self.board.get_tiles_for(o).collect()
    }

    /// Count tiles for a given owner without heap allocation.
    #[inline]
    pub fn count_tiles_for_owner(&self, o: Owner) -> usize {
        self.board.get_tiles_for(o).count()
    }

    // Except commands
    pub fn get_legal_moves(&mut self, src: Coordinates) -> Vec<(Coordinates, TileAction)> {
        self.board.get_legal_moves(src)
    }

    // Except commands
    pub fn get_legal_moves_ignoring_guard(&self, src: Coordinates) -> Vec<(Coordinates, TileAction)> {
        self.board.get_legal_moves_ignoring_guard(src)
    }

    /// Count legal moves for a tile without guard checking and without heap allocation.
    #[inline]
    pub fn count_legal_moves_ignoring_guard(&self, src: Coordinates) -> usize {
        self.board.count_legal_moves_ignoring_guard(src)
    }

    /// Check if the piece at `src` can reach `target` ignoring friendly
    /// occupancy (but respecting path obstruction). Used for computing
    /// "defended" features in the training code.
    pub fn can_reach_square_ignoring_friendly(&self, src: Coordinates, target: Coordinates) -> bool {
        self.board.can_reach_square_ignoring_friendly(src, target)
    }

    pub fn current_duke_coordinate(&self) -> Coordinates {
        self.duke_coordinate(self.current_player_turn)
    }

    pub fn duke_coordinate(&self, o: Owner) -> Coordinates {
        self.board.duke_coordinates(o)
    }

    pub fn empty_spaces_near_current_duke(&self) -> crate::game::board::DukeNeighbors {
        self.board.empty_spaces_near_current_duke(self.current_player_turn)
    }

    pub fn is_valid_placement(&mut self, offset: DukeOffset) -> bool {
        let owner = self.current_player_turn;
        self.board.is_valid_placement(owner, offset) &&
            self.board.does_not_put_in_guard(
                BoardMove::PlaceNewTile(TileType::Footman, offset, owner),
                owner,
            )
    }

    #[inline]
    pub fn is_tie(&self) -> bool {
        self.idle_move_count() >= MAX_MOVES_WITHOUT_CAPTURE_OR_PLACEMENT
    }
    pub fn is_over(&mut self) -> bool {
        !self.board.has_valid_moves(
            self.current_player_turn,
            WithNewTiles(self.bag_for_current_player().non_empty()),
        ) || self.is_tie()
    }

    #[inline]
    pub fn game_result(&mut self) -> GameResult {
        if self.is_tie() {
            GameResult::Tie
        } else if self.is_over() {
            GameResult::Won(self.current_player_turn.next_player())
        } else {
            GameResult::Ongoing
        }
    }

    // Except commands for now
    pub fn all_valid_game_moves_for_current_player(&mut self) -> impl Iterator<Item=PossibleMove> + '_ {
        self.all_valid_game_moves_for(self.current_player_turn)
    }

    // Faster than collecting the above, since it avoid some validations.
    pub fn get_random_move_for_current_player<R>(
        &mut self, rng: &mut R, new_tile_boost: Percentage,
    ) -> Option<PossibleMove> where R: Rng {
        let only_place = new_tile_boost.roll(rng);
        if self.is_over() {
            return None;
        }
        // Gist of the algorithm: get all moves, legal or otherwise. Shuffle all moves and find the
        // first legal move, i.e., one that does not put into guard.
        let mut moves: Vec<PossibleMove> = self.board.all_valid_moves_ignoring_guard(
            self.current_player_turn,
            WithNewTiles(self.bag_for_current_player().non_empty()),
        );
        moves.shuffle(rng);
        let moves = moves;
        assert_not!(moves.is_empty());
        Some(
            self.get_random_move_for_current_player_aux(only_place, &moves)
                .or_else(|| self.get_random_move_for_current_player_aux(false, &moves))
                .expect("No moves found")
        )
    }

    fn get_random_move_for_current_player_aux(
        &mut self, only_place: bool, moves: &Vec<PossibleMove>,
    ) -> Option<PossibleMove> {
        for mv in moves {
            if only_place && (match &mv {
                PossibleMove::PlaceNewTile { .. } => false,
                _ => true,
            }) {
                continue;
            }
            let is_valid = match mv {
                PossibleMove::PlaceNewTile(o, _) => self.is_valid_placement(*o),
                PossibleMove::ApplyNonCommandTileAction { .. } => {
                    self.board.does_not_put_in_guard(
                        self.possible_move_to_board_move(&mv),
                        self.current_player_turn,
                    )
                }
            };
            if is_valid {
                return Some(mv.clone());
            }
        }
        None
    }

    pub fn all_valid_game_moves_for(&mut self, o: Owner) -> impl Iterator<Item=PossibleMove> + '_ {
        self.board.all_valid_moves(
            o,
            WithNewTiles(self.bag_for_owner(o).non_empty()),
        ).into_iter()
    }

    pub fn all_valid_game_moves_for_ignoring_guard(&self, o: Owner) -> Vec<PossibleMove> {
        self.board.all_valid_moves_ignoring_guard(
            o,
            WithNewTiles(self.bag_for_owner(o).non_empty()),
        )
    }

    /// Count all valid moves for `owner` without guard checking and without
    /// heap allocation. Like `all_valid_game_moves_for_ignoring_guard(..).len()`.
    #[inline]
    pub fn count_all_valid_moves_ignoring_guard(&self, o: Owner) -> usize {
        self.board.count_all_valid_moves_ignoring_guard(
            o,
            WithNewTiles(self.bag_for_owner(o).non_empty()),
        )
    }

    /// Count all valid moves AND duke-specific moves in a single pass.
    /// Returns `(total_moves, duke_moves)`. No heap allocation.
    #[inline]
    pub fn count_moves_with_duke_ignoring_guard(&self, o: Owner) -> (usize, usize) {
        self.board.count_moves_with_duke_ignoring_guard(
            o,
            WithNewTiles(self.bag_for_owner(o).non_empty()),
        )
    }

    /// Compute heuristic data for BOTH players in a single pass over the board.
    /// Returns `(own_total_moves, own_duke_moves, own_tiles, opp_total_moves, opp_duke_moves, opp_tiles)`.
    #[inline]
    pub fn heuristic_counts_both_players(&self, owner: Owner) -> (usize, usize, usize, usize, usize, usize) {
        let other = owner.next_player();
        self.board.heuristic_counts_both_players(
            owner,
            self.bag_for_owner(owner).non_empty(),
            self.bag_for_owner(other).non_empty(),
        )
    }

    /// Iterate tile-movement moves for `owner` without guard checking,
    /// calling `f(src, dst)` for each. No heap allocation.
    #[inline]
    pub fn for_each_tile_move_ignoring_guard<F: FnMut(Coordinates, Coordinates)>(&self, owner: Owner, f: F) {
        self.board.for_each_tile_move_ignoring_guard(owner, f);
    }

    /// Iterate all squares reachable by `owner`'s tiles, including
    /// friendly-occupied destinations. No heap allocation.
    #[inline]
    pub fn for_each_reach_ignoring_friendly<F: FnMut(Coordinates, Coordinates)>(&self, owner: Owner, f: F) {
        self.board.for_each_reach_ignoring_friendly(owner, f);
    }

    pub fn as_single_string(&self) -> String {
        single_char_print_state(&self)
    }

    pub fn as_double_string(&self) -> String {
        double_char_print_state(&self)
    }

    pub fn discard_bag_for(&self, o: Owner) -> &DiscardBag {
        match o {
            Owner::TopPlayer => &self.top_player_discard,
            Owner::BottomPlayer => &self.bottom_player_discard,
        }
    }

    fn discard_bag_for_mut(&mut self, o: Owner) -> &mut DiscardBag {
        match o {
            Owner::TopPlayer => &mut self.top_player_discard,
            Owner::BottomPlayer => &mut self.bottom_player_discard,
        }
    }

    pub fn is_duke_in_guard(&self, o: Owner) -> bool {
        self.board.is_guard(o)
    }

    #[inline]
    pub fn bag_for_owner(&self, o: Owner) -> &TileBag {
        match o {
            Owner::TopPlayer => &self.top_player_bag,
            Owner::BottomPlayer => &self.bottom_player_bag,
        }
    }

    #[inline]
    pub fn bag_for_current_player(&self) -> &TileBag {
        self.bag_for_owner(self.current_player_turn)
    }

    pub fn bag_for_other_player(&self) -> &TileBag {
        self.bag_for_owner(self.current_player_turn.next_player())
    }

    pub fn to_undo(&self, mv: &GameMove) -> PossibleMove {
        match mv {
            GameMove::PlaceNewTile(o) => PossibleMove::PlaceNewTile(*o, self.current_player_turn),
            GameMove::PullAndPlay(o) => self.to_undo(&GameMove::PlaceNewTile(*o)),
            GameMove::ApplyNonCommandTileAction { src, dst } => {
                // TODO handle duplication with the other place where capturing is extracted.
                let capturing = self.board.get_board().get(*dst);
                PossibleMove::ApplyNonCommandTileAction {
                    src: *src,
                    dst: *dst,
                    capturing: capturing.cloned(),
                }
            }
        }
    }

    #[inline]
    fn pop_moves_stack(&mut self) -> () {
        debug_assert!(self.idle_stack_len >= 2, "pop_moves_stack: stack underflow");
        self.idle_stack_pop();
    }
    #[inline]
    pub fn undo(&mut self, mv: PossibleMove) -> () {
        self.current_player_turn = self.current_player_turn.next_player();
        // If the move was a capture, remove the captured tile from its owner's discard pile.
        if let PossibleMove::ApplyNonCommandTileAction { capturing: Some(ref cap), .. } = mv {
            self.discard_bag_for_mut(cap.owner)
                .remove(cap.tile_type);
        }
        match &mv {
            PossibleMove::PlaceNewTile(_, _) => self.pop_moves_stack(),
            PossibleMove::ApplyNonCommandTileAction { capturing, .. } => {
                if capturing.is_some() {
                    self.pop_moves_stack()
                } else {
                    let moves = self.idle_stack_last_mut();
                    debug_assert!(*moves >= 1, "undo non-capture: idle count underflow");
                    *moves -= 1;
                }
            }
        }
        if let Some(t) = self.board.undo(mv) {
            // Pop the second stack entry: placements push twice (once for the
            // "pull from bag" step, once for the "place tile" step inside
            // make_a_move).  Both PullAndPlay and pull_specific_tile_from_bag +
            // PlaceNewTile follow this two-push convention.
            self.pop_moves_stack();
            let bag = match self.current_player_turn {
                Owner::TopPlayer => &mut self.top_player_bag,
                Owner::BottomPlayer => &mut self.bottom_player_bag,
            };
            assert_eq!(t.owner, self.current_player_turn);
            assert_eq!(t.current_side, CurrentSide::Initial);
            bag.push(t.tile_type);
        }
    }

    pub fn get(&self, c: Coordinates) -> Option<&PlacedTile> {
        self.board.get(c)
    }
}

impl Hash for GameState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.board.get_board().hash(state);
        self.pulled_tile.hash(state);
        self.current_player_turn.hash(state);
        // Bags: hash sorted contents since swap_remove may change ordering.
        // Use stack-allocated arrays (max 12 tiles) to avoid heap allocation.
        fn hash_bag_sorted<H2: Hasher>(bag: &TileBag, state: &mut H2) {
            let remaining = bag.remaining();
            let mut buf = [0u8; 12];
            let len = remaining.len();
            for (i, t) in remaining.iter().enumerate() {
                buf[i] = *t as u8;
            }
            buf[..len].sort_unstable();
            buf[..len].hash(state);
        }
        hash_bag_sorted(&self.top_player_bag, state);
        hash_bag_sorted(&self.bottom_player_bag, state);
        self.top_player_discard.existing().hash(state);
        self.bottom_player_discard.existing().hash(state);
        self.idle_stack[..self.idle_stack_len as usize].hash(state);
    }
}

impl Rectangular for GameState {
    fn width(&self) -> u8 {
        self.board.width()
    }

    fn height(&self) -> u8 {
        self.board.height()
    }
}

#[cfg(test)]
mod tests {
    use crate::{assert_empty, assert_eq_set};
    use crate::common::utils::test_rng;
    use crate::game::units;

    use super::*;

    #[test]
    fn get_legal_moves_does_not_allow_the_duke_to_remain_in_guard() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        let footman_coordinates = Coordinates { x: 2, y: 2 };
        board.place(footman_coordinates, PlacedTile::new(Owner::TopPlayer, TileType::Footman));
        board.place(Coordinates { x: 0, y: 1 }, PlacedTile::new(Owner::BottomPlayer, TileType::Footman));

        assert_empty!(GameState::from_board(board, Owner::TopPlayer).get_legal_moves(footman_coordinates));
    }

    #[test]
    fn get_legal_moves_does_not_allow_the_duke_to_move_into_guard() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 0, y: 0 };
        board.place(duke_coordinates, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        board.place(Coordinates { x: 3, y: 1 }, PlacedTile::new(Owner::BottomPlayer, TileType::Footman));

        assert_eq!(
            vec!(
                (Coordinates { x: 1, y: 0 }, TileAction::Slide),
                (Coordinates { x: 2, y: 0 }, TileAction::Slide),
                (Coordinates { x: 4, y: 0 }, TileAction::Slide),
                (Coordinates { x: 5, y: 0 }, TileAction::Slide),
            ),
            GameState::from_board(board, Owner::TopPlayer).get_legal_moves(duke_coordinates),
        );
    }

    #[test]
    pub fn can_make_a_move_returns_false_if_moving_duke_into_guard() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 0, y: 0 };
        board.place(duke_coordinates, units::place_tile(Owner::TopPlayer, units::duke));
        board.place(
            Coordinates { x: 3, y: 1 },
            units::place_tile(Owner::BottomPlayer, units::footman),
        );
        let mut state = GameState::from_board(board, Owner::TopPlayer);
        assert_not!(state.can_make_a_move(
            &GameMove::ApplyNonCommandTileAction { src: duke_coordinates, dst: Coordinates { x: 3, y: 0 }}));
    }

    #[test]
    fn can_not_pull_from_bag_if_not_tile_removes_duke_from_guard_move_threat() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 0, y: 0 };
        board.place(duke_coordinates, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        let bag = TileBag::new(vec!(TileType::Footman));
        board.place(
            Coordinates { x: 0, y: 1 },
            PlacedTile::new(Owner::BottomPlayer, TileType::Footman),
        );

        assert_eq!(
            CanPullNewTileResult::DukeAlwaysInGuard,
            GameState::from_board_with_bag(board, Owner::TopPlayer, bag).can_pull_tile_from_bag(),
        );
    }

    #[test]
    fn can_not_pull_from_bag_if_not_tile_removes_duke_from_guard_jump() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 0, y: 0 };
        board.place(duke_coordinates, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        let bag = TileBag::new(vec!(TileType::Footman));
        board.place(
            Coordinates { x: 0, y: 2 },
            PlacedTile::new(Owner::BottomPlayer, TileType::Champion),
        );

        assert_eq!(
            CanPullNewTileResult::DukeAlwaysInGuard,
            GameState::from_board_with_bag(board, Owner::TopPlayer, bag).can_pull_tile_from_bag(),
        );
    }

    #[test]
    fn can_pull_from_bag_returns_correct_value_if_duke_is_in_guard() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 0, y: 0 };
        board.place(duke_coordinates, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        let bag = TileBag::new(vec!(TileType::Footman));
        let mut opposite_duke = PlacedTile::new(Owner::BottomPlayer, TileType::Duke);
        opposite_duke.flip();
        board.place(Coordinates { x: 0, y: 5 }, opposite_duke);

        let mut state = GameState::from_board_with_bag(board, Owner::TopPlayer, bag);
        assert_eq!( // Place in general is allowed...
                    CanPullNewTileResult::OK,
                    state.can_pull_tile_from_bag(),
        );

        state.pull_tile_from_bag(&mut test_rng());
        assert_not!( // But only below the current duke, since the current duke is in guard.
            state.is_valid_placement(DukeOffset::Right)
        );
        assert!( // Bottom blocks the enemy duke.
                 state.is_valid_placement(DukeOffset::Bottom)
        );
    }

    #[test]
    fn can_place_does_not_allow_the_duke_to_move_into_guard() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 0, y: 0 };
        board.place(duke_coordinates, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        let bag = TileBag::new(vec!(TileType::Footman));
        let mut opposite_duke = PlacedTile::new(Owner::BottomPlayer, TileType::Duke);
        opposite_duke.flip();
        board.place(Coordinates { x: 0, y: 5 }, opposite_duke);

        let mut state = GameState::from_board_with_bag(board, Owner::TopPlayer, bag);
        assert_eq!( // Place in general is allowed...
                    CanPullNewTileResult::OK,
                    state.can_pull_tile_from_bag(),
        );

        state.pull_tile_from_bag(&mut test_rng());
        assert_not!( // But only below the current duke, since the current duke is in guard.
            state.is_valid_placement(DukeOffset::Right)
        );
        assert!( // Bottom blocks the enemy duke.
                 state.is_valid_placement(DukeOffset::Bottom)
        );
    }

    #[test]
    fn all_valid_game_moves_returns_all_valid_moves() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 2, y: 0 };
        board.place(duke_coordinates, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        let footman_coordinates = Coordinates { x: 5, y: 0 };
        board.place(footman_coordinates, PlacedTile::new(Owner::TopPlayer, TileType::Footman));
        let bag = TileBag::new(vec!(TileType::Footman));
        let mut opposite_duke = PlacedTile::new(Owner::BottomPlayer, TileType::Duke);
        opposite_duke.flip();
        board.place(Coordinates { x: 0, y: 5 }, opposite_duke);

        let mut state = GameState::from_board_with_bag(board, Owner::TopPlayer, bag);
        assert_eq_set!(
            vec!(
                PossibleMove::PlaceNewTile(DukeOffset::Right, Owner::TopPlayer),
                PossibleMove::PlaceNewTile(DukeOffset::Left, Owner::TopPlayer),
                PossibleMove::PlaceNewTile(DukeOffset::Bottom, Owner::TopPlayer),

                PossibleMove::ApplyNonCommandTileAction {
                    src: duke_coordinates, dst: Coordinates {x: 1, y: 0}, capturing: None},
                PossibleMove::ApplyNonCommandTileAction {
                    src: duke_coordinates, dst: Coordinates {x: 3, y: 0}, capturing: None},
                PossibleMove::ApplyNonCommandTileAction {
                    src: duke_coordinates, dst: Coordinates {x: 4, y: 0}, capturing: None},

                PossibleMove::ApplyNonCommandTileAction {
                    src: footman_coordinates, dst: Coordinates {x: 5, y: 1}, capturing: None},
                PossibleMove::ApplyNonCommandTileAction {
                    src: footman_coordinates, dst: Coordinates {x: 4, y: 0}, capturing: None},
            ),
            state.all_valid_game_moves_for_current_player(),
        )
    }

    fn test_undo_move(mut state: GameState, mv: GameMove) {
        let undo = state.to_undo(&mv);
        let expected = state.clone();
        state.make_a_move(mv, &mut test_rng());
        state.undo(undo);
        assert_eq!(
            expected,
            state,
        )
    }

    #[test]
    fn undo_can_undo_a_pull() {
        test_undo_move(
            GameState::new(
                &TileBag::new(vec!(TileType::Knight)),
                (DukeInitialLocation::Right, FootmenSetup::Sides),
                (DukeInitialLocation::Left, FootmenSetup::Right),
            ),
            GameMove::PullAndPlay(DukeOffset::Bottom),
        )
    }

    #[test]
    fn undo_can_undo_a_move_without_capture() {
        test_undo_move(
            GameState::new(
                &TileBag::new(vec!(TileType::Knight)),
                (DukeInitialLocation::Right, FootmenSetup::Sides),
                (DukeInitialLocation::Left, FootmenSetup::Right),
            ),
            GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 1, y: 0 },
                dst: Coordinates { x: 1, y: 1 },
            },
        );
    }

    #[test]
    fn undo_can_undo_a_move_with_capture() {
        let mut gs = GameState::new(
            &TileBag::new(vec!(TileType::Knight)),
            (DukeInitialLocation::Right, FootmenSetup::Sides),
            (DukeInitialLocation::Left, FootmenSetup::Right),
        );
        let dst = Coordinates { x: 1, y: 1 };
        gs.board.place(dst, PlacedTile::new(Owner::BottomPlayer, TileType::Footman));
        test_undo_move(
            gs,
            GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 1, y: 0 },
                dst,
            },
        );
    }

    #[test]
    fn undo_can_undo_a_strike_move_with_capture() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        let mut pikeman = PlacedTile::new(Owner::TopPlayer, TileType::Pikeman);
        pikeman.flip();
        board.place(Coordinates { x: 1, y: 0 }, pikeman);
        let footman_coordinates = Coordinates { x: 2, y: 2 };
        board.place(footman_coordinates, PlacedTile::new(Owner::BottomPlayer, TileType::Footman));
        let gs = GameState::from_board(board, Owner::TopPlayer);
        test_undo_move(
            gs,
            GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 1, y: 0 },
                dst: footman_coordinates,
            },
        );
    }

    #[test]
    fn game_result_should_return_ongoing_if_no_winner_nor_tie() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::BottomPlayer, TileType::Duke));
        assert_eq!(
            GameState::from_board(board, Owner::TopPlayer).game_result(),
            GameResult::Ongoing,
        );
    }

    #[test]
    fn game_result_should_return_ongoing_if_other_player_still_has_moves() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        let mut footman = PlacedTile::new(Owner::TopPlayer, TileType::Footman);
        footman.flip();
        board.place(Coordinates { x: 4, y: 5 }, footman);
        let mut op_duke = PlacedTile::new(Owner::BottomPlayer, TileType::Duke);
        op_duke.flip();
        board.place(Coordinates { x: 5, y: 0 }, op_duke);
        // TopPlayer can still play a footman move
        assert_eq!(
            GameState::from_board(board, Owner::TopPlayer).game_result(),
            GameResult::Ongoing,
        );
    }

    #[test]
    fn game_result_should_return_some_on_winner() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        let footman = PlacedTile::new(Owner::TopPlayer, TileType::Footman);
        board.place(Coordinates { x: 4, y: 5 }, footman);
        let mut op_duke = PlacedTile::new(Owner::BottomPlayer, TileType::Duke);
        op_duke.flip();
        board.place(Coordinates { x: 5, y: 0 }, op_duke);
        assert_eq!(
            GameState::from_board(board, Owner::TopPlayer).game_result(),
            GameResult::Won(Owner::BottomPlayer),
        );
    }

    #[test]
    fn game_result_should_return_tie_after_enough_consecutive_moves_with_no_capture_or_placement() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::BottomPlayer, TileType::Duke));
        let mut gs = GameState::from_board(board, Owner::TopPlayer);
        // TODO base this count on the maximum const
        for _ in 0..7 {
            gs.make_a_move(GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 0, y: 0 },
                dst: Coordinates { x: 5, y: 0 },
            }, &mut test_rng());
            gs.make_a_move(GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 5, y: 5 },
                dst: Coordinates { x: 0, y: 5 },
            }, &mut test_rng());
            gs.make_a_move(GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 5, y: 0 },
                dst: Coordinates { x: 5, y: 5 },
            }, &mut test_rng());
            gs.make_a_move(GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 0, y: 5 },
                dst: Coordinates { x: 0, y: 0 },
            }, &mut test_rng());
            gs.make_a_move(GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 5, y: 5 },
                dst: Coordinates { x: 0, y: 5 },
            }, &mut test_rng());
            gs.make_a_move(GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 0, y: 0 },
                dst: Coordinates { x: 5, y: 0 },
            }, &mut test_rng());
            gs.make_a_move(GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 0, y: 5 },
                dst: Coordinates { x: 0, y: 0 },
            }, &mut test_rng());
            gs.make_a_move(GameMove::ApplyNonCommandTileAction {
                src: Coordinates { x: 5, y: 0 },
                dst: Coordinates { x: 5, y: 5 },
            }, &mut test_rng());
        }
        assert_eq!(gs.game_result(), GameResult::Tie)
    }

    #[test]
    #[should_panic]
    fn regression_moving_a_duke_in_footman_attack_panics() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, units::place_tile(Owner::TopPlayer, units::duke));
        board.place(
            Coordinates { x: 2, y: 1 },
            units::place_tile_flipped(Owner::TopPlayer, units::footman),
        );
        let duke_coordinates = Coordinates { x: 1, y: 5 };
        board.place(duke_coordinates, units::place_tile_flipped(Owner::BottomPlayer, units::duke));
        let mut gs = GameState::from_board(board, Owner::BottomPlayer);
        gs.make_a_move(GameMove::ApplyNonCommandTileAction {
            src: duke_coordinates,
            dst: Coordinates { x: 1, y: 2 },
        }, &mut test_rng());
    }

    #[test]
    fn all_valid_game_moves_for_uses_specified_players_bag_not_current_players() {
        // Regression: all_valid_game_moves_for(other_player) previously used the
        // current player's bag instead of the specified player's bag.
        //
        // Setup: TopPlayer (current) has an empty bag, BottomPlayer has a non-empty bag.
        // Calling all_valid_game_moves_for(BottomPlayer) should include placement moves
        // because BottomPlayer's bag is non-empty.
        let top_duke_pos = Coordinates { x: 0, y: 0 };
        let bottom_duke_pos = Coordinates { x: 5, y: 5 };
        let tiles = vec![
            (top_duke_pos, PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (bottom_duke_pos, PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
        ];
        let top_bag = TileBag::empty();
        let bottom_bag = TileBag::new(vec![TileType::Footman]);

        let mut state = GameState::from_snapshot(GameSnapshot {
            tiles,
            current_turn: Owner::TopPlayer,
            top_bag,
            bottom_bag,
            top_discard: DiscardBag::empty(),
            bottom_discard: DiscardBag::empty(),
            idle_move_count: 0,
        });

        // TopPlayer's moves should have no placement (empty bag)
        let top_moves: Vec<PossibleMove> =
            state.all_valid_game_moves_for(Owner::TopPlayer).collect();
        assert!(
            !top_moves.iter().any(|m| matches!(m, PossibleMove::PlaceNewTile(..))),
            "TopPlayer has empty bag, should have no placement moves",
        );

        // BottomPlayer's moves should include placements (non-empty bag)
        let bottom_moves: Vec<PossibleMove> =
            state.all_valid_game_moves_for(Owner::BottomPlayer).collect();
        assert!(
            bottom_moves.iter().any(|m| matches!(m, PossibleMove::PlaceNewTile(..))),
            "BottomPlayer has tiles in bag, should have placement moves, but got: {:?}",
            bottom_moves,
        );
    }

    // ── Discard pile capture tests ────────────────────────────────────

    #[test]
    fn discard_pile_empty_when_no_captures() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::BottomPlayer, TileType::Duke));
        let gs = GameState::from_board(board, Owner::TopPlayer);
        assert!(gs.player_1_discard().existing().is_empty());
        assert!(gs.player_2_discard().existing().is_empty());
    }

    #[test]
    fn capture_by_move_adds_tile_to_owners_discard() {
        // TopPlayer footman at (1,0) captures BottomPlayer footman at (1,1)
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        board.place(Coordinates { x: 1, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Footman));
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::BottomPlayer, TileType::Duke));
        board.place(Coordinates { x: 1, y: 1 }, PlacedTile::new(Owner::BottomPlayer, TileType::Footman));

        let mut gs = GameState::from_board(board, Owner::TopPlayer);
        assert!(gs.player_2_discard().existing().is_empty());

        gs.make_a_move(GameMove::ApplyNonCommandTileAction {
            src: Coordinates { x: 1, y: 0 },
            dst: Coordinates { x: 1, y: 1 },
        }, &mut test_rng());

        // BottomPlayer's Footman was captured -> goes to BottomPlayer's discard
        assert_eq!(gs.player_2_discard().existing(), &vec![TileType::Footman]);
        // TopPlayer's discard remains empty (they didn't lose any pieces)
        assert!(gs.player_1_discard().existing().is_empty());
    }

    #[test]
    fn capture_by_strike_adds_tile_to_owners_discard() {
        // Flipped Pikeman can strike diagonally at distance 2
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        let mut pikeman = PlacedTile::new(Owner::TopPlayer, TileType::Pikeman);
        pikeman.flip();
        board.place(Coordinates { x: 1, y: 0 }, pikeman);
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::BottomPlayer, TileType::Duke));
        board.place(Coordinates { x: 2, y: 2 }, PlacedTile::new(Owner::BottomPlayer, TileType::Footman));

        let mut gs = GameState::from_board(board, Owner::TopPlayer);
        gs.make_a_move(GameMove::ApplyNonCommandTileAction {
            src: Coordinates { x: 1, y: 0 },
            dst: Coordinates { x: 2, y: 2 },
        }, &mut test_rng());

        // BottomPlayer's Footman was struck -> goes to BottomPlayer's discard
        assert_eq!(gs.player_2_discard().existing(), &vec![TileType::Footman]);
        // TopPlayer didn't lose any pieces
        assert!(gs.player_1_discard().existing().is_empty());
    }

    #[test]
    fn multiple_captures_accumulate_in_discard() {
        // Set up so TopPlayer can capture twice in succession.
        // Duke side-A slides horizontally, so BottomPlayer duke can slide along row 5.
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        board.place(Coordinates { x: 2, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Footman));
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::BottomPlayer, TileType::Duke));
        // Two enemy pieces to capture
        board.place(Coordinates { x: 2, y: 1 }, PlacedTile::new(Owner::BottomPlayer, TileType::Footman));
        board.place(Coordinates { x: 3, y: 0 }, PlacedTile::new(Owner::BottomPlayer, TileType::Knight));

        let mut gs = GameState::from_board(board, Owner::TopPlayer);

        // First capture: TopPlayer footman captures enemy footman
        gs.make_a_move(GameMove::ApplyNonCommandTileAction {
            src: Coordinates { x: 2, y: 0 },
            dst: Coordinates { x: 2, y: 1 },
        }, &mut test_rng());
        // BottomPlayer lost a footman -> goes to BottomPlayer's discard
        assert_eq!(gs.player_2_discard().len(), 1);
        assert_eq!(gs.player_1_discard().len(), 0);

        // BottomPlayer makes a non-capture move (duke slides horizontally along row 5)
        gs.make_a_move(GameMove::ApplyNonCommandTileAction {
            src: Coordinates { x: 5, y: 5 },
            dst: Coordinates { x: 0, y: 5 },
        }, &mut test_rng());

        // TopPlayer duke slides to capture enemy knight at (3,0)
        gs.make_a_move(GameMove::ApplyNonCommandTileAction {
            src: Coordinates { x: 0, y: 0 },
            dst: Coordinates { x: 3, y: 0 },
        }, &mut test_rng());

        // BottomPlayer lost both a footman and a knight
        assert_eq!(gs.player_2_discard().len(), 2);
        let discard = gs.player_2_discard().existing();
        assert!(discard.contains(&TileType::Footman));
        assert!(discard.contains(&TileType::Knight));
        // TopPlayer didn't lose any
        assert_eq!(gs.player_1_discard().len(), 0);
    }

    // ── Make/undo symmetry and invariant tests ───────────────────────

    /// Play every legal non-placement move and undo it, verifying state is restored exactly.
    /// (Placement moves use swap_remove in the bag, so bag element order may change
    /// after pull+undo. This is semantically correct since bag order is irrelevant
    /// for random draws, but causes PartialEq to fail.)
    #[test]
    fn undo_symmetry_every_non_placement_move_from_initial_position() {
        let bag = TileBag::new(vec![TileType::Knight, TileType::Pikeman, TileType::Champion]);
        let gs = GameState::new(
            &bag,
            (DukeInitialLocation::Left, FootmenSetup::Left),
            (DukeInitialLocation::Right, FootmenSetup::Right),
        );
        let mut gs_mut = gs.clone();
        let moves: Vec<PossibleMove> = gs_mut.all_valid_game_moves_for_current_player().collect();
        assert!(!moves.is_empty(), "Initial position should have legal moves");

        for mv in &moves {
            // Skip placement moves (bag ordering is not preserved by swap_remove + push)
            if matches!(mv, PossibleMove::PlaceNewTile(..)) {
                continue;
            }
            let gm: GameMove = mv.into();
            let undo = gs_mut.to_undo(&gm);
            gs_mut.make_a_move(gm, &mut test_rng());
            gs_mut.undo(undo);
            assert_eq!(gs, gs_mut, "State not restored after undo of {:?}", mv);
        }
    }

    /// Play every legal placement move and undo it, verifying board and discard piles
    /// are restored (bag ordering may differ due to swap_remove, so we compare sorted bags).
    #[test]
    fn undo_symmetry_placement_moves_restore_board_and_sorted_bag() {
        let bag = TileBag::new(vec![TileType::Knight, TileType::Pikeman, TileType::Champion]);
        let gs = GameState::new(
            &bag,
            (DukeInitialLocation::Left, FootmenSetup::Left),
            (DukeInitialLocation::Right, FootmenSetup::Right),
        );
        let mut gs_mut = gs.clone();
        let moves: Vec<PossibleMove> = gs_mut.all_valid_game_moves_for_current_player().collect();

        for mv in &moves {
            if !matches!(mv, PossibleMove::PlaceNewTile(..)) {
                continue;
            }
            let gm: GameMove = mv.into();
            let undo = gs_mut.to_undo(&gm);
            gs_mut.make_a_move(gm, &mut test_rng());
            gs_mut.undo(undo);
            // Board and discard should match exactly
            assert_eq!(gs.board(), gs_mut.board(), "Board not restored after undo of {:?}", mv);
            assert_eq!(gs.player_1_discard(), gs_mut.player_1_discard());
            assert_eq!(gs.player_2_discard(), gs_mut.player_2_discard());
            assert_eq!(gs.current_player_turn(), gs_mut.current_player_turn());
            // Bag contents should be the same (order may differ)
            let mut expected_bag: Vec<TileType> = gs.top_player_bag().remaining().to_vec();
            expected_bag.sort_by_key(|t| t.index());
            let mut actual_bag: Vec<TileType> = gs_mut.top_player_bag().remaining().to_vec();
            actual_bag.sort_by_key(|t| t.index());
            assert_eq!(expected_bag, actual_bag, "Bag contents differ after undo of {:?}", mv);
            // Reset for next iteration
            gs_mut = gs.clone();
        }
    }

    /// Play multiple moves then undo all, verifying full roundtrip.
    #[test]
    fn undo_chain_restores_original_state() {
        let bag = TileBag::new(vec![TileType::Knight]);
        let gs = GameState::new(
            &bag,
            (DukeInitialLocation::Right, FootmenSetup::Sides),
            (DukeInitialLocation::Left, FootmenSetup::Right),
        );
        let mut gs_mut = gs.clone();
        let mut undo_stack: Vec<PossibleMove> = Vec::new();

        // Play 4 moves
        for _ in 0..4 {
            let moves: Vec<PossibleMove> = gs_mut.all_valid_game_moves_for_current_player().collect();
            let mv = moves.first().expect("Should have moves").clone();
            let gm: GameMove = (&mv).into();
            let undo = gs_mut.to_undo(&gm);
            gs_mut.make_a_move(gm, &mut test_rng());
            undo_stack.push(undo);
        }

        // Undo all 4
        while let Some(undo) = undo_stack.pop() {
            gs_mut.undo(undo);
        }

        assert_eq!(gs, gs_mut, "State not restored after undoing all moves");
    }

    /// After every legal move, the total tile count (board + bags + discards)
    /// should remain constant.
    #[test]
    fn tile_count_invariant_after_moves() {
        let bag = TileBag::new(vec![TileType::Knight, TileType::Pikeman]);
        let mut gs = GameState::new(
            &bag,
            (DukeInitialLocation::Left, FootmenSetup::Left),
            (DukeInitialLocation::Right, FootmenSetup::Right),
        );

        let count_all_tiles = |gs: &GameState| -> usize {
            let board_tiles = gs.board().active_coordinates().count();
            let top_bag = gs.top_player_bag().remaining().len();
            let bot_bag = gs.bottom_player_bag().remaining().len();
            let top_dis = gs.player_1_discard().len();
            let bot_dis = gs.player_2_discard().len();
            board_tiles + top_bag + bot_bag + top_dis + bot_dis
        };

        let initial_count = count_all_tiles(&gs);

        // Play several random moves
        let ai = crate::game::ai::stupid_sync_ai::StupidSyncAi {};
        for _ in 0..20 {
            if gs.game_result() != GameResult::Ongoing {
                break;
            }
            crate::game::ai::player::ArtificialPlayer::play_next_move(&ai, &mut test_rng(), &mut gs);
            assert_eq!(
                count_all_tiles(&gs), initial_count,
                "Total tile count changed after a move!",
            );
        }
    }

    /// Undo of a capture must restore the discard pile exactly.
    #[test]
    fn undo_capture_restores_discard_pile() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        board.place(Coordinates { x: 1, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Footman));
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::BottomPlayer, TileType::Duke));
        board.place(Coordinates { x: 1, y: 1 }, PlacedTile::new(Owner::BottomPlayer, TileType::Footman));

        let gs = GameState::from_board(board, Owner::TopPlayer);
        let mut gs_mut = gs.clone();

        let gm = GameMove::ApplyNonCommandTileAction {
            src: Coordinates { x: 1, y: 0 },
            dst: Coordinates { x: 1, y: 1 },
        };
        let undo = gs_mut.to_undo(&gm);
        gs_mut.make_a_move(gm, &mut test_rng());

        // After capture, discard should have 1 tile
        assert_eq!(gs_mut.player_2_discard().len(), 1);

        gs_mut.undo(undo);

        // After undo, discard should be empty again
        assert_eq!(gs_mut.player_2_discard().len(), 0);
        assert_eq!(gs, gs_mut, "State not restored after undo of capture");
    }

    // ── Hash consistency test ────────────────────────────────────────

    /// GameState Hash only includes the board (not player turn), which is
    /// intentional for transposition-table-style lookups. Verify the documented
    /// Hash includes all fields that PartialEq compares (board, turn, bags, discards).
    /// States that differ in any field should (usually) have different hashes.
    #[test]
    fn hash_consistent_with_partial_eq() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let tiles = vec![
            (Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
        ];
        let gs1 = GameState::from_snapshot(GameSnapshot {
            tiles: tiles.clone(),
            current_turn: Owner::TopPlayer,
            top_bag: TileBag::new(vec![]),
            bottom_bag: TileBag::new(vec![]),
            top_discard: DiscardBag::empty(),
            bottom_discard: DiscardBag::empty(),
            idle_move_count: 0,
        });
        let gs2 = GameState::from_snapshot(GameSnapshot {
            tiles: tiles.clone(),
            current_turn: Owner::BottomPlayer,
            top_bag: TileBag::new(vec![TileType::Footman]),
            bottom_bag: TileBag::new(vec![]),
            top_discard: DiscardBag::empty(),
            bottom_discard: DiscardBag::empty(),
            idle_move_count: 0,
        });
        // Same state should have same hash
        let gs1_copy = gs1.clone();

        let hash = |gs: &GameState| {
            let mut h = DefaultHasher::new();
            gs.hash(&mut h);
            h.finish()
        };

        // Equal states -> equal hash (required by Hash contract)
        assert_eq!(hash(&gs1), hash(&gs1_copy));
        // Different states -> different hash (not guaranteed but expected)
        assert_ne!(hash(&gs1), hash(&gs2));
        // And they're not equal via PartialEq either
        assert_ne!(gs1, gs2);
    }

    /// Two states that differ only in their idle-move counter should have
    /// different hashes, since the derived PartialEq considers them unequal.
    /// Regression test: the Hash impl previously omitted the
    /// `idle_stack` field, causing states near
    /// and far from a tie draw to collide in hash-based data structures.
    #[test]
    fn hash_differs_when_idle_move_count_differs() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let tiles = vec![
            (Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke)),
            (Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::BottomPlayer, TileType::Duke)),
        ];
        let gs_idle0 = GameState::from_snapshot(GameSnapshot {
            tiles: tiles.clone(),
            current_turn: Owner::TopPlayer,
            top_bag: TileBag::new(vec![]),
            bottom_bag: TileBag::new(vec![]),
            top_discard: DiscardBag::empty(),
            bottom_discard: DiscardBag::empty(),
            idle_move_count: 0,
        });
        let gs_idle5 = GameState::from_snapshot(GameSnapshot {
            tiles: tiles.clone(),
            current_turn: Owner::TopPlayer,
            top_bag: TileBag::new(vec![]),
            bottom_bag: TileBag::new(vec![]),
            top_discard: DiscardBag::empty(),
            bottom_discard: DiscardBag::empty(),
            idle_move_count: 5,
        });

        // They should be unequal via PartialEq
        assert_ne!(gs_idle0, gs_idle5,
            "States with different idle_move_count should be PartialEq::ne");

        // And they should hash differently (regression: previously they collided)
        let hash = |gs: &GameState| {
            let mut h = DefaultHasher::new();
            gs.hash(&mut h);
            h.finish()
        };
        assert_ne!(hash(&gs_idle0), hash(&gs_idle5),
            "States with different idle_move_count should produce different hashes");
    }

    /// Regression test: pull_specific_tile_from_bag + make_a_move(PlaceNewTile)
    /// must be undoable without corrupting the idle-move stack.  Previously,
    /// pull_specific_tile_from_bag did not push to the stack, so undo's
    /// double-pop would corrupt a previous entry.
    #[test]
    fn undo_pull_specific_then_place_preserves_state() {
        let bag = TileBag::new(vec![TileType::Knight, TileType::Pikeman, TileType::Champion]);
        let gs = GameState::new(
            &bag,
            (DukeInitialLocation::Left, FootmenSetup::Left),
            (DukeInitialLocation::Right, FootmenSetup::Right),
        );
        let mut gs_mut = gs.clone();

        // Find a valid placement offset from legal moves.
        let moves: Vec<PossibleMove> = gs_mut.all_valid_game_moves_for_current_player().collect();
        let offset = moves.iter().find_map(|m| match m {
            PossibleMove::PlaceNewTile(o, _) => Some(*o),
            _ => None,
        }).expect("Should have at least one placement move");

        // Pull a specific tile and place it (mimics negamax expectimax path).
        let tile_type = TileType::Knight;
        gs_mut.pull_specific_tile_from_bag(tile_type);
        let undo = PossibleMove::PlaceNewTile(offset, gs_mut.current_player_turn());
        gs_mut.make_a_move(GameMove::PlaceNewTile(offset), &mut test_rng());
        gs_mut.undo(undo);

        // Board, discards, player turn should match exactly.
        assert_eq!(gs.board(), gs_mut.board(),
            "Board not restored after undo of pull_specific + PlaceNewTile");
        assert_eq!(gs.current_player_turn(), gs_mut.current_player_turn());
        assert_eq!(gs.player_1_discard(), gs_mut.player_1_discard());
        assert_eq!(gs.player_2_discard(), gs_mut.player_2_discard());

        // Bag contents should match (order may differ due to swap_remove + push).
        let mut expected: Vec<TileType> = gs.top_player_bag().remaining().to_vec();
        expected.sort_by_key(|t| t.index());
        let mut actual: Vec<TileType> = gs_mut.top_player_bag().remaining().to_vec();
        actual.sort_by_key(|t| t.index());
        assert_eq!(expected, actual,
            "Bag contents differ after undo of pull_specific + PlaceNewTile");

        // Idle-move stack must match exactly.
        assert_eq!(gs.idle_move_count(), gs_mut.idle_move_count(),
            "Idle move count corrupted after undo of pull_specific + PlaceNewTile");
    }

    /// 4-level nested make/undo chain mimicking depth-4 negamax.
    /// At each level, make a move, recurse, then undo and verify the state
    /// matches the snapshot taken before the move.
    #[test]
    fn deep_undo_chain_preserves_state() {
        let bag = TileBag::new(vec![TileType::Knight, TileType::Pikeman, TileType::Champion]);
        let gs = GameState::new(
            &bag,
            (DukeInitialLocation::Left, FootmenSetup::Left),
            (DukeInitialLocation::Right, FootmenSetup::Right),
        );
        let mut gs_mut = gs.clone();

        fn sorted_bag(bag: &TileBag) -> Vec<TileType> {
            let mut v = bag.remaining().to_vec();
            v.sort_by_key(|t| t.index());
            v
        }

        fn assert_state_equivalent(expected: &GameState, actual: &GameState, label: &str) {
            assert_eq!(expected.board(), actual.board(), "{}: board differs", label);
            assert_eq!(expected.current_player_turn(), actual.current_player_turn(),
                "{}: current_player differs", label);
            assert_eq!(expected.player_1_discard(), actual.player_1_discard(),
                "{}: top discard differs", label);
            assert_eq!(expected.player_2_discard(), actual.player_2_discard(),
                "{}: bottom discard differs", label);
            assert_eq!(sorted_bag(expected.top_player_bag()), sorted_bag(actual.top_player_bag()),
                "{}: top bag differs", label);
            assert_eq!(sorted_bag(expected.bottom_player_bag()), sorted_bag(actual.bottom_player_bag()),
                "{}: bottom bag differs", label);
            assert_eq!(expected.idle_move_count(), actual.idle_move_count(),
                "{}: idle_move_count differs", label);
        }

        // 4-level nested make/undo
        let mut undo_stack: Vec<(GameState, PossibleMove)> = Vec::new();
        for depth in 0..4 {
            if gs_mut.game_result() != GameResult::Ongoing {
                break;
            }
            let snapshot = gs_mut.clone();
            let moves: Vec<PossibleMove> = gs_mut.all_valid_game_moves_for_current_player().collect();
            if moves.is_empty() { break; }
            let mv = moves[0].clone();
            let gm: GameMove = (&mv).into();
            let undo = gs_mut.to_undo(&gm);
            gs_mut.make_a_move(gm, &mut test_rng());
            undo_stack.push((snapshot, undo));

            // At each inner level, test one more make/undo cycle
            if gs_mut.game_result() == GameResult::Ongoing {
                let inner_snapshot = gs_mut.clone();
                let inner_moves: Vec<PossibleMove> = gs_mut.all_valid_game_moves_for_current_player().collect();
                if !inner_moves.is_empty() {
                    let inner_mv = inner_moves[0].clone();
                    let inner_gm: GameMove = (&inner_mv).into();
                    let inner_undo = gs_mut.to_undo(&inner_gm);
                    gs_mut.make_a_move(inner_gm, &mut test_rng());
                    gs_mut.undo(inner_undo);
                    assert_state_equivalent(&inner_snapshot, &gs_mut,
                        &format!("depth {} inner undo", depth));
                }
            }
        }

        // Undo all moves in reverse order
        while let Some((snapshot, undo)) = undo_stack.pop() {
            gs_mut.undo(undo);
            assert_state_equivalent(&snapshot, &gs_mut,
                &format!("undo at stack depth {}", undo_stack.len()));
        }

        // Final state should match the original
        assert_state_equivalent(&gs, &gs_mut, "full chain undo");
    }
}