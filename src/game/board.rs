use std::convert::TryFrom;
use std::fmt::{Display, Formatter};
use std::ops::Range;

use strum::{EnumCount, IntoEnumIterator};
use strum_macros::EnumIter;

use crate::common::board::Board;
use crate::common::coordinates::Coordinates;
use crate::common::geometry::Rectangular;
use crate::common::utils::Folding;
use crate::game::dumb_printer::{double_char_print_board, single_char_print_board};
use crate::game::offset::{Centerable, HorizontalOffset, Offsets, VerticalOffset};
use crate::game::tile::{CurrentSide, Owner, Ownership, PlacedTile, TileType};
use crate::game::tile_side::TileAction;
use crate::game::units::tile_table_lookup;
use crate::time_it_macro;

use std::sync::OnceLock;

/// Precomputed ray masks for obstruction checking on a 6x6 board.
/// `RAY_BETWEEN[src][dst]` is a bitmask of intermediate squares between src and dst
/// on a straight line (horizontal, vertical, or diagonal). 0 if not on a line or adjacent.
static RAY_BETWEEN: [[u64; 36]; 36] = {
    let mut table = [[0u64; 36]; 36];
    let mut src = 0usize;
    while src < 36 {
        let sx = (src % 6) as i32;
        let sy = (src / 6) as i32;
        let mut dst = 0usize;
        while dst < 36 {
            let dx = (dst % 6) as i32;
            let dy = (dst / 6) as i32;
            let diffx = dx - sx;
            let diffy = dy - sy;
            let on_line = diffx == 0 || diffy == 0 || diffx.abs() == diffy.abs();
            if on_line && src != dst {
                let stepx = if diffx > 0 { 1 } else if diffx < 0 { -1 } else { 0 };
                let stepy = if diffy > 0 { 1 } else if diffy < 0 { -1 } else { 0 };
                let mut mask = 0u64;
                let mut cx = sx + stepx;
                let mut cy = sy + stepy;
                while cx != dx || cy != dy {
                    mask |= 1u64 << (cy * 6 + cx);
                    cx += stepx;
                    cy += stepy;
                }
                table[src][dst] = mask;
            }
            dst += 1;
        }
        src += 1;
    }
    table
};

/// Precomputed move data for a specific (tile_type, owner, side, position) combination.
/// Splits targets into categories that can be counted via popcount vs per-target checking.
#[derive(Clone)]
struct PrecomputedMoves {
    /// Bitmask of Jump targets + Move targets that are distance-1 (no obstruction check needed).
    jump_and_near_mask: u64,
    /// Bitmask of Strike targets (count = popcount of mask & opp_occ).
    strike_mask: u64,
    /// Move targets that are distance > 1 (need obstruction + straight-line check).
    /// Stored as board indices (y*6+x).
    far_move: [u8; 8],
    far_move_len: u8,
    /// Slide/JumpSlide targets, stored as (board_index, is_jump_slide).
    slide: [u8; 20],
    slide_action: [u8; 20], // 0 = Slide, 1 = JumpSlide
    slide_len: u8,
}

impl PrecomputedMoves {
    fn empty() -> Self {
        PrecomputedMoves {
            jump_and_near_mask: 0,
            strike_mask: 0,
            far_move: [0; 8],
            far_move_len: 0,
            slide: [0; 20],
            slide_action: [0; 20],
            slide_len: 0,
        }
    }
}

/// Table indexed by [tile_table_idx (26)][side (2)][position (36)].
/// tile_table_idx = tt.index() for BottomPlayer, TileType::COUNT + tt.index() for TopPlayer.
struct MoveTable {
    entries: Vec<PrecomputedMoves>, // flat: idx = tile_idx * 2 * 36 + side * 36 + pos
}

impl MoveTable {
    fn new() -> Self {
        let n_tiles = TileType::COUNT * 2; // 26 (13 per owner)
        let mut entries = vec![PrecomputedMoves::empty(); n_tiles * 2 * 36];

        for tile_idx in 0..n_tiles {
            let tt = TileType::try_from((tile_idx % TileType::COUNT) as u8).unwrap();
            let owner = if tile_idx < TileType::COUNT {
                Owner::BottomPlayer
            } else {
                Owner::TopPlayer
            };
            let tile_ref = tile_table_lookup(tt, owner);

            let sides = [tile_ref.get_side_a(), tile_ref.get_side_b()];
            for (side_idx, tile_side) in sides.iter().enumerate() {
                let center = tile_side.center_offset();

                for pos in 0..36u8 {
                    let src_x = (pos % 6) as u8;
                    let src_y = (pos / 6) as u8;
                    let src = Coordinates { x: src_x, y: src_y };
                    let flat = tile_idx * 2 * 36 + side_idx * 36 + pos as usize;
                    let entry = &mut entries[flat];

                    for (offset, action) in tile_side.actions().iter() {
                        match *action {
                            TileAction::Unit | TileAction::Command => continue,
                            TileAction::Jump => {
                                if let Some(dst) = Self::abs_coord(src, *offset, center) {
                                    entry.jump_and_near_mask |= 1u64 << (dst.y * 6 + dst.x);
                                }
                            }
                            TileAction::Move => {
                                if let Some(dst) = Self::abs_coord(src, *offset, center) {
                                    if !src.is_straight_line_to(dst) {
                                        continue;
                                    }
                                    let si = src_y as usize * 6 + src_x as usize;
                                    let di = dst.y as usize * 6 + dst.x as usize;
                                    if RAY_BETWEEN[si][di] == 0 {
                                        // Distance 1: no obstruction possible.
                                        entry.jump_and_near_mask |= 1u64 << di;
                                    } else {
                                        entry.far_move[entry.far_move_len as usize] = di as u8;
                                        entry.far_move_len += 1;
                                    }
                                }
                            }
                            TileAction::Strike => {
                                if let Some(dst) = Self::abs_coord(src, *offset, center) {
                                    entry.strike_mask |= 1u64 << (dst.y * 6 + dst.x);
                                }
                            }
                            TileAction::Slide | TileAction::JumpSlide => {
                                let eff_action = *action;
                                let eff_offset = if eff_action == TileAction::JumpSlide {
                                    Offsets::new(offset.x.to_near(), offset.y.to_near())
                                } else {
                                    *offset
                                };
                                // Generate slide targets (same logic as target_coordinates for Slide).
                                let targets = Self::slide_targets(src, eff_offset);
                                for dst in targets {
                                    let di = dst.y as usize * 6 + dst.x as usize;
                                    let idx = entry.slide_len as usize;
                                    entry.slide[idx] = di as u8;
                                    entry.slide_action[idx] = if eff_action == TileAction::JumpSlide { 1 } else { 0 };
                                    entry.slide_len += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        MoveTable { entries }
    }

    fn abs_coord(src: Coordinates, offset: Offsets, center: VerticalOffset) -> Option<Coordinates> {
        fn voff(y: VerticalOffset) -> i32 {
            match y {
                VerticalOffset::FarTop => -2, VerticalOffset::Top => -1,
                VerticalOffset::Center => 0, VerticalOffset::Bottom => 1,
                VerticalOffset::FarBottom => 2,
            }
        }
        let x = src.x as i32 + match offset.x {
            HorizontalOffset::FarLeft => -2, HorizontalOffset::Left => -1,
            HorizontalOffset::Center => 0, HorizontalOffset::Right => 1,
            HorizontalOffset::FarRight => 2,
        };
        let y = src.y as i32 + voff(offset.y) + voff(center);
        if x < 0 || x >= 6 || y < 0 || y >= 6 { return None; }
        Some(Coordinates { x: x as u8, y: y as u8 })
    }

    fn slide_targets(src: Coordinates, offset: Offsets) -> Vec<Coordinates> {
        let mut res = Vec::new();
        if offset == HorizontalOffset::Right.center() {
            for x in 0..src.x { res.push(Coordinates { x, y: src.y }); }
        } else if offset == HorizontalOffset::Left.center() {
            for x in src.x + 1..6 { res.push(Coordinates { x, y: src.y }); }
        } else if offset == VerticalOffset::Top.center() {
            for y in 0..src.y { res.push(Coordinates { x: src.x, y }); }
        } else if offset == VerticalOffset::Bottom.center() {
            for y in src.y + 1..6 { res.push(Coordinates { x: src.x, y }); }
        } else if offset == Offsets::new(HorizontalOffset::Right, VerticalOffset::Top) {
            let (mut x, mut y) = (src.x.wrapping_sub(1), src.y.wrapping_sub(1));
            while x < 6 && y < 6 { res.push(Coordinates { x, y }); x = x.wrapping_sub(1); y = y.wrapping_sub(1); }
        } else if offset == Offsets::new(HorizontalOffset::Left, VerticalOffset::Top) {
            let (mut x, mut y) = (src.x + 1, src.y.wrapping_sub(1));
            while x < 6 && y < 6 { res.push(Coordinates { x, y }); x += 1; y = y.wrapping_sub(1); }
        } else if offset == Offsets::new(HorizontalOffset::Right, VerticalOffset::Bottom) {
            let (mut x, mut y) = (src.x.wrapping_sub(1), src.y + 1);
            while x < 6 && y < 6 { res.push(Coordinates { x, y }); x = x.wrapping_sub(1); y += 1; }
        } else if offset == Offsets::new(HorizontalOffset::Left, VerticalOffset::Bottom) {
            let (mut x, mut y) = (src.x + 1, src.y + 1);
            while x < 6 && y < 6 { res.push(Coordinates { x, y }); x += 1; y += 1; }
        }
        res
    }

    #[inline(always)]
    fn lookup(&self, tile_idx: usize, side_idx: usize, pos: usize) -> &PrecomputedMoves {
        &self.entries[tile_idx * 2 * 36 + side_idx * 36 + pos]
    }
}

static MOVE_TABLE: OnceLock<MoveTable> = OnceLock::new();

fn move_table() -> &'static MoveTable {
    MOVE_TABLE.get_or_init(MoveTable::new)
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, EnumIter)]
pub enum DukeOffset { Top, Bottom, Left, Right }

#[derive(Debug, Clone)]
pub(super) enum BoardMove {
    PlaceNewTile(TileType, DukeOffset, Owner),
    ApplyNonCommandTileAction { src: Coordinates, dst: Coordinates },
    // CommandAnotherTile { commander_src: Coordinates, unit_src: Coordinates, unit_dst: Coordinates },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PossibleMove {
    PlaceNewTile(DukeOffset, Owner),
    ApplyNonCommandTileAction { src: Coordinates, dst: Coordinates, capturing: Option<PlacedTile> },
}

impl Display for PossibleMove {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PossibleMove::PlaceNewTile { .. } => write!(f, "{:?}", self),
            PossibleMove::ApplyNonCommandTileAction { src, dst, capturing } => {
                write!(f, "ApplyNonCommandTileAction {{ src: {:?}, dst: {:?}", src, dst)?;
                if let Some(t) = capturing {
                    write!(f, ", capturing: {}", t.tile_type.get_name())?;
                }
                write!(f, "}}")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GameBoard {
    board: Board<PlacedTile>,
    /// Cached duke positions per player. Updated on place/remove/mv to avoid O(36) scans.
    /// None if the duke for that player hasn't been placed yet (only during initial setup).
    duke_cache: [Option<Coordinates>; 2],
    /// Occupancy bitboards for 6x6 board (bits 0..35, row-major: bit = y*6+x).
    occ: u64,
    top_occ: u64,
    bot_occ: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WithNewTiles(pub bool);


/// Maximum number of tiles a single player can have on the board.
/// On a 6x6 board, the theoretical max is 18 (half the cells).
const MAX_TILES_PER_PLAYER: usize = 18;

/// Maximum number of legal moves a single tile can produce.
/// A tile has at most ~12 actions, each producing at most 5 targets (slide on 6x6).
/// In practice the maximum is much lower; 48 provides ample headroom.
const MAX_LEGAL_MOVES_PER_TILE: usize = 48;

/// Stack-allocated buffer for legal moves from a single tile.
#[derive(Clone)]
struct LegalMoveBuffer {
    moves: [(Coordinates, TileAction); MAX_LEGAL_MOVES_PER_TILE],
    len: u8,
}

impl LegalMoveBuffer {
    #[inline(always)]
    fn new() -> Self {
        LegalMoveBuffer {
            moves: [(Coordinates { x: 0, y: 0 }, TileAction::Unit); MAX_LEGAL_MOVES_PER_TILE],
            len: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, c: Coordinates, a: TileAction) {
        debug_assert!((self.len as usize) < MAX_LEGAL_MOVES_PER_TILE, "LegalMoveBuffer overflow");
        self.moves[self.len as usize] = (c, a);
        self.len += 1;
    }

    #[inline(always)]
    fn len(&self) -> usize { self.len as usize }

    #[inline(always)]
    fn iter(&self) -> impl Iterator<Item = (Coordinates, TileAction)> + '_ {
        self.moves[..self.len as usize].iter().copied()
    }

    #[inline(always)]
    fn as_slice(&self) -> &[(Coordinates, TileAction)] {
        &self.moves[..self.len as usize]
    }
}

/// Stack-allocated coordinate buffer for target_coordinates results.
/// Max slide length on a 6x6 board is 5 squares; capacity 6 provides margin.
const MAX_TARGETS: usize = 6;

#[derive(Clone, Copy)]
struct TargetCoords {
    coords: [Coordinates; MAX_TARGETS],
    len: u8,
}

impl TargetCoords {
    #[inline(always)]
    fn empty() -> Self {
        TargetCoords {
            coords: [Coordinates { x: 0, y: 0 }; MAX_TARGETS],
            len: 0,
        }
    }

    #[inline(always)]
    fn from_option(opt: Option<Coordinates>) -> Self {
        match opt {
            None => Self::empty(),
            Some(c) => {
                let mut t = Self::empty();
                t.coords[0] = c;
                t.len = 1;
                t
            }
        }
    }

    #[inline(always)]
    fn push(&mut self, c: Coordinates) {
        debug_assert!((self.len as usize) < MAX_TARGETS, "TargetCoords overflow");
        self.coords[self.len as usize] = c;
        self.len += 1;
    }

    #[inline(always)]
    fn iter(&self) -> impl Iterator<Item = Coordinates> + '_ {
        self.coords[..self.len as usize].iter().copied()
    }

    /// Returns an owned iterator (no borrow on self). Safe because TargetCoords is Copy.
    #[inline(always)]
    fn into_iter(self) -> TargetCoordsIter {
        TargetCoordsIter { inner: self, pos: 0 }
    }
}

#[derive(Clone, Copy)]
struct TargetCoordsIter {
    inner: TargetCoords,
    pos: u8,
}

impl Iterator for TargetCoordsIter {
    type Item = Coordinates;

    #[inline(always)]
    fn next(&mut self) -> Option<Coordinates> {
        if self.pos < self.inner.len {
            let c = self.inner.coords[self.pos as usize];
            self.pos += 1;
            Some(c)
        } else {
            None
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.inner.len - self.pos) as usize;
        (remaining, Some(remaining))
    }
}

/// Stack-allocated buffer for empty spaces near a duke (at most 4 cardinal neighbors).
#[derive(Clone, Copy)]
pub struct DukeNeighbors {
    coords: [Coordinates; 4],
    len: u8,
}

impl DukeNeighbors {
    #[inline(always)]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = Coordinates> + '_ {
        self.coords[..self.len as usize].iter().copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppliedPubAction { Movement, Strike, Invalid }

impl GameBoard {
    pub const BOARD_SIZE: u8 = 6;

    pub(super) fn new(board: Board<PlacedTile>) -> Self {
        let top = board.find(|a| a.owner == Owner::TopPlayer && a.tile_type.is_duke());
        let bottom = board.find(|a| a.owner == Owner::BottomPlayer && a.tile_type.is_duke());
        let mut occ = 0u64;
        let mut top_occ = 0u64;
        let mut bot_occ = 0u64;
        for (c, tile) in board.active_coordinates() {
            let bit = Self::coord_bit(c);
            occ |= bit;
            match tile.owner {
                Owner::TopPlayer => top_occ |= bit,
                Owner::BottomPlayer => bot_occ |= bit,
            }
        }
        GameBoard { board, duke_cache: [top, bottom], occ, top_occ, bot_occ }
    }

    pub fn empty() -> GameBoard {
        GameBoard {
            board: Board::square(GameBoard::BOARD_SIZE),
            duke_cache: [None, None],
            occ: 0, top_occ: 0, bot_occ: 0,
        }
    }

    #[inline(always)]
    fn owner_index(o: Owner) -> usize {
        match o {
            Owner::TopPlayer => 0,
            Owner::BottomPlayer => 1,
        }
    }

    #[inline(always)]
    fn coord_bit(c: Coordinates) -> u64 {
        1u64 << (c.y as u32 * 6 + c.x as u32)
    }

    #[inline(always)]
    fn coord_idx(c: Coordinates) -> usize {
        c.y as usize * 6 + c.x as usize
    }

    #[inline(always)]
    fn owner_occ(&self, o: Owner) -> u64 {
        match o {
            Owner::TopPlayer => self.top_occ,
            Owner::BottomPlayer => self.bot_occ,
        }
    }
    fn absolute_duke_offset(&self, offset: DukeOffset, c: Coordinates) -> Option<Coordinates> {
        fn or_none<P>(b: bool, c: P) -> Option<Coordinates> where P: Fn() -> Coordinates {
            if b { Some(c()) } else { None }
        }
        match offset {
            DukeOffset::Top => or_none(c.y > 0, || Coordinates { x: c.x, y: c.y - 1 }),
            DukeOffset::Bottom => or_none(c.y < self.height() - 1, || Coordinates { x: c.x, y: c.y + 1 }),
            DukeOffset::Left => or_none(c.x > 0, || Coordinates { x: c.x - 1, y: c.y }),
            DukeOffset::Right => or_none(c.x < self.width() - 1, || Coordinates { x: c.x + 1, y: c.y }),
        }
    }

    pub fn to_absolute_duke_offset(&self, offset: DukeOffset, o: Owner) -> Option<Coordinates> {
        self.absolute_duke_offset(offset, self.duke_coordinates(o))
    }

    pub fn get_board(&self) -> &Board<PlacedTile> {
        &self.board
    }

    #[inline(always)]
    pub fn owner_pieces_bitboard(&self, owner: Owner) -> u64 {
        self.owner_occ(owner)
    }

    #[inline(always)]
    pub fn piece_count(&self, owner: Owner) -> u32 {
        self.owner_occ(owner).count_ones()
    }

    #[inline]
    pub fn place(&mut self, c: Coordinates, t: PlacedTile) -> () {
        assert!(self.board.is_empty(c), "Cannot insert tile into occupied space {:?}", c);
        if t.tile_type.is_duke() {
            self.duke_cache[Self::owner_index(t.owner)] = Some(c);
        }
        let bit = Self::coord_bit(c);
        self.occ |= bit;
        match t.owner {
            Owner::TopPlayer => self.top_occ |= bit,
            Owner::BottomPlayer => self.bot_occ |= bit,
        }
        self.board.put(c, t);
    }
    #[inline]
    fn remove(&mut self, c: Coordinates) -> PlacedTile {
        let tile = self.board.remove(c).unwrap_or_else(|| panic!("Cannot remove tile from empty space {:?}", c));
        if tile.tile_type.is_duke() {
            self.duke_cache[Self::owner_index(tile.owner)] = None;
        }
        let bit = Self::coord_bit(c);
        self.occ &= !bit;
        match tile.owner {
            Owner::TopPlayer => self.top_occ &= !bit,
            Owner::BottomPlayer => self.bot_occ &= !bit,
        }
        tile
    }

    #[inline(always)]
    pub fn get(&self, c: Coordinates) -> Option<&PlacedTile> {
        self.board.get(c)
    }

    fn to_absolute_coordinate(
        &self, src: Coordinates, offset: Offsets, center: VerticalOffset,
    ) -> Option<Coordinates> {
        fn vertical_offset(y: VerticalOffset) -> i32 {
            match y {
                VerticalOffset::FarTop => -2,
                VerticalOffset::Top => -1,
                VerticalOffset::Center => 0,
                VerticalOffset::Bottom => 1,
                VerticalOffset::FarBottom => 2,
            }
        }
        let x: i32 = src.x as i32 + match offset.x {
            HorizontalOffset::FarLeft => -2,
            HorizontalOffset::Left => -1,
            HorizontalOffset::Center => 0,
            HorizontalOffset::Right => 1,
            HorizontalOffset::FarRight => 2,
        };
        let y: i32 = src.y as i32 + vertical_offset(offset.y) + vertical_offset(center);
        u8::try_from(x)
            .and_then(|x| u8::try_from(y).map(|y| Coordinates { x, y }))
            .ok()
            .filter(|c| self.board.is_in_bounds(*c))
    }

    fn target_coordinates(
        &self, src: Coordinates, offset: Offsets, action: TileAction, center: VerticalOffset,
    ) -> TargetCoords {
        match action {
            TileAction::Move | TileAction::Jump | TileAction::Strike =>
                TargetCoords::from_option(self.to_absolute_coordinate(src, offset, center)),
            TileAction::Slide => {
                let mut res = TargetCoords::empty();
                let push_horizontal = |res: &mut TargetCoords, r: Range<u8>| {
                    for x in r { res.push(Coordinates { x, y: src.y }); }
                };
                let push_vertical = |res: &mut TargetCoords, r: Range<u8>| {
                    for y in r { res.push(Coordinates { x: src.x, y }); }
                };
                fn push_diagonal<I1, I2>(res: &mut TargetCoords, x: I1, y: I2)
                    where I1: Iterator<Item=u8>, I2: Iterator<Item=u8> {
                    for (x, y) in x.zip(y) { res.push(Coordinates { x, y }); }
                }
                if offset == HorizontalOffset::Right.center() {
                    push_horizontal(&mut res, 0..src.x);
                } else if offset == HorizontalOffset::Left.center() {
                    push_horizontal(&mut res, src.x + 1..self.width());
                } else if offset == VerticalOffset::Top.center() {
                    push_vertical(&mut res, 0..src.y);
                } else if offset == VerticalOffset::Bottom.center() {
                    push_vertical(&mut res, src.y + 1..self.height());
                    // Diagonals
                } else if offset == Offsets::new(HorizontalOffset::Right, VerticalOffset::Top) {
                    push_diagonal(&mut res, (0..src.x).rev(), (0..src.y).rev());
                } else if offset == Offsets::new(HorizontalOffset::Left, VerticalOffset::Top) {
                    push_diagonal(&mut res, src.x + 1..self.width(), (0..src.y).rev());
                } else if offset == Offsets::new(HorizontalOffset::Right, VerticalOffset::Bottom) {
                    push_diagonal(&mut res, (0..src.x).rev(), src.y + 1..self.height());
                } else if offset == Offsets::new(HorizontalOffset::Left, VerticalOffset::Bottom) {
                    push_diagonal(&mut res, src.x + 1..self.width(), src.y + 1..self.height());
                } else {
                    panic!("Invalid slide offset {:?}", offset)
                };
                if cfg!(debug_assertions) {
                    res.iter().for_each(|e| assert!(self.board.is_in_bounds(e)));
                    res.iter().for_each(|e| assert!(e.is_straight_line_to(src)));
                }
                res
            }
            TileAction::JumpSlide => {
                // JumpSlide acts like Slide but can jump over one adjacent tile.
                // Map far offset to near offset to determine direction, then reuse Slide logic.
                // JumpSlide offsets should use "far" offsets to define the jump direction.
                debug_assert!(
                    matches!(offset.x, HorizontalOffset::FarLeft | HorizontalOffset::FarRight | HorizontalOffset::Center)
                    && matches!(offset.y, VerticalOffset::FarTop | VerticalOffset::FarBottom | VerticalOffset::Center),
                    "JumpSlide offset should use far offsets, got ({:?}, {:?})", offset.x, offset.y
                );
                let near_offset = Offsets::new(offset.x.to_near(), offset.y.to_near());
                self.target_coordinates(src, near_offset, TileAction::Slide, center)
            }
            TileAction::Unit => panic!("ASSERTION ERROR"),
            TileAction::Command => panic!("ASSERTION ERROR"),
        }
    }

    #[inline]
    fn unobstructed(&self, src: Coordinates, dst: Coordinates) -> bool {
        RAY_BETWEEN[Self::coord_idx(src)][Self::coord_idx(dst)] & self.occ == 0
    }

    pub fn can_place_new_tile_near_duke(&self, o: Owner) -> bool {
        let duke_location = self.duke_coordinates(o);
        DukeOffset::iter()
            .filter_map(|offset| self.absolute_duke_offset(offset, duke_location))
            .any(|c| self.board.is_empty(c))
    }

    /// Returns empty spaces adjacent to the duke. At most 4 results (cardinal directions),
    /// stack-allocated to avoid heap allocation.
    pub fn empty_spaces_near_current_duke(&self, o: Owner) -> DukeNeighbors {
        let duke_location = self.duke_coordinates(o);
        let mut result = DukeNeighbors { coords: [Coordinates { x: 0, y: 0 }; 4], len: 0 };
        for offset in DukeOffset::iter() {
            if let Some(c) = self.absolute_duke_offset(offset, duke_location) {
                if self.board.is_empty(c) {
                    result.coords[result.len as usize] = c;
                    result.len += 1;
                }
            }
        }
        result
    }

    #[inline(always)]
    fn different_team_or_empty(&self, src: Coordinates, dst: Coordinates) -> bool {
        let src_owner = self.board.get(src).expect("No unit found in src").owner;
        Self::coord_bit(dst) & self.owner_occ(src_owner) == 0
    }

    #[inline]
    fn can_apply_action(&self, src: Coordinates, dst: Coordinates, action: TileAction) -> bool {
        if !self.different_team_or_empty(src, dst) {
            return false;
        }
        match action {
            TileAction::Unit => panic!("Cannot apply action Unit"),
            TileAction::Move =>
                src.is_straight_line_to(dst) && self.unobstructed(src, dst),
            TileAction::Jump => true,
            TileAction::Slide =>
                src.is_straight_line_to(dst) && self.unobstructed(src, dst),
            TileAction::Command => panic!("Commands shouldn't have been used here"),
            TileAction::JumpSlide => {
                if !src.is_straight_line_to(dst) {
                    return false;
                }
                self.unobstructed_jump_slide(src, dst)
            }
            TileAction::Strike => self.get(dst).exists(|o| o.different_team(&self.get(src).unwrap())),
        }
    }

    /// Like `can_apply_action` but uses bitboards for friendly/obstruction checks.
    #[inline]
    fn can_apply_action_fast(&self, src: Coordinates, dst: Coordinates, action: TileAction) -> bool {
        if Self::coord_bit(dst) & self.owner_occ(self.board.get(src).unwrap().owner) != 0 {
            return false;
        }
        match action {
            TileAction::Move | TileAction::Slide =>
                src.is_straight_line_to(dst) && self.unobstructed(src, dst),
            TileAction::Jump => true,
            TileAction::JumpSlide =>
                src.is_straight_line_to(dst) && self.unobstructed_jump_slide(src, dst),
            TileAction::Strike => Self::coord_bit(dst) & self.occ != 0,
            _ => false,
        }
    }

    /// JumpSlide obstruction: skip the first intermediate square (the one being jumped over).
    #[inline]
    fn unobstructed_jump_slide(&self, src: Coordinates, dst: Coordinates) -> bool {
        let si = Self::coord_idx(src);
        let di = Self::coord_idx(dst);
        let ray = RAY_BETWEEN[si][di];
        if ray == 0 { return true; }
        // The first intermediate square is the one adjacent to src in the direction of dst.
        let dx = (di % 6) as i32 - (si % 6) as i32;
        let dy = (di / 6) as i32 - (si / 6) as i32;
        let sx = if dx > 0 { 1 } else if dx < 0 { -1i32 } else { 0 };
        let sy = if dy > 0 { 1 } else if dy < 0 { -1i32 } else { 0 };
        let adj_x = (si % 6) as i32 + sx;
        let adj_y = (si / 6) as i32 + sy;
        let adj_bit = 1u64 << (adj_y * 6 + adj_x);
        (ray & !adj_bit) & self.occ == 0
    }

    fn can_apply(
        &self,
        src: Coordinates,
        dst: Coordinates,
    ) -> AppliedPubAction {
        self.get(src)
            .expect("src position is empty")
            .get_action_from_coordinates(src, dst)
            .map_or(
                AppliedPubAction::Invalid,
                |a| {
                    match (a, self.can_apply_action(src, dst, a)) {
                        (_, false) => AppliedPubAction::Invalid,
                        (TileAction::Strike, true) => AppliedPubAction::Strike,
                        (_, true) => AppliedPubAction::Movement,
                    }
                },
            )
    }

    // fn can_command(
//     &self,
//     commander_src: Coordinates,
//     unit_src: Coordinates,
//     unit_dst: Coordinates,
// ) -> bool {
//     let commander_tile = self.get(commander_src).expect("No unit found in commander_src");
//     let unit_tile = self.get(unit_src).expect("No commanded unit found in unit_src");
//     assert!(commander_tile.same_team(unit_tile), "Cannot command a unit from a different team");
//     self.different_team_or_empty(unit_src, unit_dst)
// }
//
    pub fn duke_coordinates(&self, o: Owner) -> Coordinates {
        // O(1) lookup via cached duke position, instead of scanning all 36 cells.
        self.duke_cache[Self::owner_index(o)]
            .unwrap_or_else(|| panic!("Could not find the duke for {:?}", o))
    }

    #[inline(always)]
    fn flip(&mut self, c: Coordinates) -> () {
        self.board.get_mut(c).unwrap().flip()
    }

    pub fn can_move(&self, src: Coordinates, dst: Coordinates) -> bool {
        self.can_apply(src, dst) != AppliedPubAction::Invalid
    }

    /// Check if the offset is a valid placement (ignoring guard).
    fn is_valid_placement_space(&self, owner: Owner, offset: DukeOffset) -> Option<Coordinates> {
        match self.absolute_duke_offset(offset, self.duke_coordinates(owner)) {
            None => None,
            Some(c) => if Self::coord_bit(c) & self.occ != 0 { None } else { Some(c) }
        }
    }

    /// Apply-check-undo guard check for a placement. Takes `&mut self`.
    fn placement_does_not_put_in_guard(&mut self, c: Coordinates, owner: Owner) -> bool {
        self.place(c, PlacedTile::new(owner, TileType::Footman));
        let in_guard = self.is_guard(owner);
        self.remove(c);
        !in_guard
    }

    /// Apply-check-undo guard check for a tile action, given the already-validated action type.
    /// Avoids re-validating the move through `can_apply`.
    fn tile_action_does_not_put_in_guard(
        &mut self, src: Coordinates, dst: Coordinates, action: TileAction, owner: Owner,
    ) -> bool {
        let saved_duke = self.duke_cache;
        let saved_occ = self.occ;
        let saved_top = self.top_occ;
        let saved_bot = self.bot_occ;

        let is_strike = action == TileAction::Strike;
        if is_strike {
            if let Some(target) = self.board.get(dst) {
                if target.tile_type.is_duke() {
                    self.duke_cache[Self::owner_index(target.owner)] = None;
                }
            }
            self.flip(src);
            if let Some(target) = self.board.get(dst) {
                let bit = Self::coord_bit(dst);
                self.occ &= !bit;
                match target.owner {
                    Owner::TopPlayer => self.top_occ &= !bit,
                    Owner::BottomPlayer => self.bot_occ &= !bit,
                }
            }
            let captured = self.board.remove(dst);
            let in_guard = self.is_guard(owner);
            self.flip(src);
            if let Some(cap) = captured {
                self.board.put(dst, cap);
            }
            self.duke_cache = saved_duke;
            self.occ = saved_occ;
            self.top_occ = saved_top;
            self.bot_occ = saved_bot;
            !in_guard
        } else {
            if let Some(tile) = self.board.get(src) {
                if tile.tile_type.is_duke() {
                    self.duke_cache[Self::owner_index(tile.owner)] = Some(dst);
                }
            }
            if let Some(captured) = self.board.get(dst) {
                if captured.tile_type.is_duke() {
                    self.duke_cache[Self::owner_index(captured.owner)] = None;
                }
            }
            // Update bitboards for the move: clear src, handle dst capture, set dst.
            let src_bit = Self::coord_bit(src);
            let dst_bit = Self::coord_bit(dst);
            let src_owner = self.board.get(src).unwrap().owner;
            self.occ = (self.occ & !src_bit) | dst_bit;
            match src_owner {
                Owner::TopPlayer => self.top_occ = (self.top_occ & !src_bit) | dst_bit,
                Owner::BottomPlayer => self.bot_occ = (self.bot_occ & !src_bit) | dst_bit,
            }
            if let Some(captured) = self.board.get(dst) {
                match captured.owner {
                    Owner::TopPlayer => self.top_occ &= !dst_bit,
                    Owner::BottomPlayer => self.bot_occ &= !dst_bit,
                }
            }
            self.flip(src);
            let captured = self.board.mv(src, dst);
            let in_guard = self.is_guard(owner);
            // Undo via board directly, restore bitboards from saved state.
            let mut mover = self.board.remove(dst).unwrap();
            mover.flip();
            self.board.put(src, mover);
            if let Some(cap) = captured {
                self.board.put(dst, cap);
            }
            self.duke_cache = saved_duke;
            self.occ = saved_occ;
            self.top_occ = saved_top;
            self.bot_occ = saved_bot;
            !in_guard
        }
    }

    /// Apply-check-undo guard check for a move. Takes `&mut self`.
    fn move_does_not_put_in_guard(&mut self, mv: BoardMove, owner: Owner) -> bool {
        match mv {
            BoardMove::PlaceNewTile(_tile_type, duke_offset, mv_owner) => {
                let c = self.absolute_duke_offset(duke_offset, self.duke_coordinates(mv_owner))
                    .expect("Invalid duke offset");
                self.place(c, PlacedTile::new(mv_owner, _tile_type));
                let in_guard = self.is_guard(owner);
                self.remove(c);
                !in_guard
            }
            BoardMove::ApplyNonCommandTileAction { src, dst } => {
                let action = self.can_apply(src, dst);
                let saved_duke = self.duke_cache;
                let saved_occ = self.occ;
                let saved_top = self.top_occ;
                let saved_bot = self.bot_occ;
                let result = match action {
                    AppliedPubAction::Movement => {
                        if let Some(tile) = self.board.get(src) {
                            if tile.tile_type.is_duke() {
                                self.duke_cache[Self::owner_index(tile.owner)] = Some(dst);
                            }
                        }
                        if let Some(captured) = self.board.get(dst) {
                            if captured.tile_type.is_duke() {
                                self.duke_cache[Self::owner_index(captured.owner)] = None;
                            }
                        }
                        let src_bit = Self::coord_bit(src);
                        let dst_bit = Self::coord_bit(dst);
                        let src_owner = self.board.get(src).unwrap().owner;
                        self.occ = (self.occ & !src_bit) | dst_bit;
                        match src_owner {
                            Owner::TopPlayer => self.top_occ = (self.top_occ & !src_bit) | dst_bit,
                            Owner::BottomPlayer => self.bot_occ = (self.bot_occ & !src_bit) | dst_bit,
                        }
                        if let Some(captured) = self.board.get(dst) {
                            match captured.owner {
                                Owner::TopPlayer => self.top_occ &= !dst_bit,
                                Owner::BottomPlayer => self.bot_occ &= !dst_bit,
                            }
                        }
                        self.flip(src);
                        let captured = self.board.mv(src, dst);
                        let in_guard = self.is_guard(owner);
                        let mut mover = self.board.remove(dst).unwrap();
                        mover.flip();
                        self.board.put(src, mover);
                        if let Some(cap) = captured {
                            self.board.put(dst, cap);
                        }
                        !in_guard
                    }
                    AppliedPubAction::Strike => {
                        if let Some(target) = self.board.get(dst) {
                            if target.tile_type.is_duke() {
                                self.duke_cache[Self::owner_index(target.owner)] = None;
                            }
                            let bit = Self::coord_bit(dst);
                            self.occ &= !bit;
                            match target.owner {
                                Owner::TopPlayer => self.top_occ &= !bit,
                                Owner::BottomPlayer => self.bot_occ &= !bit,
                            }
                        }
                        self.flip(src);
                        let captured = self.board.remove(dst);
                        let in_guard = self.is_guard(owner);
                        self.flip(src);
                        if let Some(cap) = captured {
                            self.board.put(dst, cap);
                        }
                        !in_guard
                    }
                    AppliedPubAction::Invalid =>
                        panic!("Cannot move unit in {:?} to {:?} (invalid action)", &src, &dst)
                };
                self.duke_cache = saved_duke;
                self.occ = saved_occ;
                self.top_occ = saved_top;
                self.bot_occ = saved_bot;
                result
            }
        }
    }

    pub(super) fn make_a_move(&mut self, gm: BoardMove) -> Option<PlacedTile> {
        let result = match gm {
            BoardMove::PlaceNewTile(tile_type, duke_offset, owner) => {
                let c = self.absolute_duke_offset(duke_offset, self.duke_coordinates(owner))
                    .expect("Request duke location is out of bounds");
                debug_assert!(self.is_valid_placement_space(owner, duke_offset).is_some());
                self.place(c, PlacedTile::new(owner, tile_type));
                None
            }
            BoardMove::ApplyNonCommandTileAction { src, dst } => {
                match self.can_apply(src, dst) {
                    AppliedPubAction::Movement => {
                        let src_bit = Self::coord_bit(src);
                        let dst_bit = Self::coord_bit(dst);
                        let src_owner = self.board.get(src).unwrap().owner;
                        // Update duke cache if a duke is being moved.
                        if self.board.get(src).unwrap().tile_type.is_duke() {
                            self.duke_cache[Self::owner_index(src_owner)] = Some(dst);
                        }
                        // If capturing a duke at dst, clear its cache.
                        if let Some(captured) = self.board.get(dst) {
                            if captured.tile_type.is_duke() {
                                self.duke_cache[Self::owner_index(captured.owner)] = None;
                            }
                            // Clear captured piece from owner's bitboard.
                            match captured.owner {
                                Owner::TopPlayer => self.top_occ &= !dst_bit,
                                Owner::BottomPlayer => self.bot_occ &= !dst_bit,
                            }
                        }
                        // Move mover: clear src, set dst.
                        self.occ = (self.occ & !src_bit) | dst_bit;
                        match src_owner {
                            Owner::TopPlayer => self.top_occ = (self.top_occ & !src_bit) | dst_bit,
                            Owner::BottomPlayer => self.bot_occ = (self.bot_occ & !src_bit) | dst_bit,
                        }
                        self.flip(src);
                        self.board.mv(src, dst)
                    }
                    AppliedPubAction::Strike => {
                        if let Some(target) = self.board.get(dst) {
                            if target.tile_type.is_duke() {
                                self.duke_cache[Self::owner_index(target.owner)] = None;
                            }
                            let bit = Self::coord_bit(dst);
                            self.occ &= !bit;
                            match target.owner {
                                Owner::TopPlayer => self.top_occ &= !bit,
                                Owner::BottomPlayer => self.bot_occ &= !bit,
                            }
                        }
                        self.flip(src);
                        self.board.remove(dst)
                    }
                    AppliedPubAction::Invalid =>
                        panic!("Cannot move unit in {:?} to {:?} (invalid action)", &src, &dst)
                }
            }

            // BoardMove::CommandAnotherTile { commander_src, unit_src, unit_dst } => {
            //     let commander = self.board.get(commander_src).expect("Cannot command from an empty tile");
            //     assert_eq!(
            //         commander.owner,
            //         o,
            //         "Cannot command using unowned command in {:?}",
            //         commander_src
            //     );
            //     assert!(
            //         self.can_command(commander_src, unit_src, unit_dst),
            //         "Can't apply command (commander: {:?}, unit_src: {:?}, unit_dst: {:?}",
            //         commander_src, unit_src, unit_dst,
            //     );
            //     self.flip(commander_src);
            //     self.board.mv(unit_src, unit_dst);
            // }
        };
        // Verify duke cache is consistent after make_a_move.
        debug_assert_eq!(
            self.duke_cache[Self::owner_index(Owner::TopPlayer)],
            self.board.find(|t| t.owner == Owner::TopPlayer && t.tile_type.is_duke()),
            "Duke cache inconsistent for TopPlayer after make_a_move",
        );
        debug_assert_eq!(
            self.duke_cache[Self::owner_index(Owner::BottomPlayer)],
            self.board.find(|t| t.owner == Owner::BottomPlayer && t.tile_type.is_duke()),
            "Duke cache inconsistent for BottomPlayer after make_a_move",
        );
        result
    }

    pub fn get_tiles_for(&self, o: Owner) -> impl Iterator<Item = (Coordinates, &PlacedTile)> {
        self.board
            .active_coordinates()
            .into_iter()
            .filter(move |e| e.1.owner.same_team(&o))
    }

    pub fn get_legal_moves_ignoring_guard(&self, src: Coordinates) -> Vec<(Coordinates, TileAction)> {
        let buf = self.get_legal_moves_no_guard(src);
        buf.as_slice().to_vec()
    }

    /// Count legal moves without guard checking, without heap allocation.
    #[inline]
    pub fn count_legal_moves_ignoring_guard(&self, src: Coordinates) -> usize {
        self.count_legal_moves_no_guard(src)
    }

    /// Returns candidate moves for the tile at `src` without guard checking.
    /// Uses a stack-allocated buffer to avoid heap allocation.
    #[inline]
    fn get_legal_moves_no_guard(&self, src: Coordinates) -> LegalMoveBuffer {
        let tile = self.get(src).unwrap();
        let tile_side = tile.get_current_side();
        let center_offset = tile_side.center_offset();
        let mut buf = LegalMoveBuffer::new();
        for (offset, action) in tile_side.actions().iter() {
            if *action == TileAction::Command || *action == TileAction::Unit {
                continue;
            }
            let targets = self.target_coordinates(src, *offset, *action, center_offset);
            for c in targets.into_iter() {
                if self.can_apply_action_fast(src, c, *action) {
                    buf.push(c, *action);
                }
            }
        }
        buf
    }

    #[inline]
    fn count_legal_moves_no_guard(&self, src: Coordinates) -> usize {
        let tile = self.get(src).unwrap();
        let table = move_table();
        let tile_idx = match tile.owner {
            Owner::BottomPlayer => tile.tile_type.index(),
            Owner::TopPlayer => TileType::COUNT + tile.tile_type.index(),
        };
        let side_idx = match tile.current_side {
            CurrentSide::Initial => 0,
            CurrentSide::Flipped => 1,
        };
        let pos = Self::coord_idx(src);
        let entry = table.lookup(tile_idx, side_idx, pos);

        let my_occ = self.owner_occ(tile.owner);
        let opp_occ = self.occ & !my_occ;

        // Jump + distance-1 Move: just check not friendly.
        let mut count = (entry.jump_and_near_mask & !my_occ).count_ones() as usize;

        // Strike: must be occupied by enemy.
        count += (entry.strike_mask & opp_occ).count_ones() as usize;

        // Far Move: need obstruction check per target.
        for i in 0..entry.far_move_len as usize {
            let di = entry.far_move[i] as usize;
            if (1u64 << di) & my_occ == 0 && RAY_BETWEEN[pos][di] & self.occ == 0 {
                count += 1;
            }
        }

        // Slide/JumpSlide: need obstruction check per target.
        for i in 0..entry.slide_len as usize {
            let di = entry.slide[i] as usize;
            let dst_bit = 1u64 << di;
            if dst_bit & my_occ != 0 { continue; }
            if entry.slide_action[i] == 0 {
                // Slide: straight-line unobstructed.
                if RAY_BETWEEN[pos][di] & self.occ == 0 {
                    count += 1;
                }
            } else {
                // JumpSlide: skip first intermediate square.
                let dx = (di % 6) as i32 - (pos % 6) as i32;
                let dy = (di / 6) as i32 - (pos / 6) as i32;
                let sx = if dx > 0 { 1 } else if dx < 0 { -1i32 } else { 0 };
                let sy = if dy > 0 { 1 } else if dy < 0 { -1i32 } else { 0 };
                let adj_x = (pos % 6) as i32 + sx;
                let adj_y = (pos / 6) as i32 + sy;
                let adj_bit = 1u64 << (adj_y * 6 + adj_x);
                if (RAY_BETWEEN[pos][di] & !adj_bit) & self.occ == 0 {
                    count += 1;
                }
            }
        }

        count
    }

    /// Like `get_legal_moves_no_guard` but also includes friendly-occupied
    /// destinations (using `can_apply_action_ignoring_friendly`).
    /// Used for computing "defended" features efficiently.
    #[inline]
    fn get_reachable_squares_ignoring_friendly(&self, src: Coordinates) -> LegalMoveBuffer {
        let tile = self.get(src).unwrap();
        let tile_side = tile.get_current_side();
        let center_offset = tile_side.center_offset();
        let mut buf = LegalMoveBuffer::new();
        for (offset, action) in tile_side.actions().iter() {
            if *action == TileAction::Command || *action == TileAction::Unit {
                continue;
            }
            let targets = self.target_coordinates(src, *offset, *action, center_offset);
            for c in targets.into_iter() {
                if self.can_apply_action_ignoring_friendly(src, c, *action) {
                    buf.push(c, *action);
                }
            }
        }
        buf
    }

    /// Iterate all squares reachable by `owner`'s tiles, including
    /// friendly-occupied destinations, calling `f(src, dst)` for each.
    /// No heap allocation.
    #[inline]
    pub fn for_each_reach_ignoring_friendly<F: FnMut(Coordinates, Coordinates)>(&self, owner: Owner, mut f: F) {
        for (src, _) in self.get_tiles_for(owner) {
            let buf = self.get_reachable_squares_ignoring_friendly(src);
            for &(dst, _) in buf.as_slice() {
                f(src, dst);
            }
        }
    }

    #[inline]
    pub fn is_guard(&self, owner: Owner) -> bool {
        time_it_macro!("is_guard", {
            let duke_pos = self.duke_coordinates(owner);
            let enemy_occ = match owner {
                Owner::TopPlayer => self.bot_occ,
                Owner::BottomPlayer => self.top_occ,
            };
            let mut bits = enemy_occ;
            while bits != 0 {
                let idx = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let c = Coordinates { x: (idx % 6) as u8, y: (idx / 6) as u8 };
                if self.can_attack_square(c, duke_pos) {
                    return true;
                }
            }
            false
        })
    }

    /// Check if the tile at `src` can attack/reach `target` in a single action
    /// (ignoring guard constraints). This is equivalent to checking whether
    /// `target` appears in `get_legal_moves_aux(src, CheckForGuard(false))`,
    /// but avoids generating all moves -- we only probe one target square.
    #[inline]
    fn can_attack_square(&self, src: Coordinates, target: Coordinates) -> bool {
        let tile = match self.get(src) {
            Some(t) => t,
            None => return false,
        };
        let tile_side = tile.get_current_side();
        let center_offset = tile_side.center_offset();

        for (offset, action) in tile_side.actions().iter() {
            match *action {
                TileAction::Unit | TileAction::Command => continue,
                TileAction::Move | TileAction::Jump | TileAction::Strike => {
                    if let Some(dst) = self.to_absolute_coordinate(src, *offset, center_offset) {
                        if dst == target && self.can_apply_action_fast(src, dst, *action) {
                            return true;
                        }
                    }
                }
                TileAction::Slide => {
                    if self.is_target_on_slide(src, *offset, target)
                        && self.can_apply_action_fast(src, target, TileAction::Slide)
                    {
                        return true;
                    }
                }
                TileAction::JumpSlide => {
                    let near_offset = Offsets::new(offset.x.to_near(), offset.y.to_near());
                    if self.is_target_on_slide(src, near_offset, target)
                        && self.can_apply_action_fast(src, target, TileAction::JumpSlide)
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Like `can_attack_square`, but ignores whether `target` is occupied by a
    /// friendly piece. Used by the training feature extractor to compute
    /// "defended" counts: a friendly tile is defended if another friendly piece
    /// could reach its square (i.e., could recapture if an enemy took it).
    pub fn can_reach_square_ignoring_friendly(&self, src: Coordinates, target: Coordinates) -> bool {
        let tile = match self.get(src) {
            Some(t) => t,
            None => return false,
        };
        let tile_side = tile.get_current_side();
        let center_offset = tile_side.center_offset();

        for (offset, action) in tile_side.actions().iter() {
            match *action {
                TileAction::Unit | TileAction::Command => continue,
                TileAction::Move | TileAction::Jump | TileAction::Strike => {
                    if let Some(dst) = self.to_absolute_coordinate(src, *offset, center_offset) {
                        if dst == target && self.can_apply_action_ignoring_friendly(src, dst, *action) {
                            return true;
                        }
                    }
                }
                TileAction::Slide => {
                    if self.is_target_on_slide(src, *offset, target)
                        && self.can_apply_action_ignoring_friendly(src, target, TileAction::Slide)
                    {
                        return true;
                    }
                }
                TileAction::JumpSlide => {
                    let near_offset = Offsets::new(offset.x.to_near(), offset.y.to_near());
                    if self.is_target_on_slide(src, near_offset, target)
                        && self.can_apply_action_ignoring_friendly(src, target, TileAction::JumpSlide)
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Like `can_apply_action` but does not reject moves to friendly-occupied
    /// squares. Path obstruction and straight-line checks still apply.
    fn can_apply_action_ignoring_friendly(&self, src: Coordinates, dst: Coordinates, action: TileAction) -> bool {
        match action {
            TileAction::Unit => panic!("Cannot apply action Unit"),
            TileAction::Move =>
                src.is_straight_line_to(dst) && self.unobstructed(src, dst),
            TileAction::Jump => true,
            TileAction::Slide =>
                src.is_straight_line_to(dst) && self.unobstructed(src, dst),
            TileAction::Command => panic!("Commands shouldn't have been used here"),
            TileAction::JumpSlide => {
                if !src.is_straight_line_to(dst) {
                    return false;
                }
                let skip = std::cell::Cell::new(true);
                !src.on_the_linear_path_to(dst, |x, y| {
                    if skip.get() {
                        skip.set(false);
                        false
                    } else {
                        self.board.is_occupied(Coordinates { x, y })
                    }
                })
            }
            // Strike can target any occupied or empty square in range; for
            // "reachability" purposes we treat it as reachable.
            TileAction::Strike => true,
        }
    }

    /// Check if `target` lies on the slide line defined by `src` + direction `offset`.
    /// Does NOT check obstruction -- that is handled by `can_apply_action`.
    fn is_target_on_slide(&self, src: Coordinates, offset: Offsets, target: Coordinates) -> bool {
        if target == src || !self.board.is_in_bounds(target) {
            return false;
        }
        // The offset encodes the direction of the slide relative to center.
        // We need to check that target lies in the correct direction from src.
        let dx = target.x as i32 - src.x as i32;
        let dy = target.y as i32 - src.y as i32;

        // Determine the expected direction from the offset.
        // Slide offsets are always "near" offsets: Left/Right/Top/Bottom or
        // near-diagonal combinations (Left+Top, Right+Bottom, etc.)
        let (expect_dx, expect_dy) = match (offset.x, offset.y) {
            // Straight directions (note: target_coordinates reverses Left/Right)
            (HorizontalOffset::Right, VerticalOffset::Center) => (-1i32, 0i32), // slide left
            (HorizontalOffset::Left, VerticalOffset::Center) => (1, 0),         // slide right
            (HorizontalOffset::Center, VerticalOffset::Top) => (0, -1),         // slide up
            (HorizontalOffset::Center, VerticalOffset::Bottom) => (0, 1),       // slide down
            // Diagonals (same reversal pattern as target_coordinates)
            (HorizontalOffset::Right, VerticalOffset::Top) => (-1, -1),
            (HorizontalOffset::Left, VerticalOffset::Top) => (1, -1),
            (HorizontalOffset::Right, VerticalOffset::Bottom) => (-1, 1),
            (HorizontalOffset::Left, VerticalOffset::Bottom) => (1, 1),
            _ => return false,
        };

        // Check that target is in the correct direction and on the line.
        if expect_dx == 0 {
            // Vertical slide
            dx == 0 && (dy.signum() == expect_dy)
        } else if expect_dy == 0 {
            // Horizontal slide
            dy == 0 && (dx.signum() == expect_dx)
        } else {
            // Diagonal slide: |dx| == |dy| and correct direction
            dx.abs() == dy.abs() && dx.signum() == expect_dx && dy.signum() == expect_dy
        }
    }

    // Returns the tile that was removed, if such a tile exists, e.g., when placing a new tile,
    // undoing the action would remove the new tile from the board.
    pub fn undo(&mut self, mv: PossibleMove) -> Option<PlacedTile> {
        let result = match mv {
            PossibleMove::PlaceNewTile(offset, owner) => {
                let duke_pos = self.duke_coordinates(owner);
                let absolute_coordinate = self
                    .to_absolute_duke_offset(offset, owner)
                    .unwrap_or_else(|| panic!(
                        "Invalid tile placement {:?} relative to duke {:?}",
                        offset,
                        duke_pos,
                    ));
                Some(self.remove(absolute_coordinate))
            }
            PossibleMove::ApplyNonCommandTileAction { src, dst, capturing } => {
                if Self::coord_bit(dst) & self.occ == 0 { // Strike
                    let captured = capturing.expect("No captured but attacker didn't move");
                    self.flip(src);
                    self.place(dst, captured);
                } else {
                    let mut mover = self.remove(dst);
                    mover.flip();
                    self.place(src, mover);
                    if let Some(captured) = capturing {
                        self.place(dst, captured.clone());
                    }
                }
                None
            }
        };
        // Verify duke cache is consistent after undo.
        debug_assert_eq!(
            self.duke_cache[Self::owner_index(Owner::TopPlayer)],
            self.board.find(|t| t.owner == Owner::TopPlayer && t.tile_type.is_duke()),
            "Duke cache inconsistent for TopPlayer after undo",
        );
        debug_assert_eq!(
            self.duke_cache[Self::owner_index(Owner::BottomPlayer)],
            self.board.find(|t| t.owner == Owner::BottomPlayer && t.tile_type.is_duke()),
            "Duke cache inconsistent for BottomPlayer after undo",
        );
        result
    }

    pub fn all_valid_moves_ignoring_guard(&self, owner: Owner, new_tiles: WithNewTiles) -> Vec<PossibleMove> {
        let mut result = Vec::new();
        let mut bits = self.owner_occ(owner);
        while bits != 0 {
            let idx = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let src = Coordinates { x: (idx % 6) as u8, y: (idx / 6) as u8 };
            let buf = self.get_legal_moves_no_guard(src);
            for &(dst, _) in buf.as_slice() {
                result.push(PossibleMove::ApplyNonCommandTileAction {
                    src,
                    dst,
                    capturing: self.board.get(dst).cloned(),
                });
            }
        }

        if let WithNewTiles(true) = new_tiles {
            for offset in DukeOffset::iter() {
                if self.is_valid_placement_space(owner, offset).is_some() {
                    result.push(PossibleMove::PlaceNewTile(offset, owner));
                }
            }
        }
        result
    }

    /// Count all valid moves (tile moves + placements) for `owner` without
    /// guard checking. Like `all_valid_moves_ignoring_guard(..).len()` but
    /// without heap allocation.
    #[inline]
    pub fn count_all_valid_moves_ignoring_guard(&self, owner: Owner, new_tiles: WithNewTiles) -> usize {
        self.count_moves_with_duke_ignoring_guard(owner, new_tiles).0
    }

    /// Count all valid moves AND duke-specific moves in a single pass.
    /// Returns `(total_moves, duke_moves)`.
    #[inline]
    pub fn count_moves_with_duke_ignoring_guard(&self, owner: Owner, new_tiles: WithNewTiles) -> (usize, usize) {
        let duke_pos = self.duke_coordinates(owner);
        let mut total = 0usize;
        let mut duke_moves = 0usize;
        for (src, _) in self.get_tiles_for(owner) {
            let n = self.count_legal_moves_no_guard(src);
            total += n;
            if src == duke_pos {
                duke_moves = n;
            }
        }
        if let WithNewTiles(true) = new_tiles {
            for offset in DukeOffset::iter() {
                if self.is_valid_placement_space(owner, offset).is_some() {
                    total += 1;
                }
            }
        }
        (total, duke_moves)
    }

    /// Compute heuristic data for BOTH players in a single pass over the board.
    /// Returns `(own_total_moves, own_duke_moves, own_tiles, opp_total_moves, opp_duke_moves, opp_tiles)`.
    #[inline]
    pub fn heuristic_counts_both_players(
        &self, owner: Owner,
        own_has_bag: bool, opp_has_bag: bool,
    ) -> (usize, usize, usize, usize, usize, usize) {
        let own_duke = self.duke_coordinates(owner);
        let other = match owner {
            Owner::TopPlayer => Owner::BottomPlayer,
            Owner::BottomPlayer => Owner::TopPlayer,
        };
        let opp_duke = self.duke_coordinates(other);
        let own_occ = self.owner_occ(owner);
        let opp_occ = self.owner_occ(other);

        let mut own_total = 0usize;
        let mut own_duke_moves = 0usize;
        let own_tiles = own_occ.count_ones() as usize;
        let mut opp_total = 0usize;
        let mut opp_duke_moves = 0usize;
        let opp_tiles = opp_occ.count_ones() as usize;

        // Iterate own pieces via bitboard.
        let mut bits = own_occ;
        while bits != 0 {
            let idx = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let src = Coordinates { x: (idx % 6) as u8, y: (idx / 6) as u8 };
            let n = self.count_legal_moves_no_guard(src);
            own_total += n;
            if src == own_duke { own_duke_moves = n; }
        }
        // Iterate opponent pieces via bitboard.
        let mut bits = opp_occ;
        while bits != 0 {
            let idx = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let src = Coordinates { x: (idx % 6) as u8, y: (idx / 6) as u8 };
            let n = self.count_legal_moves_no_guard(src);
            opp_total += n;
            if src == opp_duke { opp_duke_moves = n; }
        }

        if own_has_bag {
            for offset in DukeOffset::iter() {
                if self.is_valid_placement_space(owner, offset).is_some() {
                    own_total += 1;
                }
            }
        }
        if opp_has_bag {
            for offset in DukeOffset::iter() {
                if self.is_valid_placement_space(other, offset).is_some() {
                    opp_total += 1;
                }
            }
        }

        (own_total, own_duke_moves, own_tiles, opp_total, opp_duke_moves, opp_tiles)
    }

    /// Iterate over all tile-movement moves (no placements) for `owner` without
    /// guard checking, calling `f(src, dst)` for each. Avoids allocating a Vec.
    #[inline]
    pub fn for_each_tile_move_ignoring_guard<F: FnMut(Coordinates, Coordinates)>(&self, owner: Owner, mut f: F) {
        for (src, _) in self.get_tiles_for(owner) {
            let buf = self.get_legal_moves_no_guard(src);
            for &(dst, _) in buf.as_slice() {
                f(src, dst);
            }
        }
    }

    #[allow(dead_code)]
    pub fn as_single_string(&self) -> String { single_char_print_board(&self.board) }
    #[allow(dead_code)]
    pub fn as_double_string(&self) -> String { double_char_print_board(&self.board) }
    #[allow(dead_code)]
    pub fn debug_double(&self) { println!("{}", self.as_double_string()); }
}

impl Rectangular for GameBoard {
    fn width(&self) -> u8 { self.board.width() }
    fn height(&self) -> u8 { self.board.height() }
}

// Guard-checked methods (apply + check + undo). These require `&mut self`.
impl GameBoard {
    /// Check if a move does not put the owner in guard.
    pub(super) fn does_not_put_in_guard(&mut self, mv: BoardMove, owner: Owner) -> bool {
        #[cfg(debug_assertions)]
        let snapshot = self.clone();
        let result = self.move_does_not_put_in_guard(mv, owner);
        #[cfg(debug_assertions)]
        debug_assert_eq!(self, &snapshot, "does_not_put_in_guard: board not restored after apply/undo");
        result
    }

    pub fn is_valid_placement(&mut self, owner: Owner, offset: DukeOffset) -> bool {
        match self.is_valid_placement_space(owner, offset) {
            None => false,
            Some(c) => {
                #[cfg(debug_assertions)]
                let snapshot = self.clone();
                let result = self.placement_does_not_put_in_guard(c, owner);
                #[cfg(debug_assertions)]
                debug_assert_eq!(self, &snapshot, "is_valid_placement: board not restored after apply/undo");
                result
            }
        }
    }

    // Except commands.
    pub fn get_legal_moves(&mut self, src: Coordinates) -> Vec<(Coordinates, TileAction)> {
        #[cfg(debug_assertions)]
        let snapshot = self.clone();
        let owner = self.get(src).unwrap().owner;
        let candidates = self.get_legal_moves_no_guard(src);
        let mut result = Vec::new();
        for &(dst, action) in candidates.as_slice() {
            if self.tile_action_does_not_put_in_guard(src, dst, action, owner) {
                result.push((dst, action));
            }
        }
        let result = result;
        #[cfg(debug_assertions)]
        debug_assert_eq!(self, &snapshot, "get_legal_moves: board not restored after apply/undo");
        result
    }

    pub fn all_valid_moves(&mut self, owner: Owner, new_tiles: WithNewTiles) -> Vec<PossibleMove> {
        #[cfg(debug_assertions)]
        let snapshot = self.clone();
        // Collect tile coordinates into stack buffer to avoid holding references
        // into the board while mutating it during guard checks.
        let mut tile_coords = [Coordinates { x: 0, y: 0 }; MAX_TILES_PER_PLAYER];
        let mut n_tiles = 0usize;
        for (c, _) in self.get_tiles_for(owner) {
            tile_coords[n_tiles] = c;
            n_tiles += 1;
        }

        let mut result = Vec::new();

        // Tile action moves: collect candidates per tile and filter inline.
        for &src in &tile_coords[..n_tiles] {
            let candidates = self.get_legal_moves_no_guard(src);
            for &(dst, action) in candidates.as_slice() {
                if self.tile_action_does_not_put_in_guard(src, dst, action, owner) {
                    result.push(PossibleMove::ApplyNonCommandTileAction {
                        src,
                        dst,
                        capturing: self.board.get(dst).cloned(),
                    });
                }
            }
        }

        // Placement moves.
        if let WithNewTiles(true) = new_tiles {
            for offset in DukeOffset::iter() {
                if let Some(c) = self.is_valid_placement_space(owner, offset) {
                    if self.placement_does_not_put_in_guard(c, owner) {
                        result.push(PossibleMove::PlaceNewTile(offset, owner));
                    }
                }
            }
        }

        #[cfg(debug_assertions)]
        debug_assert_eq!(self, &snapshot, "all_valid_moves: board not restored after apply/undo");
        result
    }

    /// Check if there is at least one valid move (short-circuits on first found).
    pub fn has_valid_moves(&mut self, owner: Owner, new_tiles: WithNewTiles) -> bool {
        #[cfg(debug_assertions)]
        let snapshot = self.clone();
        let mut tile_coords = [Coordinates { x: 0, y: 0 }; MAX_TILES_PER_PLAYER];
        let mut n_tiles = 0usize;
        let mut bits = self.owner_occ(owner);
        while bits != 0 {
            let idx = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            tile_coords[n_tiles] = Coordinates { x: (idx % 6) as u8, y: (idx / 6) as u8 };
            n_tiles += 1;
        }

        for &src in &tile_coords[..n_tiles] {
            let candidates = self.get_legal_moves_no_guard(src);
            for &(dst, action) in candidates.as_slice() {
                if self.tile_action_does_not_put_in_guard(src, dst, action, owner) {
                    #[cfg(debug_assertions)]
                    debug_assert_eq!(self, &snapshot, "has_valid_moves: board not restored after apply/undo");
                    return true;
                }
            }
        }

        if let WithNewTiles(true) = new_tiles {
            for offset in DukeOffset::iter() {
                if let Some(c) = self.is_valid_placement_space(owner, offset) {
                    if self.placement_does_not_put_in_guard(c, owner) {
                        #[cfg(debug_assertions)]
                        debug_assert_eq!(self, &snapshot, "has_valid_moves: board not restored after apply/udo");
                        return true;
                    }
                }
            }
        }

        #[cfg(debug_assertions)]
        debug_assert_eq!(self, &snapshot, "has_valid_moves: board not restored after apply/undo");
        false
    }
}

#[cfg(test)]
mod test {
    use crate::{assert_empty, assert_eq_set, assert_not};
    use crate::game::units;

    use super::*;

    // get_legal_moves
    #[test]
    fn get_legal_moves_moves_only() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, units::place_tile(Owner::TopPlayer, units::duke));

        let c = Coordinates { x: 2, y: 4 };
        board.place(c, units::place_tile(Owner::TopPlayer, units::footman));
        assert_eq_set!(
            vec![
                (Coordinates { x: 3, y: 4 }, TileAction::Move),
                (Coordinates { x: 2, y: 5 }, TileAction::Move),
                (Coordinates { x: 1, y: 4 }, TileAction::Move),
                (Coordinates { x: 2, y: 3 }, TileAction::Move),
            ],
            board.get_legal_moves(c),
        );
    }

    #[test]
    fn get_legal_moves_and_jumps() {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 1, y: 4 };
        board.place(Coordinates { x: 5, y: 5 }, units::place_tile(Owner::TopPlayer, units::duke));
        board.place(c, units::place_tile(Owner::TopPlayer, units::champion));
        assert_eq_set!(
            vec![
                (Coordinates { x: 1, y: 2 }, TileAction::Jump),
                (Coordinates { x: 1, y: 3 }, TileAction::Move),
                (Coordinates { x: 1, y: 5 }, TileAction::Move),
                (Coordinates { x: 0, y: 4 }, TileAction::Move),
                (Coordinates { x: 2, y: 4 }, TileAction::Move),
                (Coordinates { x: 3, y: 4 }, TileAction::Jump),
            ],
            board.get_legal_moves(c),
        );
    }

    #[test]
    fn get_legal_moves_horizontal_slides() {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 2, y: 4 };
        board.place(c, units::place_tile(Owner::TopPlayer, units::duke));
        assert_eq_set!(
            vec![
                (Coordinates { x: 0, y: 4 }, TileAction::Slide),
                (Coordinates { x: 1, y: 4 }, TileAction::Slide),
                (Coordinates { x: 3, y: 4 }, TileAction::Slide),
                (Coordinates { x: 4, y: 4 }, TileAction::Slide),
                (Coordinates { x: 5, y: 4 }, TileAction::Slide),
            ],
            board.get_legal_moves(c),
        );
    }

    #[test]
    fn get_legal_moves_vertical_slides() {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 2, y: 4 };
        board.place(c, units::place_tile_flipped(Owner::TopPlayer, units::duke));
        assert_eq_set!(
            vec![
                (Coordinates { x: 2, y: 0 }, TileAction::Slide),
                (Coordinates { x: 2, y: 1 }, TileAction::Slide),
                (Coordinates { x: 2, y: 2 }, TileAction::Slide),
                (Coordinates { x: 2, y: 3 }, TileAction::Slide),
                (Coordinates { x: 2, y: 5 }, TileAction::Slide),
            ],
            board.get_legal_moves(c),
        );
    }

    #[test]
    fn can_move_returns_true_for_diagonal_sliding() {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 2, y: 4 };
        board.place(c, units::place_tile(Owner::TopPlayer, units::priest));
        assert!(board.can_move(c, Coordinates { x: 4, y: 2 }));
    }

    #[test]
    fn can_move_returns_true_for_capture() {
        let mut board = GameBoard::empty();
        let src = Coordinates { x: 2, y: 4 };
        let dst = Coordinates { x: 2, y: 5 };
        board.place(src, units::place_tile(Owner::TopPlayer, units::footman));
        board.place(dst, units::place_tile(Owner::BottomPlayer, units::footman));
        assert!(board.can_move(src, dst));
    }

    #[test]
    fn can_move_returns_false_for_occupied_with_same() {
        let mut board = GameBoard::empty();
        let src = Coordinates { x: 2, y: 4 };
        let dst = Coordinates { x: 2, y: 5 };
        board.place(src, units::place_tile(Owner::TopPlayer, units::footman));
        board.place(dst, units::place_tile(Owner::TopPlayer, units::footman));
        assert_not!(board.can_move(src, dst));
    }

    #[test]
    fn can_move_returns_false_for_movement_out_of_scope() {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 2, y: 2 };
        board.place(c, units::place_tile(Owner::TopPlayer, units::footman));
        assert_not!(board.can_move(c, Coordinates { x: 5, y: 5 }));
    }

    #[test]
    fn can_move_returns_false_for_empty_strike() {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 0, y: 0 };
        let mut tile = units::place_tile(Owner::TopPlayer, units::pikeman);
        tile.flip();
        board.place(c, tile);
        assert_not!(board.can_move(c, Coordinates { x: 1, y: 2 }));
    }

    #[test]
    fn get_legal_moves_diagonal_slides() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, units::place_tile(Owner::TopPlayer, units::duke));

        let c = Coordinates { x: 2, y: 4 };
        board.place(c, units::place_tile(Owner::TopPlayer, units::priest));
        assert_eq_set!(
            vec![
                (Coordinates { x: 1, y: 3 }, TileAction::Slide),
                (Coordinates { x: 0, y: 2 }, TileAction::Slide),

                (Coordinates { x: 3, y: 5 }, TileAction::Slide),

                (Coordinates { x: 1, y: 5 }, TileAction::Slide),

                (Coordinates { x: 3, y: 3 }, TileAction::Slide),
                (Coordinates { x: 4, y: 2 }, TileAction::Slide),
                (Coordinates { x: 5, y: 1 }, TileAction::Slide),
            ],
            board.get_legal_moves(c),
        );
    }

    #[test]
    fn is_guard_takes_strikes_into_account() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, units::place_tile(Owner::TopPlayer, units::duke));
        let coordinates = Coordinates { x: 1, y: 2 };
        board.place(coordinates, units::place_tile_flipped(Owner::BottomPlayer, units::pikeman));
        assert!(board.is_guard(Owner::TopPlayer));
    }

    // make_a_move
    #[test]
    fn make_a_move() {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 2, y: 4 };
        board.place(c, units::place_tile(Owner::TopPlayer, units::footman));
        let c2 = Coordinates { x: 1, y: 4 };
        board.make_a_move(
            BoardMove::ApplyNonCommandTileAction { src: c, dst: c2 },
        );
        assert!(board.get(c2).is_some());
        assert!(board.get(c).is_none());
    }

    #[test]
    fn is_valid_placement_returns_false_if_in_guard() {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 0, y: 0 };
        board.place(c, units::place_tile(Owner::TopPlayer, units::duke));
        let c2 = Coordinates { x: 0, y: 2 };
        board.place(c2, units::place_tile_flipped(Owner::BottomPlayer, units::footman));
        assert_not!(board.is_valid_placement(Owner::TopPlayer, DukeOffset::Right));
    }

    #[test]
    fn get_legal_moves_can_block_guard() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        let mut footman = PlacedTile::new(Owner::TopPlayer, TileType::Footman);
        footman.flip();
        let footman_coordinates = Coordinates { x: 4, y: 5 };
        board.place(footman_coordinates, footman);
        let mut op_duke = PlacedTile::new(Owner::BottomPlayer, TileType::Duke);
        op_duke.flip();
        board.place(Coordinates { x: 5, y: 0 }, op_duke);

        // TopPlayer can still play a footman move
        assert_eq!(
            board.get_legal_moves(footman_coordinates),
            vec![(Coordinates { x: 5, y: 4 }, TileAction::Move)],
        )
    }

    #[test]
    fn get_legal_moves_does_not_allow_placing_duke_in_strike() {
        let mut board = GameBoard::empty();
        let duke_coordinates = Coordinates { x: 0, y: 0 };
        board.place(duke_coordinates, units::place_tile(Owner::TopPlayer, units::duke));
        let pikeman_coordinates = Coordinates { x: 2, y: 2 };
        board.place(pikeman_coordinates, units::place_tile_flipped(Owner::BottomPlayer, units::pikeman));
        board.place(Coordinates { x: 1, y: 0 }, units::place_tile(Owner::BottomPlayer, units::footman));
        assert_empty!(board.get_legal_moves(duke_coordinates));
    }

    #[test]
    fn unobstructed_near_straight() {
        test_unobstructed(
            units::place_tile(Owner::TopPlayer, units::footman),
            vec![
                Coordinates { x: 0, y: 1 },
                Coordinates { x: 2, y: 1 },
                Coordinates { x: 1, y: 0 },
                Coordinates { x: 1, y: 2 },
            ],
        )
    }

    #[test]
    fn unobstructed_near_diagonal() {
        test_unobstructed(
            units::place_tile_flipped(Owner::TopPlayer, units::footman),
            vec![
                Coordinates { x: 0, y: 0 },
                Coordinates { x: 2, y: 2 },
                Coordinates { x: 0, y: 2 },
                Coordinates { x: 2, y: 0 },
            ],
        )
    }

    fn test_unobstructed(placed_tile: PlacedTile, dsts: Vec<Coordinates>) {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 1, y: 1 };
        board.place(c, placed_tile);
        let board = board;
        for dst in dsts {
            let mut b = board.clone();
            b.place(dst, units::place_tile(Owner::BottomPlayer, units::duke));
            assert!(board.can_move(c, dst), "Can't move from {} to {}", c, dst)
        }
    }


    #[test]
    fn sanity_guard1() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, units::place_tile(Owner::TopPlayer, units::duke));
        board.place(
            Coordinates { x: 2, y: 1 },
            units::place_tile(Owner::TopPlayer, units::footman),
        );
        let duke_coordinates = Coordinates { x: 2, y: 2 };
        board.place(duke_coordinates, units::place_tile_flipped(Owner::BottomPlayer, units::duke));
        assert!(board.is_guard(Owner::BottomPlayer))
    }

    #[test]
    fn sanity_guard2() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, units::place_tile(Owner::TopPlayer, units::duke));
        board.place(
            Coordinates { x: 2, y: 1 },
            units::place_tile_flipped(Owner::TopPlayer, units::footman),
        );
        let duke_coordinates = Coordinates { x: 1, y: 2 };
        board.place(duke_coordinates, units::place_tile(Owner::BottomPlayer, units::duke));
        assert!(board.is_guard(Owner::BottomPlayer))
    }

    // ── JumpSlide tests (Assassin) ──────────────────────────────────────
    // TopPlayer tiles are vertically flipped, so the Assassin's initial-side
    // FarTop JumpSlide becomes a downward JumpSlide for TopPlayer.

    #[test]
    fn jumpslide_can_move_to_adjacent_square() {
        // Assassin (TopPlayer, initial side) has JumpSlide downward.
        // With no obstacles it should slide 1 square down (like Slide).
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, units::place_tile(Owner::TopPlayer, units::duke));
        let src = Coordinates { x: 3, y: 2 };
        board.place(src, units::place_tile(Owner::TopPlayer, units::assassin));
        // 1 square down
        let dst = Coordinates { x: 3, y: 3 };
        assert!(board.can_move(src, dst));
    }

    #[test]
    fn jumpslide_can_jump_over_adjacent_piece() {
        // Assassin (TopPlayer, initial side) has JumpSlide downward.
        // Place a blocker directly adjacent (1 square down). JumpSlide should
        // jump over it and reach the square beyond.
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, units::place_tile(Owner::TopPlayer, units::duke));
        let src = Coordinates { x: 3, y: 1 };
        board.place(src, units::place_tile(Owner::TopPlayer, units::assassin));
        // Blocker at (3,2) -- adjacent to src downward
        board.place(Coordinates { x: 3, y: 2 }, units::place_tile(Owner::BottomPlayer, units::footman));
        // Assassin should jump over (3,2) and reach (3,3)
        let dst = Coordinates { x: 3, y: 3 };
        assert!(board.can_move(src, dst));
    }

    #[test]
    fn jumpslide_blocked_by_non_adjacent_piece() {
        // Assassin (TopPlayer, initial side) has JumpSlide downward.
        // Place a blocker 2 squares away (not adjacent). JumpSlide should NOT
        // be able to pass through it to reach a square beyond.
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, units::place_tile(Owner::TopPlayer, units::duke));
        let src = Coordinates { x: 3, y: 0 };
        board.place(src, units::place_tile(Owner::TopPlayer, units::assassin));
        // Blocker at (3,2) -- 2 squares away from src (not adjacent)
        board.place(Coordinates { x: 3, y: 2 }, units::place_tile(Owner::BottomPlayer, units::footman));
        // Assassin should NOT reach (3,3) past the non-adjacent blocker
        let dst = Coordinates { x: 3, y: 3 };
        assert_not!(board.can_move(src, dst));
    }

    // ── GameBoard construction and basic operations ───────────────────
    #[test]
    fn empty_board_has_correct_dimensions() {
        let board = GameBoard::empty();
        assert_eq!(board.width(), GameBoard::BOARD_SIZE);
        assert_eq!(board.height(), GameBoard::BOARD_SIZE);
    }

    #[test]
    fn place_and_get_returns_tile() {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 2, y: 3 };
        let tile = PlacedTile::new(Owner::TopPlayer, TileType::Footman);
        board.place(c, tile);
        let got = board.get(c);
        assert!(got.is_some());
        assert_eq!(got.unwrap().tile_type, TileType::Footman);
        assert_eq!(got.unwrap().owner, Owner::TopPlayer);
    }

    #[test]
    fn get_returns_none_on_empty_square() {
        let board = GameBoard::empty();
        assert!(board.get(Coordinates { x: 0, y: 0 }).is_none());
    }

    #[test]
    #[should_panic]
    fn place_on_occupied_panics() {
        let mut board = GameBoard::empty();
        let c = Coordinates { x: 2, y: 3 };
        board.place(c, PlacedTile::new(Owner::TopPlayer, TileType::Footman));
        board.place(c, PlacedTile::new(Owner::BottomPlayer, TileType::Knight));
    }

    // ── can_attack_square tests ───────────────────────────────────────
    #[test]
    fn can_attack_square_footman_move_target() {
        // Footman (initial side) has Move in 4 cardinal directions.
        let mut board = GameBoard::empty();
        let src = Coordinates { x: 3, y: 3 };
        board.place(src, units::place_tile(Owner::TopPlayer, units::footman));
        // Footman should be able to attack adjacent cardinal squares
        assert!(board.can_attack_square(src, Coordinates { x: 3, y: 2 }));
        assert!(board.can_attack_square(src, Coordinates { x: 3, y: 4 }));
        assert!(board.can_attack_square(src, Coordinates { x: 2, y: 3 }));
        assert!(board.can_attack_square(src, Coordinates { x: 4, y: 3 }));
    }

    #[test]
    fn can_attack_square_footman_cannot_reach_diagonal() {
        // Footman (initial side) has cardinal Moves only -- not diagonals.
        let mut board = GameBoard::empty();
        let src = Coordinates { x: 3, y: 3 };
        board.place(src, units::place_tile(Owner::TopPlayer, units::footman));
        assert_not!(board.can_attack_square(src, Coordinates { x: 4, y: 4 }));
        assert_not!(board.can_attack_square(src, Coordinates { x: 2, y: 2 }));
    }

    #[test]
    fn can_attack_square_duke_slide_reaches_far_square() {
        // Duke (initial side) has horizontal Slides.
        let mut board = GameBoard::empty();
        let src = Coordinates { x: 2, y: 3 };
        board.place(src, units::place_tile(Owner::TopPlayer, units::duke));
        // Should slide to far horizontal squares
        assert!(board.can_attack_square(src, Coordinates { x: 5, y: 3 }));
        assert!(board.can_attack_square(src, Coordinates { x: 0, y: 3 }));
    }

    #[test]
    fn can_attack_square_duke_slide_blocked_by_piece() {
        // Duke (initial side) has horizontal Slides. A piece in between blocks.
        let mut board = GameBoard::empty();
        let src = Coordinates { x: 0, y: 3 };
        board.place(src, units::place_tile(Owner::TopPlayer, units::duke));
        board.place(Coordinates { x: 2, y: 3 }, PlacedTile::new(Owner::BottomPlayer, TileType::Footman));
        // Duke slide should be blocked from reaching x=3
        assert_not!(board.can_attack_square(src, Coordinates { x: 3, y: 3 }));
        // But can still reach x=1 (before the blocker)
        assert!(board.can_attack_square(src, Coordinates { x: 1, y: 3 }));
    }

    #[test]
    fn can_attack_square_returns_false_for_empty_src() {
        let board = GameBoard::empty();
        assert_not!(board.can_attack_square(Coordinates { x: 0, y: 0 }, Coordinates { x: 1, y: 0 }));
    }

    #[test]
    fn can_attack_square_champion_jump_ignores_obstruction() {
        // Champion (initial side) has Jump actions -- not blocked by intermediate pieces.
        let mut board = GameBoard::empty();
        let src = Coordinates { x: 2, y: 3 };
        board.place(src, units::place_tile(Owner::TopPlayer, units::champion));
        // Place a blocker in between
        board.place(Coordinates { x: 2, y: 2 }, PlacedTile::new(Owner::BottomPlayer, TileType::Footman));
        // Champion should jump over the blocker to (2,1)
        assert!(board.can_attack_square(src, Coordinates { x: 2, y: 1 }));
    }

    // ── is_guard tests ────────────────────────────────────────────────
    #[test]
    fn is_guard_returns_true_when_duke_is_attacked() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 3, y: 3 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        // Place an enemy footman that can reach the duke
        board.place(Coordinates { x: 3, y: 4 }, units::place_tile(Owner::BottomPlayer, units::footman));
        assert!(board.is_guard(Owner::TopPlayer));
    }

    #[test]
    fn is_guard_returns_false_when_duke_is_safe() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        // Place an enemy footman far away from the duke
        board.place(Coordinates { x: 5, y: 5 }, units::place_tile(Owner::BottomPlayer, units::footman));
        assert_not!(board.is_guard(Owner::TopPlayer));
    }

    #[test]
    fn is_guard_friendly_piece_does_not_guard() {
        // A friendly piece adjacent to the duke should not trigger guard.
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 3, y: 3 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        board.place(Coordinates { x: 3, y: 4 }, units::place_tile(Owner::TopPlayer, units::footman));
        assert_not!(board.is_guard(Owner::TopPlayer));
    }

    // ── Move validation: move to empty vs friendly vs obstructed ──────
    #[test]
    fn can_move_to_empty_square() {
        let mut board = GameBoard::empty();
        let src = Coordinates { x: 3, y: 3 };
        board.place(src, units::place_tile(Owner::TopPlayer, units::footman));
        assert!(board.can_move(src, Coordinates { x: 3, y: 4 }));
    }

    #[test]
    fn can_move_blocked_by_friendly_piece() {
        let mut board = GameBoard::empty();
        let src = Coordinates { x: 3, y: 3 };
        let dst = Coordinates { x: 3, y: 4 };
        board.place(src, units::place_tile(Owner::TopPlayer, units::footman));
        board.place(dst, units::place_tile(Owner::TopPlayer, units::knight));
        assert_not!(board.can_move(src, dst));
    }

    #[test]
    fn slide_blocked_by_intervening_piece() {
        // Duke (initial side) slides horizontally. A piece in the path blocks it.
        let mut board = GameBoard::empty();
        let src = Coordinates { x: 0, y: 3 };
        board.place(src, units::place_tile(Owner::TopPlayer, units::duke));
        board.place(Coordinates { x: 2, y: 3 }, PlacedTile::new(Owner::BottomPlayer, TileType::Footman));
        // Can slide to (1,3) but NOT past the blocker to (3,3)
        assert!(board.can_move(src, Coordinates { x: 1, y: 3 }));
        assert_not!(board.can_move(src, Coordinates { x: 3, y: 3 }));
    }

    #[test]
    fn jump_over_piece_succeeds() {
        // Champion (initial side) has Jump. Should jump over adjacent occupied.
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, units::place_tile(Owner::TopPlayer, units::duke));
        let src = Coordinates { x: 2, y: 3 };
        board.place(src, units::place_tile(Owner::TopPlayer, units::champion));
        // Place a blocker adjacent
        board.place(Coordinates { x: 2, y: 2 }, PlacedTile::new(Owner::BottomPlayer, TileType::Footman));
        // Champion jumps to (2,1) over the blocker
        assert!(board.can_move(src, Coordinates { x: 2, y: 1 }));
    }

    #[test]
    fn move_obstructed_by_intervening_piece() {
        // A Move (non-jump) is obstructed by pieces in the path.
        // Pikeman (initial side) has a far Move at distance 2 along cardinal.
        // Actually, let's use footman which only moves 1 square, and test that
        // moving 2 squares is invalid.
        let mut board = GameBoard::empty();
        let src = Coordinates { x: 3, y: 3 };
        board.place(src, units::place_tile(Owner::TopPlayer, units::footman));
        // Footman can only move 1 square -- 2 squares should fail.
        assert_not!(board.can_move(src, Coordinates { x: 3, y: 5 }));
    }

    // ── duke_coordinates ──────────────────────────────────────────────
    #[test]
    fn duke_coordinates_finds_correct_position() {
        let mut board = GameBoard::empty();
        let duke_pos = Coordinates { x: 3, y: 4 };
        board.place(duke_pos, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        board.place(Coordinates { x: 1, y: 1 }, PlacedTile::new(Owner::BottomPlayer, TileType::Duke));
        assert_eq!(board.duke_coordinates(Owner::TopPlayer), duke_pos);
        assert_eq!(board.duke_coordinates(Owner::BottomPlayer), Coordinates { x: 1, y: 1 });
    }

    // ── get_tiles_for ─────────────────────────────────────────────────
    #[test]
    fn get_tiles_for_returns_only_owned_tiles() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        board.place(Coordinates { x: 1, y: 1 }, PlacedTile::new(Owner::TopPlayer, TileType::Footman));
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::BottomPlayer, TileType::Duke));
        let top_tiles: Vec<_> = board.get_tiles_for(Owner::TopPlayer).collect();
        assert_eq!(top_tiles.len(), 2);
        assert!(top_tiles.iter().all(|(_, t)| t.owner == Owner::TopPlayer));
        let bot_tiles: Vec<_> = board.get_tiles_for(Owner::BottomPlayer).collect();
        assert_eq!(bot_tiles.len(), 1);
    }

    // ── can_reach_square_ignoring_friendly ────────────────────────────

    #[test]
    fn can_reach_ignoring_friendly_duke_slide_to_friendly_square() {
        // Duke (initial side) slides horizontally. Normally can't move to a
        // friendly-occupied square, but ignoring_friendly should return true.
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        board.place(Coordinates { x: 3, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Footman));
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::BottomPlayer, TileType::Duke));

        // Duke at (0,0) slides right to (3,0) where friendly footman sits
        assert!(board.can_reach_square_ignoring_friendly(
            Coordinates { x: 0, y: 0 }, Coordinates { x: 3, y: 0 }),
            "Duke should be able to reach friendly-occupied square ignoring friendly");
    }

    #[test]
    fn can_reach_ignoring_friendly_blocked_by_intervening_piece() {
        // Duke slide should still be blocked by an intervening piece
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        board.place(Coordinates { x: 2, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Footman));
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::BottomPlayer, TileType::Duke));

        // Duke at (0,0) tries to slide to (3,0) but footman at (2,0) blocks
        assert!(!board.can_reach_square_ignoring_friendly(
            Coordinates { x: 0, y: 0 }, Coordinates { x: 3, y: 0 }),
            "Duke slide should be blocked by intervening piece even when ignoring friendly");
    }

    #[test]
    fn can_reach_ignoring_friendly_footman_move_to_friendly() {
        // Footman (initial side) has Move in 4 cardinal directions.
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 2, y: 2 }, PlacedTile::new(Owner::TopPlayer, TileType::Footman));
        board.place(Coordinates { x: 3, y: 2 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::BottomPlayer, TileType::Duke));

        // Footman at (2,2) moves right to (3,2) where friendly duke sits
        assert!(board.can_reach_square_ignoring_friendly(
            Coordinates { x: 2, y: 2 }, Coordinates { x: 3, y: 2 }),
            "Footman should reach friendly-occupied adjacent square");
    }

    #[test]
    fn can_reach_ignoring_friendly_returns_false_for_empty_src() {
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::BottomPlayer, TileType::Duke));

        assert!(!board.can_reach_square_ignoring_friendly(
            Coordinates { x: 3, y: 3 }, Coordinates { x: 0, y: 0 }),
            "Empty source should return false");
    }

    #[test]
    fn can_reach_ignoring_friendly_returns_false_for_out_of_range() {
        // Footman can only move 1 square, not 3
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Footman));
        board.place(Coordinates { x: 0, y: 5 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::BottomPlayer, TileType::Duke));

        assert!(!board.can_reach_square_ignoring_friendly(
            Coordinates { x: 0, y: 0 }, Coordinates { x: 3, y: 0 }),
            "Footman should not reach a square 3 away");
    }

    #[test]
    fn can_reach_ignoring_friendly_champion_jump_over_friendly() {
        // Champion (initial side) has Jump actions. Jumps ignore obstruction.
        let mut board = GameBoard::empty();
        board.place(Coordinates { x: 2, y: 2 }, PlacedTile::new(Owner::TopPlayer, TileType::Champion));
        board.place(Coordinates { x: 2, y: 1 }, PlacedTile::new(Owner::TopPlayer, TileType::Footman));
        // Champion jumps to (2,0) — friendly at (2,1) doesn't block jump
        board.place(Coordinates { x: 0, y: 0 }, PlacedTile::new(Owner::TopPlayer, TileType::Duke));
        board.place(Coordinates { x: 5, y: 5 }, PlacedTile::new(Owner::BottomPlayer, TileType::Duke));

        assert!(board.can_reach_square_ignoring_friendly(
            Coordinates { x: 2, y: 2 }, Coordinates { x: 2, y: 0 }),
            "Champion jump should reach (2,0) over friendly at (2,1)");
    }
}
