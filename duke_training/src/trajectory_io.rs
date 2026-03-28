//! Save and load game trajectories with full GameState reconstruction.
//!
//! Binary format v2 (little-endian):
//!   Header: b"DTRJ" + version(u32) + num_games(u32)
//!   Per game:
//!     result: u8 (0=TopWin, 1=BottomWin, 2=Tie)
//!     num_states: u16
//!     Per state:
//!       current_player: u8 (0=Top, 1=Bottom)
//!       idle_moves: u8
//!       num_board_tiles: u8
//!       Per board tile:
//!         x: u8, y: u8, tile_type: u8, current_side: u8, owner: u8
//!       top_bag_len: u8, then tile_type bytes
//!       bottom_bag_len: u8, then tile_type bytes
//!       top_discard_len: u8, then tile_type bytes
//!       bottom_discard_len: u8, then tile_type bytes

use std::io::{self, Read, Write, BufWriter};

use duke_rust::common::coordinates::Coordinates;
use duke_rust::game::bag::{DiscardBag, TileBag};
use duke_rust::game::state::{GameResult, GameSnapshot, GameState};
use duke_rust::game::tile::{CurrentSide, Owner, PlacedTile, TileType};

use crate::serialization;

const MAGIC: &[u8; 4] = b"DTRJ";
const VERSION: u32 = 2;

/// A loaded game trajectory with full GameStates.
pub struct GameTrajectory {
    pub states: Vec<GameState>,
    pub result: GameResult,
}

/// Streaming writer -- writes games incrementally, patches count on finish.
pub struct TrajectoryWriter {
    writer: BufWriter<std::fs::File>,
    num_games: u32,
}

impl TrajectoryWriter {
    pub fn new(path: &str) -> io::Result<Self> {
        let f = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(f);
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&0u32.to_le_bytes())?; // placeholder
        Ok(Self { writer, num_games: 0 })
    }

    pub fn write_game(&mut self, states: &[GameState], result: &GameResult) -> io::Result<()> {
        assert!(states.len() <= u16::MAX as usize,
            "game has {} states, exceeds u16::MAX for serialization", states.len());
        serialization::write_result(&mut self.writer, result)?;
        self.writer.write_all(&(states.len() as u16).to_le_bytes())?;
        for gs in states {
            write_game_state(&mut self.writer, gs)?;
        }
        self.num_games += 1;
        Ok(())
    }

    /// Flush data and update the game count header so the file is recoverable if the process crashes.
    pub fn sync(&mut self) -> io::Result<()> {
        use std::io::Seek;
        self.writer.flush()?;
        let f = self.writer.get_mut();
        let pos = f.stream_position()?;
        f.seek(io::SeekFrom::Start(8))?;
        f.write_all(&self.num_games.to_le_bytes())?;
        f.flush()?;
        f.seek(io::SeekFrom::Start(pos))?;
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<u32> {
        self.sync()?;
        Ok(self.num_games)
    }
}

impl Drop for TrajectoryWriter {
    fn drop(&mut self) {
        let _ = self.sync();
    }
}

/// Load all trajectories from a file, reconstructing full GameStates.
///
/// Reads the entire file into memory first, then parses from a byte slice
/// to avoid millions of tiny syscalls via BufReader.
pub fn load_trajectories(path: &str) -> io::Result<Vec<GameTrajectory>> {
    let data = std::fs::read(path)?;
    let mut cursor = &data[..];

    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid trajectory file magic"));
    }

    let mut buf4 = [0u8; 4];
    cursor.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported trajectory version (expected {}, got {})", VERSION, version),
        ));
    }

    cursor.read_exact(&mut buf4)?;
    let num_games = u32::from_le_bytes(buf4) as usize;

    let mut games = Vec::with_capacity(num_games);
    for _ in 0..num_games {
        let result = serialization::read_result(&mut cursor)?;
        let mut buf2 = [0u8; 2];
        cursor.read_exact(&mut buf2)?;
        let num_states = u16::from_le_bytes(buf2) as usize;

        let mut states = Vec::with_capacity(num_states);
        for _ in 0..num_states {
            states.push(read_game_state(&mut cursor)?);
        }
        games.push(GameTrajectory { states, result });
    }

    Ok(games)
}

// --- Serialization helpers ---

pub fn write_game_state(w: &mut impl Write, gs: &GameState) -> io::Result<()> {
    // Current player
    let player_byte: u8 = match gs.current_player_turn() {
        Owner::TopPlayer => 0,
        Owner::BottomPlayer => 1,
    };
    w.write_all(&[player_byte])?;

    // Idle move count
    assert!(gs.idle_move_count() <= 255,
        "idle_move_count {} exceeds u8::MAX, would be truncated in serialization",
        gs.idle_move_count());
    w.write_all(&[gs.idle_move_count() as u8])?;

    // Board tiles
    let board = gs.board();
    let tiles: Vec<_> = board.active_coordinates().collect();
    assert!(tiles.len() <= 255,
        "board has {} tiles, exceeds u8::MAX for serialization", tiles.len());
    w.write_all(&[tiles.len() as u8])?;
    for (coords, placed) in &tiles {
        w.write_all(&[coords.x as u8, coords.y as u8])?;
        w.write_all(&[placed.tile_type as u8])?;
        w.write_all(&[match placed.current_side {
            CurrentSide::Initial => 0u8,
            CurrentSide::Flipped => 1u8,
        }])?;
        w.write_all(&[match placed.owner {
            Owner::TopPlayer => 0u8,
            Owner::BottomPlayer => 1u8,
        }])?;
    }

    // Bags and discard piles
    write_tile_list(w, gs.bag_for_owner(Owner::TopPlayer).remaining())?;
    write_tile_list(w, gs.bag_for_owner(Owner::BottomPlayer).remaining())?;
    write_tile_list(w, gs.discard_bag_for(Owner::TopPlayer).existing())?;
    write_tile_list(w, gs.discard_bag_for(Owner::BottomPlayer).existing())?;

    Ok(())
}

fn write_tile_list(w: &mut impl Write, tiles: &[TileType]) -> io::Result<()> {
    assert!(tiles.len() <= 255,
        "tile list has {} entries, exceeds u8::MAX for serialization", tiles.len());
    w.write_all(&[tiles.len() as u8])?;
    for tile_type in tiles {
        w.write_all(&[*tile_type as u8])?;
    }
    Ok(())
}

fn read_game_state(r: &mut impl Read) -> io::Result<GameState> {
    let mut buf1 = [0u8; 1];

    // Current player
    r.read_exact(&mut buf1)?;
    let current_player = match buf1[0] {
        0 => Owner::TopPlayer,
        1 => Owner::BottomPlayer,
        b => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("Invalid player byte: {}", b))),
    };

    // Idle move count
    r.read_exact(&mut buf1)?;
    let idle_moves = buf1[0] as usize;

    // Board tiles
    r.read_exact(&mut buf1)?;
    let num_tiles = buf1[0] as usize;
    let mut tiles = Vec::with_capacity(num_tiles);
    for _ in 0..num_tiles {
        let mut tile_buf = [0u8; 5]; // x, y, tile_type, side, owner
        r.read_exact(&mut tile_buf)?;
        let coords = Coordinates { x: tile_buf[0], y: tile_buf[1] };
        let tile_type = tile_type_from_u8(tile_buf[2])?;
        let side = match tile_buf[3] {
            0 => CurrentSide::Initial,
            1 => CurrentSide::Flipped,
            b => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("Invalid side byte: {}", b))),
        };
        let owner = match tile_buf[4] {
            0 => Owner::TopPlayer,
            1 => Owner::BottomPlayer,
            b => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("Invalid owner byte: {}", b))),
        };
        let mut placed = PlacedTile::new(owner, tile_type);
        if side == CurrentSide::Flipped {
            placed.flip();
        }
        tiles.push((coords, placed));
    }

    // Bags and discard piles
    let top_bag = TileBag::new(read_tile_type_list(r)?);
    let bottom_bag = TileBag::new(read_tile_type_list(r)?);
    let top_discard = DiscardBag::from_tiles(read_tile_type_list(r)?);
    let bottom_discard = DiscardBag::from_tiles(read_tile_type_list(r)?);

    Ok(GameState::from_snapshot(GameSnapshot {
        tiles,
        current_turn: current_player,
        top_bag,
        bottom_bag,
        top_discard,
        bottom_discard,
        idle_move_count: idle_moves,
    }))
}

fn read_tile_type_list(r: &mut impl Read) -> io::Result<Vec<TileType>> {
    let mut buf1 = [0u8; 1];
    r.read_exact(&mut buf1)?;
    let len = buf1[0] as usize;
    let mut tiles = Vec::with_capacity(len);
    for _ in 0..len {
        r.read_exact(&mut buf1)?;
        let tt = tile_type_from_u8(buf1[0])?;
        tiles.push(tt);
    }
    Ok(tiles)
}

fn tile_type_from_u8(b: u8) -> io::Result<TileType> {
    TileType::try_from(b).map_err(|b| {
        io::Error::new(io::ErrorKind::InvalidData, format!("Invalid TileType byte: {}", b))
    })
}
