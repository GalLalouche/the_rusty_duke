//! Shared serialization helpers for binary file formats.

use std::io::{self, Read, Write};

use duke_rust::game::state::GameResult;
use duke_rust::game::tile::Owner;

/// Write a `GameResult` as a single byte: 0=TopWin, 1=BottomWin, 2=Tie, 3=Ongoing.
pub fn write_result(w: &mut impl Write, result: &GameResult) -> io::Result<()> {
    let byte = match result {
        GameResult::Won(Owner::TopPlayer) => 0u8,
        GameResult::Won(Owner::BottomPlayer) => 1u8,
        GameResult::Tie => 2u8,
        GameResult::Ongoing => 3u8,
    };
    w.write_all(&[byte])
}

/// Read a `GameResult` from a single byte.
pub fn read_result(r: &mut impl Read) -> io::Result<GameResult> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    match buf[0] {
        0 => Ok(GameResult::Won(Owner::TopPlayer)),
        1 => Ok(GameResult::Won(Owner::BottomPlayer)),
        2 => Ok(GameResult::Tie),
        3 => Ok(GameResult::Ongoing),
        b => Err(io::Error::new(io::ErrorKind::InvalidData, format!("Invalid result byte: {}", b))),
    }
}
