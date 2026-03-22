//! Extract and cache heuristic features from saved game trajectories.
//!
//! Binary format (little-endian):
//!   Header: b"FEAT" + version(u32) + num_features(u32) + num_games(u32)
//!   Per game:
//!     result: u8 (0=TopWin, 1=BottomWin, 2=Tie)
//!     num_states: u32
//!     Per state:
//!       current_player: u8 (0=Top, 1=Bottom)
//!       features: [f64; num_features]

use std::io::{self, Read, Write, BufWriter, BufReader};

use duke_rust::game::state::GameResult;
use duke_rust::game::tile::Owner;

const MAGIC: &[u8; 4] = b"FEAT";
const VERSION: u32 = 1;

/// A single state's cached features.
pub struct CachedState {
    pub current_player: Owner,
    pub features: Vec<f64>,
}

/// A game's cached feature data.
pub struct CachedGame {
    pub result: GameResult,
    pub states: Vec<CachedState>,
}

/// Metadata from a feature cache file header.
pub struct FeatureCacheHeader {
    pub num_features: usize,
    pub num_games: usize,
}

/// Save extracted features to a binary cache file.
pub fn save_feature_cache(
    path: &str,
    num_features: usize,
    games: &[CachedGame],
) -> io::Result<()> {
    let f = std::fs::File::create(path)?;
    let mut w = BufWriter::new(f);

    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(num_features as u32).to_le_bytes())?;
    w.write_all(&(games.len() as u32).to_le_bytes())?;

    for game in games {
        write_result(&mut w, &game.result)?;
        w.write_all(&(game.states.len() as u32).to_le_bytes())?;
        for state in &game.states {
            let player_byte: u8 = match state.current_player {
                Owner::TopPlayer => 0,
                Owner::BottomPlayer => 1,
            };
            w.write_all(&[player_byte])?;
            for &val in &state.features {
                w.write_all(&val.to_le_bytes())?;
            }
        }
    }

    w.flush()
}

/// Load feature cache, returning header info and all games.
pub fn load_feature_cache(path: &str) -> io::Result<(FeatureCacheHeader, Vec<CachedGame>)> {
    let data = std::fs::read(path)?;
    let mut cursor = &data[..];

    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid feature cache magic"));
    }

    let mut buf4 = [0u8; 4];
    cursor.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported feature cache version (expected {}, got {})", VERSION, version),
        ));
    }

    cursor.read_exact(&mut buf4)?;
    let num_features = u32::from_le_bytes(buf4) as usize;

    cursor.read_exact(&mut buf4)?;
    let num_games = u32::from_le_bytes(buf4) as usize;

    let header = FeatureCacheHeader { num_features, num_games };

    let mut games = Vec::with_capacity(num_games);
    let mut buf1 = [0u8; 1];
    let mut buf8 = [0u8; 8];

    for _ in 0..num_games {
        let result = read_result(&mut cursor)?;
        cursor.read_exact(&mut buf4)?;
        let num_states = u32::from_le_bytes(buf4) as usize;

        let mut states = Vec::with_capacity(num_states);
        for _ in 0..num_states {
            cursor.read_exact(&mut buf1)?;
            let current_player = match buf1[0] {
                0 => Owner::TopPlayer,
                1 => Owner::BottomPlayer,
                b => return Err(io::Error::new(
                    io::ErrorKind::InvalidData, format!("Invalid player byte: {}", b),
                )),
            };
            let mut features = Vec::with_capacity(num_features);
            for _ in 0..num_features {
                cursor.read_exact(&mut buf8)?;
                features.push(f64::from_le_bytes(buf8));
            }
            states.push(CachedState { current_player, features });
        }
        games.push(CachedGame { result, states });
    }

    Ok((header, games))
}

/// Build a RegressionAccumulator directly from cached features (no GameState needed).
pub fn accumulate_from_cache(games: &[CachedGame], num_features: usize) -> crate::learned_heuristic::RegressionAccumulator {
    use crate::learned_heuristic::{RegressionAccumulator, NUM_FEATURES};

    assert_eq!(num_features, NUM_FEATURES,
        "Feature cache has {} features but accumulator expects {}", num_features, NUM_FEATURES);

    let mut acc = RegressionAccumulator::new();
    for game in games {
        for state in &game.states {
            if state.features.len() != num_features {
                continue;
            }
            let current = state.current_player;
            let target = match game.result {
                GameResult::Won(winner) => {
                    if winner == current { 1.0 } else { -1.0 }
                }
                GameResult::Tie | GameResult::Ongoing => 0.0,
            };
            // Copy features into fixed-size array for accumulator
            let mut feats = [0.0f64; NUM_FEATURES];
            feats.copy_from_slice(&state.features);
            acc.add_sample(&feats, target);
        }
    }
    acc
}

// --- Helpers ---

fn write_result(w: &mut impl Write, result: &GameResult) -> io::Result<()> {
    let byte = match result {
        GameResult::Won(Owner::TopPlayer) => 0u8,
        GameResult::Won(Owner::BottomPlayer) => 1u8,
        GameResult::Tie => 2u8,
        GameResult::Ongoing => 3u8,
    };
    w.write_all(&[byte])
}

fn read_result(r: &mut impl Read) -> io::Result<GameResult> {
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
