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

use std::io::{self, Read, Write, BufWriter};

use duke_rust::game::state::GameResult;
use duke_rust::game::tile::Owner;

use crate::serialization;

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

/// Streaming feature cache writer — crash-safe with periodic sync.
pub struct FeatureCacheWriter {
    writer: BufWriter<std::fs::File>,
    num_features: usize,
    num_games: u32,
}

impl FeatureCacheWriter {
    pub fn new(path: &str, num_features: usize) -> io::Result<Self> {
        let f = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(f);
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&(num_features as u32).to_le_bytes())?;
        writer.write_all(&0u32.to_le_bytes())?; // placeholder game count
        Ok(Self { writer, num_features, num_games: 0 })
    }

    pub fn write_game(&mut self, game: &CachedGame) -> io::Result<()> {
        serialization::write_result(&mut self.writer, &game.result)?;
        self.writer.write_all(&(game.states.len() as u32).to_le_bytes())?;
        for state in &game.states {
            assert_eq!(state.features.len(), self.num_features,
                "feature vector length mismatch: expected {}, got {}",
                self.num_features, state.features.len());
            let player_byte: u8 = match state.current_player {
                Owner::TopPlayer => 0,
                Owner::BottomPlayer => 1,
            };
            self.writer.write_all(&[player_byte])?;
            for &val in &state.features {
                self.writer.write_all(&val.to_le_bytes())?;
            }
        }
        self.num_games += 1;
        Ok(())
    }

    pub fn sync(&mut self) -> io::Result<()> {
        use std::io::Seek;
        self.writer.flush()?;
        let f = self.writer.get_mut();
        let pos = f.stream_position()?;
        // Patch game count at offset 12 (after magic + version + num_features)
        f.seek(io::SeekFrom::Start(12))?;
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

impl Drop for FeatureCacheWriter {
    fn drop(&mut self) {
        let _ = self.sync();
    }
}

/// Save extracted features to a binary cache file (batch, non-streaming).
pub fn save_feature_cache(
    path: &str,
    num_features: usize,
    games: &[CachedGame],
) -> io::Result<()> {
    let mut writer = FeatureCacheWriter::new(path, num_features)?;
    for game in games {
        writer.write_game(game)?;
    }
    writer.finish()?;
    Ok(())
}

/// Parse a feature cache header from any reader.
/// Returns (num_features, num_games).
fn parse_header(reader: &mut impl Read) -> io::Result<FeatureCacheHeader> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid feature cache magic"));
    }

    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported feature cache version (expected {}, got {})", VERSION, version),
        ));
    }

    reader.read_exact(&mut buf4)?;
    let num_features = u32::from_le_bytes(buf4) as usize;

    reader.read_exact(&mut buf4)?;
    let num_games = u32::from_le_bytes(buf4) as usize;

    Ok(FeatureCacheHeader { num_features, num_games })
}

/// Deserialize a single game from any reader.
fn read_one_game(reader: &mut impl Read, num_features: usize) -> io::Result<CachedGame> {
    let result = serialization::read_result(reader)?;

    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let num_states = u32::from_le_bytes(buf4) as usize;

    let mut buf1 = [0u8; 1];
    let mut buf8 = [0u8; 8];
    let mut states = Vec::with_capacity(num_states);
    for _ in 0..num_states {
        reader.read_exact(&mut buf1)?;
        let current_player = match buf1[0] {
            0 => Owner::TopPlayer,
            1 => Owner::BottomPlayer,
            b => return Err(io::Error::new(
                io::ErrorKind::InvalidData, format!("Invalid player byte: {}", b),
            )),
        };
        let mut features = Vec::with_capacity(num_features);
        for _ in 0..num_features {
            reader.read_exact(&mut buf8)?;
            features.push(f64::from_le_bytes(buf8));
        }
        states.push(CachedState { current_player, features });
    }

    Ok(CachedGame { result, states })
}

/// Load feature cache, returning header info and all games.
pub fn load_feature_cache(path: &str) -> io::Result<(FeatureCacheHeader, Vec<CachedGame>)> {
    let data = std::fs::read(path)?;
    let mut cursor = &data[..];

    let header = parse_header(&mut cursor)?;

    let mut games = Vec::with_capacity(header.num_games);
    for _ in 0..header.num_games {
        games.push(read_one_game(&mut cursor, header.num_features)?);
    }

    Ok((header, games))
}

/// Stream a feature cache file in chunks, calling `process` for each chunk of games.
/// Never loads more than `chunk_size` games into memory at once.
pub fn stream_feature_cache(
    path: &str,
    chunk_size: usize,
    mut process: impl FnMut(&[CachedGame], &FeatureCacheHeader),
) -> io::Result<FeatureCacheHeader> {
    use std::io::BufReader;

    assert!(chunk_size > 0, "chunk_size must be > 0");

    let f = std::fs::File::open(path)?;
    let mut r = BufReader::with_capacity(1 << 20, f); // 1MB buffer

    let header = parse_header(&mut r)?;

    let mut chunk = Vec::with_capacity(chunk_size);
    let mut games_read = 0;

    while games_read < header.num_games {
        chunk.clear();
        let batch = chunk_size.min(header.num_games - games_read);
        for _ in 0..batch {
            chunk.push(read_one_game(&mut r, header.num_features)?);
        }
        games_read += batch;
        process(&chunk, &header);
    }

    Ok(header)
}

/// Build a RegressionAccumulator directly from cached features (no GameState needed).
pub fn accumulate_from_cache(games: &[CachedGame], num_features: usize) -> crate::regression::RegressionAccumulator {
    use crate::regression::RegressionAccumulator;
    use crate::learned_heuristic::NUM_FEATURES;

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

