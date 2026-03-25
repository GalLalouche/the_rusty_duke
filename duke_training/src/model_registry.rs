//! SQLite-based model registry for tracking trained models and benchmark results.
//!
//! Stores model metadata (architecture, param count, training hyperparameters)
//! and benchmark results (win/loss/tie records, Elo ratings) in a local SQLite DB.

use rusqlite::{params, Connection, OptionalExtension};
use std::time::SystemTime;

use crate::generic_mlp::GenericMlp;
use crate::nnue::{NnueWeights, NUM_FEATURES};

/// Format the current time as an ISO 8601 string (UTC).
fn now_iso8601() -> String {
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Convert epoch seconds to date-time components
    // Using a simple algorithm (valid for 2000-2099)
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Days since 1970-01-01
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y, m, d, hours, minutes, seconds
    )
}

/// Training hyperparameters and lineage for a registered model.
pub struct TrainingInfo {
    pub iterations: Option<u32>,
    pub sigma: Option<f32>,
    pub lr: Option<f32>,
    pub opponent: Option<String>,
    pub parent_model_id: Option<i64>,
}

/// A benchmark result record (win/loss/tie against an opponent).
pub struct BenchmarkRecord {
    pub id: Option<i64>,
    pub opponent: String,
    pub opponent_model_id: Option<i64>,
    pub num_games: u32,
    pub wins: u32,
    pub losses: u32,
    pub ties: u32,
    pub win_rate: Option<f64>,
    pub elo: Option<f64>,
    pub benchmark_date: Option<String>,
}

/// A model record from the registry.
pub struct ModelRecord {
    pub id: i64,
    pub file_path: String,
    pub file_format: String,
    pub architecture: String,
    pub input_size: usize,
    pub param_count: usize,
    pub description: Option<String>,
    pub created_at: String,
    pub training_iterations: Option<u32>,
    pub training_sigma: Option<f64>,
    pub training_lr: Option<f64>,
    pub training_opponent: Option<String>,
    pub parent_model_id: Option<i64>,
}

/// SQLite-backed model registry.
pub struct ModelRegistry {
    conn: Connection,
}

impl ModelRegistry {
    /// Open (or create) a registry database at the given path.
    ///
    /// Creates tables if they don't exist and enables WAL mode.
    pub fn open(db_path: &str) -> Result<Self, rusqlite::Error> {
        let conn = if db_path == ":memory:" {
            Connection::open_in_memory()?
        } else {
            Connection::open(db_path)?
        };
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS models (
                id                  INTEGER PRIMARY KEY,
                file_path           TEXT NOT NULL UNIQUE,
                file_format         TEXT NOT NULL,
                architecture        TEXT NOT NULL,
                input_size          INTEGER NOT NULL,
                param_count         INTEGER NOT NULL,
                description         TEXT,
                created_at          TEXT NOT NULL,
                training_iterations INTEGER,
                training_sigma      REAL,
                training_lr         REAL,
                training_opponent   TEXT,
                parent_model_id     INTEGER REFERENCES models(id)
            );

            CREATE TABLE IF NOT EXISTS benchmarks (
                id                INTEGER PRIMARY KEY,
                model_id          INTEGER NOT NULL REFERENCES models(id),
                opponent           TEXT NOT NULL,
                opponent_model_id  INTEGER REFERENCES models(id),
                num_games          INTEGER NOT NULL,
                wins               INTEGER NOT NULL,
                losses             INTEGER NOT NULL,
                ties               INTEGER NOT NULL,
                win_rate           REAL NOT NULL,
                elo                REAL,
                benchmark_date     TEXT NOT NULL
            );",
        )?;

        Ok(Self { conn })
    }

    /// Register a GenericMlp model. Returns the new model ID.
    pub fn register_gmlp(
        &self,
        path: &str,
        net: &GenericMlp,
        description: Option<&str>,
        training: Option<&TrainingInfo>,
    ) -> Result<i64, rusqlite::Error> {
        let arch = net.arch_string();
        let param_count = net.weights.len();
        let now = now_iso8601();

        let (iterations, sigma, lr, opponent, parent) = match training {
            Some(t) => (
                t.iterations.map(|v| v as i64),
                t.sigma.map(|v| v as f64),
                t.lr.map(|v| v as f64),
                t.opponent.as_deref(),
                t.parent_model_id,
            ),
            None => (None, None, None, None, None),
        };

        self.conn.execute(
            "INSERT INTO models (file_path, file_format, architecture, input_size,
                param_count, description, created_at,
                training_iterations, training_sigma, training_lr,
                training_opponent, parent_model_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                path,
                "gmlp",
                arch,
                net.input_size as i64,
                param_count as i64,
                description,
                now,
                iterations,
                sigma,
                lr,
                opponent,
                parent,
            ],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    /// Register an NNUE model. Returns the new model ID.
    pub fn register_nnue(
        &self,
        path: &str,
        weights: &NnueWeights,
        description: Option<&str>,
        training: Option<&TrainingInfo>,
    ) -> Result<i64, rusqlite::Error> {
        let arch = format!("{}->{}->{}->1", NUM_FEATURES, weights.l1_size, weights.l2_size);
        let param_count = weights.l1_weight.len()
            + weights.l1_bias.len()
            + weights.l2_weight.len()
            + weights.l2_bias.len()
            + weights.l3_weight.len()
            + weights.l3_bias.len();
        let now = now_iso8601();

        let (iterations, sigma, lr, opponent, parent) = match training {
            Some(t) => (
                t.iterations.map(|v| v as i64),
                t.sigma.map(|v| v as f64),
                t.lr.map(|v| v as f64),
                t.opponent.as_deref(),
                t.parent_model_id,
            ),
            None => (None, None, None, None, None),
        };

        self.conn.execute(
            "INSERT INTO models (file_path, file_format, architecture, input_size,
                param_count, description, created_at,
                training_iterations, training_sigma, training_lr,
                training_opponent, parent_model_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                path,
                "nnue",
                arch,
                NUM_FEATURES as i64,
                param_count as i64,
                description,
                now,
                iterations,
                sigma,
                lr,
                opponent,
                parent,
            ],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    /// Record a benchmark result for a model. Returns the new benchmark ID.
    pub fn record_benchmark(
        &self,
        model_id: i64,
        result: &BenchmarkRecord,
    ) -> Result<i64, rusqlite::Error> {
        let now = result
            .benchmark_date
            .clone()
            .unwrap_or_else(|| now_iso8601());
        let win_rate = result.win_rate.unwrap_or_else(|| {
            let total = result.num_games as f64;
            if total > 0.0 {
                (result.wins as f64 + 0.5 * result.ties as f64) / total
            } else {
                0.0
            }
        });

        self.conn.execute(
            "INSERT INTO benchmarks (model_id, opponent, opponent_model_id,
                num_games, wins, losses, ties, win_rate, elo, benchmark_date)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                model_id,
                result.opponent,
                result.opponent_model_id,
                result.num_games as i64,
                result.wins as i64,
                result.losses as i64,
                result.ties as i64,
                win_rate,
                result.elo,
                now,
            ],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    /// Find a model by its file path.
    pub fn find_by_path(&self, path: &str) -> Result<Option<ModelRecord>, rusqlite::Error> {
        self.conn
            .query_row(
                "SELECT id, file_path, file_format, architecture, input_size,
                    param_count, description, created_at,
                    training_iterations, training_sigma, training_lr,
                    training_opponent, parent_model_id
                 FROM models WHERE file_path = ?1",
                params![path],
                |row| row_to_model_record(row),
            )
            .optional()
    }

    /// List all registered models, ordered by creation time (newest first).
    pub fn list_models(&self) -> Result<Vec<ModelRecord>, rusqlite::Error> {
        let mut stmt = self.conn.prepare(
            "SELECT id, file_path, file_format, architecture, input_size,
                param_count, description, created_at,
                training_iterations, training_sigma, training_lr,
                training_opponent, parent_model_id
             FROM models ORDER BY created_at DESC",
        )?;

        let rows = stmt.query_map([], |row| row_to_model_record(row))?;
        rows.collect()
    }

    /// Get all benchmark records for a model.
    pub fn get_benchmarks(&self, model_id: i64) -> Result<Vec<BenchmarkRecord>, rusqlite::Error> {
        let mut stmt = self.conn.prepare(
            "SELECT id, opponent, opponent_model_id, num_games, wins, losses,
                ties, win_rate, elo, benchmark_date
             FROM benchmarks WHERE model_id = ?1 ORDER BY benchmark_date DESC",
        )?;

        let rows = stmt.query_map(params![model_id], |row| {
            Ok(BenchmarkRecord {
                id: Some(row.get::<_, i64>(0)?),
                opponent: row.get(1)?,
                opponent_model_id: row.get(2)?,
                num_games: row.get::<_, i64>(3)? as u32,
                wins: row.get::<_, i64>(4)? as u32,
                losses: row.get::<_, i64>(5)? as u32,
                ties: row.get::<_, i64>(6)? as u32,
                win_rate: Some(row.get::<_, f64>(7)?),
                elo: row.get(8)?,
                benchmark_date: Some(row.get::<_, String>(9)?),
            })
        })?;

        rows.collect()
    }
}

fn row_to_model_record(row: &rusqlite::Row) -> Result<ModelRecord, rusqlite::Error> {
    Ok(ModelRecord {
        id: row.get(0)?,
        file_path: row.get(1)?,
        file_format: row.get(2)?,
        architecture: row.get(3)?,
        input_size: row.get::<_, i64>(4)? as usize,
        param_count: row.get::<_, i64>(5)? as usize,
        description: row.get(6)?,
        created_at: row.get(7)?,
        training_iterations: row.get::<_, Option<i64>>(8)?.map(|v| v as u32),
        training_sigma: row.get(9)?,
        training_lr: row.get(10)?,
        training_opponent: row.get(11)?,
        parent_model_id: row.get(12)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_gmlp(input_size: usize, hidden_layers: Vec<usize>) -> GenericMlp {
        let param_count = GenericMlp::param_count(input_size, &hidden_layers);
        GenericMlp::from_flat(vec![0.0; param_count], input_size, hidden_layers)
    }

    fn make_test_nnue(l1: usize, l2: usize) -> NnueWeights {
        NnueWeights {
            l1_size: l1,
            l2_size: l2,
            l1_weight: vec![0.0; NUM_FEATURES * l1],
            l1_bias: vec![0.0; l1],
            l2_weight: vec![0.0; l2 * l1],
            l2_bias: vec![0.0; l2],
            l3_weight: vec![0.0; l2],
            l3_bias: vec![0.0; 1],
        }
    }

    #[test]
    fn test_register_gmlp_and_query() {
        let reg = ModelRegistry::open(":memory:").unwrap();
        let net = make_test_gmlp(41, vec![64, 32]);
        let training = TrainingInfo {
            iterations: Some(100),
            sigma: Some(0.01),
            lr: Some(0.005),
            opponent: Some("base".to_string()),
            parent_model_id: None,
        };
        let id = reg
            .register_gmlp("/tmp/test.gmlp", &net, Some("test model"), Some(&training))
            .unwrap();
        assert_eq!(id, 1);

        let record = reg.find_by_path("/tmp/test.gmlp").unwrap().unwrap();
        assert_eq!(record.id, 1);
        assert_eq!(record.file_format, "gmlp");
        assert_eq!(record.architecture, "41->64->32->1");
        assert_eq!(record.input_size, 41);
        assert_eq!(record.param_count, GenericMlp::param_count(41, &[64, 32]));
        assert_eq!(record.description.as_deref(), Some("test model"));
        assert_eq!(record.training_iterations, Some(100));
        assert!((record.training_sigma.unwrap() - 0.01).abs() < 1e-6);
        assert!((record.training_lr.unwrap() - 0.005).abs() < 1e-6);
        assert_eq!(record.training_opponent.as_deref(), Some("base"));
        assert_eq!(record.parent_model_id, None);
    }

    #[test]
    fn test_register_nnue_and_query() {
        let reg = ModelRegistry::open(":memory:").unwrap();
        let weights = make_test_nnue(256, 32);
        let id = reg
            .register_nnue("/tmp/test.nnue", &weights, None, None)
            .unwrap();
        assert_eq!(id, 1);

        let record = reg.find_by_path("/tmp/test.nnue").unwrap().unwrap();
        assert_eq!(record.file_format, "nnue");
        assert_eq!(
            record.architecture,
            format!("{}->256->32->1", NUM_FEATURES)
        );
        assert_eq!(record.input_size, NUM_FEATURES);
    }

    #[test]
    fn test_list_multiple_models() {
        let reg = ModelRegistry::open(":memory:").unwrap();
        let net1 = make_test_gmlp(41, vec![64]);
        let net2 = make_test_gmlp(1106, vec![128, 64]);
        reg.register_gmlp("/tmp/a.gmlp", &net1, Some("model A"), None)
            .unwrap();
        reg.register_gmlp("/tmp/b.gmlp", &net2, Some("model B"), None)
            .unwrap();

        let models = reg.list_models().unwrap();
        assert_eq!(models.len(), 2);
        // Ordered newest first, but with in-memory DB timestamps may be identical;
        // just check both are present.
        let paths: Vec<&str> = models.iter().map(|m| m.file_path.as_str()).collect();
        assert!(paths.contains(&"/tmp/a.gmlp"));
        assert!(paths.contains(&"/tmp/b.gmlp"));
    }

    #[test]
    fn test_record_and_get_benchmarks() {
        let reg = ModelRegistry::open(":memory:").unwrap();
        let net = make_test_gmlp(41, vec![32]);
        let model_id = reg
            .register_gmlp("/tmp/bench.gmlp", &net, None, None)
            .unwrap();

        let bench = BenchmarkRecord {
            id: None,
            opponent: "base".to_string(),
            opponent_model_id: None,
            num_games: 1000,
            wins: 600,
            losses: 350,
            ties: 50,
            win_rate: None, // should be auto-computed
            elo: Some(1650.0),
            benchmark_date: None,
        };
        let bench_id = reg.record_benchmark(model_id, &bench).unwrap();
        assert!(bench_id > 0);

        let benchmarks = reg.get_benchmarks(model_id).unwrap();
        assert_eq!(benchmarks.len(), 1);
        let b = &benchmarks[0];
        assert_eq!(b.opponent, "base");
        assert_eq!(b.num_games, 1000);
        assert_eq!(b.wins, 600);
        assert_eq!(b.losses, 350);
        assert_eq!(b.ties, 50);
        // win_rate = (600 + 0.5*50) / 1000 = 0.625
        assert!((b.win_rate.unwrap() - 0.625).abs() < 1e-6);
        assert!((b.elo.unwrap() - 1650.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_by_path_not_found() {
        let reg = ModelRegistry::open(":memory:").unwrap();
        let result = reg.find_by_path("/nonexistent/model.gmlp").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_duplicate_path_errors() {
        let reg = ModelRegistry::open(":memory:").unwrap();
        let net = make_test_gmlp(41, vec![32]);
        reg.register_gmlp("/tmp/dup.gmlp", &net, None, None)
            .unwrap();
        let result = reg.register_gmlp("/tmp/dup.gmlp", &net, None, None);
        assert!(result.is_err());
    }
}
