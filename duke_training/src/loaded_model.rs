//! Model loading utilities: LoadedModel struct, load_opponent, and load_opponent_quantized.
//!
//! These are consumers of GenericMlp, not part of it. Separated from generic_mlp.rs
//! to keep that module focused on the network implementation and evaluator wrappers.

use crate::generic_mlp::{
    GenericMlp, CombinedNetEvaluator, GuardFeatureEvaluator,
    GenericNnueEvaluator, GenericAppendedEvaluator, AllAppendedEvaluator,
    QuantizedNnueEvaluator, QuantizedAppendedEvaluator,
};
use crate::game_setup::{GameEvaluator, StaticHeuristicEvaluator};
use crate::learned_heuristic::{
    load_lr_weights_raw,
    AllFeaturesWeights, CombinedWeights, LearnedHeuristicWeights,
    NUM_ALL_FEATURES, NUM_COMBINED_FEATURES, NUM_FEATURES as LR_NUM_FEATURES,
};
use crate::model_registry::ModelRegistry;
use crate::nnue::{NnueEvaluator, NnueWeights, NUM_FEATURES};

/// Total guard feature count: 24 expensive + 41 combined = 65.
pub const NUM_GUARD_ALL_FEATURES: usize = 24 + NUM_COMBINED_FEATURES;

/// A loaded model ready for evaluation. Can represent a DB-registered model,
/// a file-based model, or a built-in player (base/random).
pub struct LoadedModel {
    /// DB model ID, None if not in registry.
    pub id: Option<i64>,
    /// Short display label (e.g. "Base", "Random", "es_final (1106->64->64->32->1)").
    pub label: String,
    /// None = Random player (no evaluator needed).
    pub evaluator: Option<Box<dyn GameEvaluator + Sync + Send>>,
}

impl LoadedModel {
    /// Load from a spec string: "base", "random", or a file path (.gmlp/.nnue).
    pub fn from_spec(spec: &str) -> Self {
        let (eval, desc) = load_opponent(spec);
        let label = if spec == "base" || spec == "random" {
            spec.to_string()
        } else {
            // Use filename stem + arch from description
            let stem = std::path::Path::new(spec)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| spec.to_string());
            // Extract parenthesized part from desc if present
            if let Some(start) = desc.find('(') {
                format!("{} {}", stem, &desc[start..])
            } else {
                stem
            }
        };
        LoadedModel { id: None, label, evaluator: eval }
    }

    /// Load from a database model ID. Opens the registry, looks up the model,
    /// loads the file, and populates the struct.
    pub fn from_db_id(registry: &ModelRegistry, model_id: i64) -> Result<Self, String> {
        let record = registry
            .get_model(model_id)
            .map_err(|e| format!("DB error looking up model #{}: {}", model_id, e))?
            .ok_or_else(|| format!("Model #{} not found in registry", model_id))?;

        let (eval, _desc) = load_opponent(&record.file_path);
        let label = if let Some(ref desc) = record.description {
            desc.clone()
        } else {
            format!("{} ({})",
                std::path::Path::new(&record.file_path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| record.file_path.clone()),
                record.architecture)
        };

        Ok(LoadedModel {
            id: Some(model_id),
            label,
            evaluator: eval,
        })
    }

    /// Load from a spec string with quantization for .gmlp models.
    /// Same as [`from_spec`] but wraps 1106/1147 .gmlp models in quantized evaluators.
    pub fn from_spec_quantized(spec: &str) -> Self {
        let (eval, desc) = load_opponent_quantized(spec);
        let label = if spec == "base" || spec == "random" {
            spec.to_string()
        } else {
            let stem = std::path::Path::new(spec)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| spec.to_string());
            if let Some(start) = desc.find('(') {
                format!("{} {}", stem, &desc[start..])
            } else {
                stem
            }
        };
        LoadedModel { id: None, label, evaluator: eval }
    }

    /// Load from a database model ID with quantization for .gmlp models.
    /// Same as [`from_db_id`] but wraps 1106/1147 .gmlp models in quantized evaluators.
    pub fn from_db_id_quantized(registry: &ModelRegistry, model_id: i64) -> Result<Self, String> {
        let record = registry
            .get_model(model_id)
            .map_err(|e| format!("DB error looking up model #{}: {}", model_id, e))?
            .ok_or_else(|| format!("Model #{} not found in registry", model_id))?;

        let (eval, _desc) = load_opponent_quantized(&record.file_path);
        let label = if let Some(ref desc) = record.description {
            // Append " (Q)" to mark quantized
            if record.file_path.ends_with(".gmlp") {
                format!("{} (Q)", desc)
            } else {
                desc.clone()
            }
        } else {
            let stem = std::path::Path::new(&record.file_path)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| record.file_path.clone());
            if record.file_path.ends_with(".gmlp") {
                format!("{} ({}) (Q)", stem, record.architecture)
            } else {
                format!("{} ({})", stem, record.architecture)
            }
        };

        Ok(LoadedModel {
            id: Some(model_id),
            label,
            evaluator: eval,
        })
    }

    /// Convert to a Player reference for match_runner.
    pub fn as_player(&self) -> crate::match_runner::Player<'_> {
        match &self.evaluator {
            Some(eval) => crate::match_runner::Player::Evaluator(eval.as_ref()),
            None => crate::match_runner::Player::Random,
        }
    }

    /// Convert to a Player reference with a specific search depth.
    /// depth=1 uses greedy (depth-1) search, depth=2 uses minimax depth-2.
    /// Random players always stay Random regardless of depth.
    pub fn as_player_with_depth(&self, depth: u32) -> crate::match_runner::Player<'_> {
        match &self.evaluator {
            Some(eval) => match depth {
                2 => crate::match_runner::Player::EvaluatorDepth2(eval.as_ref()),
                _ => crate::match_runner::Player::Evaluator(eval.as_ref()),
            },
            None => crate::match_runner::Player::Random,
        }
    }
}

/// Load an opponent evaluator from a file path or keyword, with quantization for .gmlp models.
///
/// Same as [`load_opponent`], but wraps `.gmlp` models (input_size 1106 or 1147) in
/// quantized evaluators using `GenericMlp::quantize()`. Non-`.gmlp` specs fall through
/// to [`load_opponent`].
pub fn load_opponent_quantized(spec: &str) -> (Option<Box<dyn GameEvaluator + Sync + Send>>, String) {
    match spec {
        path if path.ends_with(".gmlp") => {
            let net = GenericMlp::load(path).expect("Failed to load .gmlp opponent");
            let qnet = net.quantize();
            let desc = format!("GMLP-Q ({})", qnet.arch_string());
            let eval: Box<dyn GameEvaluator + Sync + Send> = match qnet.input_size {
                1106 => Box::new(QuantizedNnueEvaluator { qnet }),
                1147 => Box::new(QuantizedAppendedEvaluator { qnet }),
                other => {
                    // Quantization not yet supported for other input sizes (e.g. 1171).
                    // Fall through to non-quantized loading.
                    eprintln!(
                        "WARNING: quantization not supported for input_size {} in '{}'. \
                         Loading as NON-QUANTIZED f32 model. Performance will differ from quantized models.",
                        other, path
                    );
                    return load_opponent(path);
                }
            };
            (Some(eval), desc)
        }
        _ => load_opponent(spec), // non-gmlp: fall through to normal
    }
}

/// Load an opponent evaluator from a file path or keyword.
///
/// Returns `None` for the "random" keyword (caller should use `Player::Random`),
/// or `Some(evaluator)` for all other cases.
///
/// Supported formats:
///   - "base" or absent  -> StaticHeuristicEvaluator
///   - "random"          -> None (caller uses Player::Random)
///   - path ending .gmlp -> GenericMlp, dispatched by input_size
///   - path ending .nnue -> NnueWeights wrapped in NnueEvaluator
pub fn load_opponent(spec: &str) -> (Option<Box<dyn GameEvaluator + Sync + Send>>, String) {
    match spec {
        "base" => {
            let eval = StaticHeuristicEvaluator::new();
            (Some(Box::new(eval)), "Base heuristic".to_string())
        }
        "random" => {
            (None, "Random".to_string())
        }
        path if path.ends_with(".gmlp") => {
            let net = GenericMlp::load(path).expect("Failed to load .gmlp opponent");
            let desc = format!("GMLP ({})", net.arch_string());
            let eval: Box<dyn GameEvaluator + Sync + Send> = match net.input_size {
                41 => Box::new(CombinedNetEvaluator::new(net)),
                65 => Box::new(GuardFeatureEvaluator::new(net)),
                1106 => Box::new(GenericNnueEvaluator { net }),
                1147 => Box::new(GenericAppendedEvaluator { net }),
                1171 => Box::new(AllAppendedEvaluator { net }),
                other => panic!(
                    "Unknown input_size {} in .gmlp file '{}'. Expected 41, 65, 1106, 1147, or 1171.",
                    other, path
                ),
            };
            (Some(eval), desc)
        }
        path if path.ends_with(".nnue") => {
            let weights = NnueWeights::load(path).expect("Failed to load .nnue opponent");
            let desc = format!("NNUE ({}->{}->{}->1)", NUM_FEATURES, weights.l1_size, weights.l2_size);
            let eval = NnueEvaluator::new(weights);
            (Some(Box::new(eval)), desc)
        }
        path if path.ends_with(".json") => {
            let raw = load_lr_weights_raw(path).expect("Failed to load .json LR weights");
            let n = raw.len();
            match n {
                LR_NUM_FEATURES => {
                    let mut weights = [0.0f64; LR_NUM_FEATURES];
                    weights.copy_from_slice(&raw);
                    let lhw = LearnedHeuristicWeights { weights };
                    let desc = format!("LR-Guard ({} weights)", n);
                    (Some(Box::new(lhw)), desc)
                }
                NUM_COMBINED_FEATURES => {
                    let mut weights = [0.0f64; NUM_COMBINED_FEATURES];
                    weights.copy_from_slice(&raw);
                    let cw = CombinedWeights { weights };
                    let desc = format!("LR-Cheap ({} weights)", n);
                    (Some(Box::new(cw)), desc)
                }
                NUM_ALL_FEATURES => {
                    let mut weights = [0.0f64; NUM_ALL_FEATURES];
                    weights.copy_from_slice(&raw);
                    let aw = AllFeaturesWeights { weights };
                    let desc = format!("LR-All ({} weights)", n);
                    (Some(Box::new(aw)), desc)
                }
                _ => {
                    panic!(
                        "JSON weight file '{}' has {} weights. Expected {} (LR-Guard), {} (LR-Cheap), or {} (LR-All).",
                        path, n, LR_NUM_FEATURES, NUM_COMBINED_FEATURES, NUM_ALL_FEATURES
                    );
                }
            }
        }
        other => {
            panic!(
                "Unknown opponent '{}'. Use 'base', 'random', or a path ending in .gmlp / .nnue / .json",
                other
            );
        }
    }
}
