# Project Summary for Agents

## What This Is

AI training system for **The Duke**, a 2-player abstract strategy board game on a 6x6 board with tile-flipping mechanics and random tile draws from a bag. The goal is to build a neural network that can evaluate board positions and play well.

## Crate Structure

### `duke_rust` (game engine)
- `src/game/state.rs` — `GameState`: the core game state (board, bags, discards, current player)
- `src/game/board.rs` — `GameBoard`: 6x6 board, move generation, guard checking (apply/undo, no cloning)
- `src/game/tile.rs` — `PlacedTile` (3 bytes: tile_type + current_side + owner), `TileType` enum (13 types)
- `src/game/bag.rs` — `TileBag` and `DiscardBag` (fixed arrays, no heap)
- `src/game/ai/` — AI players: alpha-beta, random, heuristic evaluators
- `src/common/board.rs` — `Board<T>`: generic 6x6 board with `[Option<T>; 36]` storage

### `duke_training` (ML training)
- `src/game_setup.rs` — `GameEvaluator` trait, `negamax_ab` (with alpha-beta pruning + expectimax for tile draws), `greedy_move_deep`, game playing utilities
- `src/generic_mlp.rs` — `GenericMlp`: variable-depth FC network, `L1Accumulator` for incremental sparse updates
- `src/cnn.rs` — Manual CNN: Box/Diamond/Cross kernels, im2col+sgemm, forward/backward
- `src/halfda.rs` — HalfDA NNUE encoding (67,392 sparse features), `HalfDAAccumulator`, `HalfDAEvaluator`
- `src/encoding.rs` — Sparse board encoding: 30 planes x 36 squares + 26 bag features = 1106 total
- `src/learned_heuristic.rs` — 41 cheap heuristic features (manhattan, board control, duke mobility, discards)
- `src/supervised_common.rs` — Shared code: `AdamState`, `load_lpos`, `label_to_target`, `AdaptiveLrScheduler`
- `src/loaded_model.rs` — `LoadedModel`: loads any model type (.gmlp, .nnue, .json, HalfDA directory)
- `src/trajectory_io.rs` — Save/load game trajectories (DTRJ format)
- `src/match_runner.rs` — Run matches between players, compute win rates
- `src/model_registry.rs` — SQLite model registry with WHR ratings

## Key Binaries

| Binary | Purpose |
|---|---|
| `generate_games` | Self-play game generation with depth-N negamax. Saves trajectories + eval scores. |
| `score_trajectories` | Score existing trajectory positions with depth-N negamax. |
| `halfda_train` | Train HalfDA NNUE (67k->2048->1) with lambda mixing (eval + game outcome). GPU via burn/wgpu. |
| `supervised_train` | Train FC networks on labeled positions (LPOS/FLPS). Supports residual connections, sparse-init, val-split. |
| `supervised_train_cnn` | Train CNN on labeled positions. Box/Diamond/Cross kernels with im2col+sgemm. |
| `label_positions` | Label game positions: LR-Cheap depth-N eval, all 41 features (FLPS), or synthetic labels. |
| `elo_tournament` | Round-robin tournament with WHR ratings. Accepts model files, DB IDs, "base", "random". |
| `es_train` | Evolutionary Strategies training (perturb weights, play games, keep winners). |
| `generate_games` | Depth-N self-play with eval score saving. |

## Encodings

### 1106 Sparse Board Encoding (used by FC/CNN trainers)
- 30 planes x 36 squares = 1080 binary board features (13 tile types x 2 owners + 4 side planes)
- 26 bag features (13 tile types x 2 players)
- ~24 active features per position out of 1106

### HalfDA Encoding (used by NNUE trainer)
- duke_square (36) x piece_square (36) x piece_type (13) x color (2) x side (2) = 67,392 features
- ~7 active per position (one per non-duke piece on board)
- Designed for incremental accumulator updates (like Stockfish NNUE)

### 41 Cheap Features (used by LR-Cheap evaluator)
- Manhattan distance (4): piece proximity to both dukes
- Board control (9): approx move counts, reachable squares, contested, defended, threatened
- Duke mobility (2): per-player
- Discard vector (26): per-tile-type per-player (recently fixed: was always zero before discard bug fix)

## Data Formats

| Format | Magic | Description |
|---|---|---|
| DTRJ | — | Game trajectories (full GameStates + results) |
| LPOS | `LPOS` | Labeled positions (sparse encoding + single label + count) |
| FLPS | `FLPS` | Feature-labeled positions (sparse encoding + 41 labels + count) |
| GMLP | `GMLP` | GenericMlp weights (flat f32 vector) |
| HDA1 | `HDA1` | HalfDA sparse L1 weights |
| HDA2 | `HDA2` | HalfDA dense output weights |

## Data on Disk (D:/temp)

| Path | Contents |
|---|---|
| `ckpt_heuristic_1m/` | 200k heuristic game trajectories (main training data) |
| `ckpt_random_10m/` | 8M random game trajectories (25GB) |
| `halfda_d3_50k/` | 50k LR-Cheap d3 self-play games + scores |
| `combined_lr_41.json` | LR-Cheap evaluator weights (41 features) |
| `labeled_positions_v2.bin` | 8.3M positions labeled with LR-Cheap depth-2 |
| `labeled_all_features_norm.bin` | 8.3M positions with all 41 features, normalized |
| `duke_models.db` | SQLite model registry |

## Performance Characteristics

- Game engine (100 games d3 self-play): **~2 seconds** (with alpha-beta, apply/undo, duke caching)
- FC training: **~60k pos/sec** (256-wide), **~25k pos/sec** (1024-wide)
- CNN training: **~14k pos/sec** (Box kernel with im2col+sgemm)
- HalfDA training: **~25k pos/sec** (138M params, sparse L1 on CPU, dense on GPU)
- Position scoring at d3: **~1330 pos/sec** (parallel with rayon)

## Key Optimizations Applied

- **Alpha-beta pruning** in negamax (was missing! 6x speedup)
- **Apply/undo** instead of clone for guard checking AND negamax search
- **Duke position caching** (O(1) instead of O(36) scan)
- **Captures-first move ordering** (25% faster at depth 4)
- **im2col+sgemm** for CNN convolutions (5x CNN speedup)
- **Stack-allocated bags** (TileBag/DiscardBag as fixed arrays, no heap)
- **LTO + target-cpu=native** in release builds
- **Batched sgemm** for FC training (10x training speedup)

## Current Training Pipeline

1. **Generate games**: `generate_games` plays LR-Cheap d3 self-play, saves trajectories + scores
2. **Score positions**: `score_trajectories` labels each position with depth-3 negamax eval
3. **Train NNUE**: `halfda_train` trains 67k->2048->1 network with lambda mixing (eval + game outcome)
4. **Evaluate**: `elo_tournament` compares trained model vs base/LR-Cheap at various depths

## Rules for Agents

- **NEVER run training/scoring/game binaries** unless explicitly told to by the user
- Only `cargo test` and `cargo build` are safe to run freely
- Always commit after completing work
- Always add regression tests for bug fixes
- Use subagents for code changes
- Copy binaries aside before long runs so cargo can rebuild
- Never run two training processes simultaneously
