# Speed Optimization Suggestions

## Hot Paths

The main hot paths for training and AI game-playing are:

1. **`greedy_move`** — single-ply move selection (called every turn in selfplay/training).
2. **`greedy_move_incremental`** — same, with L1 accumulator for NNUE models.
3. **`negamax_ab`** — depth-N alpha-beta search (used by `greedy_move_deep`).
4. **`play_selfplay_game` / `play_two_player_game`** — outer game loop.

## Optimization 1: `greedy_move` — make/undo instead of GameState clone

**Problem:** `greedy_move` clones the entire `GameState` (~740 bytes, dominated by
the 512-byte `idle_stack`) for every candidate move (~20 per turn). This is pure
waste since `greedy_move_deep_with_score` already uses the make/undo pattern.

**Fix:** Replace `gs.clone()` + `mv.play(&mut clone)` + `evaluator.evaluate(&clone)`
with `mv.play(gs)` + `evaluator.evaluate(gs)` + `gs.undo(undo)`.

**Expected impact:** High. Eliminates ~20 × 740-byte memcpy per turn.

**Status:** Done — ~neutral for heuristic eval (evaluation dominates), but architecturally sound and benefits NNUE/deep search.

---

## Optimization 2: `greedy_move_incremental` — make/undo instead of clone

**Problem:** Same as Opt 1, but for the NNUE incremental accumulator path. The
function clones `GameState` for each candidate to extract post-move features,
despite the L1 accumulator being designed for incremental updates.

**Fix:** Use make/undo and extract features from the mutated `gs` directly.
The base features (old_board, old_bag, old_combined) are extracted once before
the loop and remain valid since each undo restores the state.

**Expected impact:** High. Same clone savings plus this is the primary NNUE path.

**Status:** Done — not benchmarked directly (NNUE path), but same architectural improvement.

---

## Optimization 3: `negamax_ab` — stack-allocated move buffer

**Problem:** `negamax_ab` allocates `Vec<PossibleMove>` at every interior node via
`gs.all_valid_game_moves_for_current_player().collect()`. At depth 4 with
branching factor ~20, this is ~8000+ heap allocations per root move.

**Fix:** Use a stack-allocated buffer. Max legal moves per position is bounded:
18 tiles × ~12 actions max + 4 placement offsets ≈ ~220 max. A fixed-size
`[PossibleMove; 256]` buffer + length counter avoids all heap allocation.

**Expected impact:** Medium. Vec allocation is O(1) amortized but the allocator
overhead adds up at thousands of nodes. Also improves cache locality.

**Status:** Cancelled — Vec alloc for ~10 items is negligible vs evaluation cost.

---

## Optimization 4: `GameState::Hash` — stack-allocated bag sort

**Problem:** The `Hash` impl for `GameState` allocates two `Vec<u8>` (one per
player's bag), sorts them, then hashes. Bag contents are at most 12 tiles.

**Fix:** Use `[u8; 12]` stack arrays instead of `Vec<u8>`. Sort in-place.

**Expected impact:** Low-medium. Only matters if `GameState` is hashed frequently
(e.g., transposition tables). Still worth fixing as a low-effort win.

**Status:** Done — eliminates two heap allocations per hash call.

---

## Optimization 5: Avoid `AiMove` intermediary in `greedy_move`

**Problem:** `greedy_move` calls `AiMove::all_moves(gs)` which maps each
`PossibleMove` into an `AiMove`, cloning the `capturing: Option<PlacedTile>`
field. Then for each move, `to_undo_move()` clones it again back to `PossibleMove`.

**Fix:** Work with `PossibleMove` directly — it already has the `capturing` info
needed for undo. Skip the AiMove conversion entirely.

**Expected impact:** Low. Saves a few small allocations and clones per turn.

**Status:** Cancelled — `PlacedTile` is `Copy`, conversion is zero-cost.

---

## Optimization 6: Reduce `idle_stack` impact on clone cost

**Problem:** The `idle_stack: [u8; 512]` is the largest field in `GameState`
(512 of ~740 bytes). Every `clone()` copies all 512 bytes even though
`idle_stack_len` is typically < 50 in practice.

**Fix:** Reduced `IDLE_STACK_CAP` from 512 to 128. Analysis shows max possible
stack depth is ~51 (26 placements + 25 captures), plus search depth headroom.
128 is more than sufficient. Reduces `GameState` from ~740 to ~356 bytes (~52%).

**Expected impact:** Low-medium. Better cache locality, less memory for stored
trajectories. Marginal in `bench_greedy` since per-candidate cloning was
already eliminated by Opt 1.

**Status:** Done — marginal in bench, structural benefit.

---

## Optimization 9: Inline guard check in greedy_move

**Problem:** `greedy_move` generates moves via `all_valid_moves()`, which does a
lightweight make/undo for each candidate to check guard. Then the greedy loop
does another full make/undo per candidate for evaluation. Each candidate move
is effectively made twice.

**Fix:** Generate tile action moves WITHOUT guard checking (via
`all_valid_moves_ignoring_guard`), then check guard inline after `make_a_move`
in the evaluation loop. Placements still use guard-checked generation (since
`pull_tile_from_bag` hard-asserts validity). This combines two make/undo cycles
into one for tile action moves (~85% of candidates).

**Expected impact:** Medium. Halves the make/undo count for tile action moves.

**Status:** Done — ~3-5% improvement (marginal; lightweight guard make/undo was already cheap)

---

## Optimization 8: Combine heuristic evaluation into single pass

**Problem:** `StaticHeuristicEvaluator::evaluate` calls `count_moves_with_duke_ignoring_guard`
twice (once per player), each doing a full board iteration. The duke move count
is already included in the total move count, duplicating work.

**Fix:** Add `heuristic_counts_both_players` that iterates the board once,
collecting all 6 values (total_moves, duke_moves, tile_count for each player).

**Expected impact:** Medium. Halves the board iteration cost in heuristic eval.

**Status:** Done — ~10% improvement (cumulative with Opts 1,2,4)

---

## Results

Benchmark: `bench_greedy` — 2000 seeded games, StaticHeuristicEvaluator, release mode.

| Optimization | Individual Δ | Cumulative | Benchmark (median us/move) |
|---|---|---|---|
| Baseline | — | — | ~10.5 us/move |
| Opt 1: greedy make/undo | ~neutral | ~neutral | ~10.5 (heuristic eval dominates) |
| Opt 2: greedy_incremental make/undo | N/A (NNUE path) | — | not benchmarked here |
| Opt 3: negamax stack moves | cancelled | — | — |
| Opt 4: Hash stack sort | marginal | — | (no hashing in bench) |
| Opt 5: No AiMove intermediary | cancelled (PlacedTile is Copy, zero overhead) | — | — |
| Opt 6: Smaller clone (512→128) | marginal | — | ~9.2 (cache/structural) |
| Opt 8: Single-pass heuristic eval | ~10% | ~10% | ~9.4 us/move |
| Opt 9: Inline guard check | ~3-5% | ~12% | ~9.2 us/move |

## Future Opportunities (not yet implemented)

1. **Incremental attack map for `is_guard`**: Maintain a per-square attack
   count that's updated incrementally on make/undo, making `is_guard` O(1)
   instead of O(enemy_pieces × actions). Would significantly speed up move
   generation with guard checking.

2. **Bitboard representation**: Use u64 bitboards for 6×6 occupancy (36 bits),
   per-owner masks, and per-tile-type masks. Would enable fast piece enumeration,
   attack detection, and move generation via bit manipulation.

3. **Precomputed action tables**: For each tile type/side, precompute absolute
   offsets relative to each board position, stored as lookup tables. Avoids
   repeated `to_absolute_coordinate` arithmetic at runtime.

4. **Transposition table for negamax**: Cache evaluations for positions seen
   during search. Requires a fast hash (already improved by Opt 4) and would
   significantly prune repeated positions in the search tree.

5. **SIMD-accelerated NNUE forward pass**: Use SIMD intrinsics for the
   matrix-vector multiplications in the neural network forward pass.
