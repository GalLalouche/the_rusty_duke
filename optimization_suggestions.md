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
| **Opt 10: Occupancy bitboards** | **~30%** | **~41%** | **~6.2 us/move** |
| **Opt 11: Incremental heuristic eval** | **~25%** | **~55%** | **~4.7 us/move** |

---

## Optimization 10: Occupancy bitboards

**Problem:** Board operations like `is_occupied`, `unobstructed`, `different_team_or_empty`,
`is_guard`, and `heuristic_counts_both_players` scan the 36-cell board array
repeatedly, checking `Option<PlacedTile>` per cell. Obstruction checks for
Move/Slide iterate intermediate squares one by one.

**Fix:** Added three `u64` bitboards (`occ`, `top_occ`, `bot_occ`) to `GameBoard`,
maintained through all mutation paths (place, remove, make_a_move, undo,
guard-check functions). Key accelerations:

- `unobstructed(src, dst)` → `RAY_BETWEEN[src][dst] & occ == 0` (single AND+compare
  replacing a loop over intermediate squares)
- `different_team_or_empty` → single bit check against owner bitboard
- `is_guard` → iterate only enemy pieces via bit extraction (`trailing_zeros` loop)
- `heuristic_counts_both_players` → per-owner bit iteration, `count_ones()` for tile counts
- `is_valid_placement_space` → bitboard occupancy check
- `has_valid_moves` → bitboard piece iteration

Also precomputed `RAY_BETWEEN[36][36]` — a const lookup table of u64 ray masks
for all straight-line pairs on the 6×6 board (~10KB, fits in L1 cache).

**Expected impact:** High. Replaces O(n) loops with O(1) bit operations.

**Status:** Done — **~30% improvement** (10.5 → ~6.2 us/move cumulative).

---

## Optimization 11: Incremental heuristic evaluation

**Problem:** `greedy_move` evaluates ~20 candidate moves per turn. Each evaluation
calls `heuristic_counts_both_players` which iterates ALL ~10 pieces, computing
legal move counts for each. But only 1-2 pieces change per candidate move —
the remaining ~8-9 pieces have (approximately) the same move counts.

**Fix:** Added `greedy_move_heuristic_incremental` which precomputes per-piece
move counts once before the candidate loop, then for each candidate:
1. Only recomputes the moved piece's count at its new position
2. Subtracts captured piece's count if applicable
3. Uses cached counts for all other pieces

This reduces per-candidate evaluation from O(total_pieces) to O(1), trading
exact accuracy for speed (ignoring blocking/unblocking side effects on
other pieces' slide paths, which is a small approximation error for an
already-approximate heuristic).

**Expected impact:** Very high. Reduces evaluation cost by ~5-10x.

**Status:** Done — **~25% improvement** on top of bitboards (10.5 → ~4.7 us/move cumulative, **2.2x total speedup**).

---

## Optimization 7 (revisited): Precomputed move bitmasks

**Problem:** `count_legal_moves_no_guard` recomputes target squares every call:
`to_absolute_coordinate` arithmetic, action-type dispatch, bounds checking.
These are pure functions of (tile_type, owner, side, position) and don't
change between calls for the same configuration.

**Fix:** Added a `MoveTable` (lazy-initialized `OnceLock`) that precomputes,
for every (tile_type × 2 owners × 2 sides × 36 positions = 1872 entries):
- `jump_mask: u64` — Jump targets
- `near_move_mask: u64` — distance-1 Move targets
- `strike_mask: u64` — Strike targets (count via popcount after masking with enemy occ)
- `far_move[8]` — Move targets at distance ≥ 2 (need obstruction check)
- `slide[20]` — Slide/JumpSlide targets (need per-target obstruction check)

At runtime, `count_legal_moves_no_guard` becomes: one table lookup + two popcounts
+ a few array iterations for Slide/far-Move targets.

**Expected impact:** Moderate-high. Eliminates ~5 function calls per action.

**Status:** Done — **~30% improvement** on top of Opts 10-11 (10.5 → ~3.2 us/move cumulative, **3.3x total speedup**).

---

## Optimization 12: Move-table driven generation and guard checks

**Problem:** Even with precomputed move counts, hot paths still rebuilt legal move
targets and attack/reach checks on every call:
- `get_legal_moves_no_guard` iterated action lists and called `target_coordinates`
- `can_attack_square` (inside `is_guard`) iterated actions and recomputed offsets
- `get_reachable_squares_ignoring_friendly` repeated the same target expansion

These methods run frequently during greedy move scoring and guard filtering.

**Fix:** Reused `MoveTable` directly in runtime hot paths:
1. Rewrote `get_legal_moves_no_guard` to emit moves from precomputed masks/arrays
   (bit iteration + obstruction checks only where needed).
2. Rewrote `can_attack_square` to test target membership in precomputed masks and
   only do ray/jumpslide obstruction checks for matching far targets.
3. Rewrote `can_reach_square_ignoring_friendly` and
   `get_reachable_squares_ignoring_friendly` similarly.
4. Added `jump_slide_clear` helper for shared, branch-light jump-slide checking.

**Expected impact:** High on guard-heavy move generation and heuristic labeling.

**Status:** Done — **~28% improvement** on top of Opt 7-revisited
(~3.2 → ~2.3 us/move), **~4.6x total speedup** vs baseline (10.5 → ~2.3 us/move).

---

## Future Opportunities (not yet implemented)

1. **Incremental attack map for `is_guard`**: Maintain a per-square attack
   count that's updated incrementally on make/undo, making `is_guard` O(1)
   instead of O(enemy_pieces × actions). Would significantly speed up move
   generation with guard checking.

2. **Transposition table for negamax**: Cache evaluations for positions seen
   during search. Requires a fast hash (already improved by Opt 4) and would
   significantly prune repeated positions in the search tree.

3. **SIMD-accelerated NNUE forward pass**: Use SIMD intrinsics for the
   matrix-vector multiplications in the neural network forward pass.
