# Assembly Line Reviews

**Date:** 2026-03-30
**Baseline tests:** duke_rust: 210 | duke_training lib: 188 | supervised_train bin: 7 | **Total: 405**

## Review 1: Test Coverage
*Status: complete*

**24 new tests added:**
- `bag.rs` (6): remove_specific, DiscardBag::remove — previously untested
- `board.rs` (6): can_reach_square_ignoring_friendly — the defended-fix function had no direct tests
- `tests.rs` (12): encoding plane indices, bag feature counting, no duplicate features, flat encoding zeroing, forward_sparse vs forward_f32 consistency, sigmoid range, param_count, save/load roundtrip, L1Accumulator consistency, negamax terminal/depth-0/depth-1

**No bugs found.** Heuristic features audited — no issues similar to the defended bug.

**Tests after: 430**

## Review 2: Performance Optimizations
*Status: complete*

**6 issues fixed:**
1. CNN per-batch gradient allocation (400KB/batch) — hoisted before loop
2. CNN per-batch target Vec allocation — pre-allocated, reused
3. FC per-batch position ref Vec allocation — pre-allocated, reused
4. CNN repeated offset Vec allocations (~512/batch) — cached in CnnLayout struct
5. CNN Adam powf → running multiply + 3-pass split
6. CNN apply_diamond_mask — precomputed const non-diamond positions

**Est. impact:** CNN +15-30%, FC +2-5%. No test count change.

**Tests after: 430**

## Review 3: Software Design
*Status: complete*

**Extracted `supervised_common.rs` module:**
- LabeledPosition struct (was in 3 files)
- load_lpos with LPOS+FLPS support (was in 3 files)
- AdamState optimizer (was in 2 files)
- label_to_target + LABEL_CLAMP (was in 3 files)
- COMBINED_FEATURE_NAMES (was in 3 files)
- build_weighted_indices (was in 2 files)
- run_benchmark evaluation loop (was in 3 files)

**Net: -370 lines of duplication.**

**5 recommendations for future refactoring (not done — would exceed 500 lines each):**
1. **Split cnn.rs (4458 lines)** into cnn/model.rs, cnn/kernels.rs, cnn/batch_ops.rs, cnn/evaluator.rs
2. **Move FcScratch + batch_forward/backward** from supervised_train.rs into lib for reuse by other FC trainers
3. **Unify GenericMlp and CnnModel serialization** via a WeightStore trait (overlapping save/load/param_count)
4. **Extract adaptive LR into LrScheduler** struct in supervised_common (duplicated between FC and CNN trainers)
5. **Add model.as_evaluator() trait method** to eliminate clone-based evaluate_model wrappers in each binary

**Tests after: 430**

## Review 4: Bugs and Correctness
*Status: complete*

**1 bug found and fixed:** Residual gradient leaking through dead ReLU neurons in `batch_backward()`. The skip connection gradient was added AFTER the ReLU mask was applied, causing dead neurons (pre_act <= 0) to receive spurious gradient updates. Fixed by moving residual gradient addition before ReLU mask. Regression test added with numerical gradient verification (analytical vs numerical within 5%).

**Impact:** FC experiments using --residual had corrupted gradients. The 256×8 res2 overnight result (3.0e-4) is likely affected — needs re-run with fixed binary.

**Tests after: duke_rust: 222 | duke_training: 201 | supervised_train: 8 | Total: 431**

## Final Summary

- 4 reviews completed
- 26 tests added (405 → 431)
- 1 bug fixed (residual gradient ordering)
- 6 performance optimizations
- 370 lines of duplication eliminated
- 5 design recommendations flagged for future
