# Assembly Line Reviews (Round 2)

**Date:** 2026-03-31
**Baseline tests:** duke_rust: 228 | duke_training lib: 212 | supervised_train bin: 8 | **Total: 448**

## Review 1: Test Coverage, Assertions, Robustness
*Status: complete*

18 tests added, 3 debug_asserts. Hash/PartialEq contract violation fixed separately.

**Tests after: 458**

## Review 2: Software Design
*Status: complete*

Extracted AdaptiveLrScheduler, game_outcome_target, APPENDED_INPUT_SIZE to shared modules. Replaced magic numbers with named constants across 8 files. 12 files, +180/-155.

**Tests after: 458**

## Review 3: Performance
*Status: complete*

**2x game engine speedup** (500 games d2: 2.5s → 1.3s):
- Box<dyn Iterator> → stack-allocated LegalMoveBuffer in move generation
- Vec allocations → stack arrays in tile coordinate collection
- count_tiles_for_owner / count_legal_moves_ignoring_guard avoid Vec when only count needed
- tile_action_does_not_put_in_guard avoids redundant can_apply re-validation
- time_it_macro → no-op in release (was SystemTime::now + HashMap per is_guard!)
- #[inline] on ~30 hot-path functions
- active_coordinates iterates backing array directly
- distance_to uses abs_diff

**Tests after: 458**

## Review 4: Bugs and Correctness
*Status: complete*

**2 bugs found and fixed:**

1. **Hash missing idle_move_count**: `GameState::Hash` omitted `moves_without_capture_or_placement_stack`, so states near a tie draw collided with states far from it. Added to Hash impl. Regression test: `hash_differs_when_idle_move_count_differs`.

2. **TileBag::remove_specific O(n) instead of O(1)**: Used `Vec::remove` (shifts elements) instead of `swap_remove`. Fixed. Regression test: `remove_specific_swap_remove_preserves_other_tiles`.

Thorough review of apply/undo symmetry, move generation, guard checking, training backward pass, CNN im2col, HalfDA encoding, Adam optimizer, negamax/expectimax — no further bugs found.

**Tests after: duke_rust: 236 | duke_training: 224 | Total: 460**

## Final Summary

- 4 reviews completed
- 22 tests added (448 → 460)
- 3 bugs fixed (Hash missing idle count, TileBag O(n) remove, Hash/PartialEq contract)
- 2x game engine speedup
- Magic values replaced with constants across 8 files
- Adaptive LR scheduler extracted to shared module
