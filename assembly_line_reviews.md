# Assembly Line Reviews Log

## Overview
4 review focuses requested. 2 completed in worktrees, 2 hit rate limit bug (#40273).

## Reviewer 1: Test Coverage + Assertions
- **Status**: Completed
- **Findings**: 4 bugs found in pre-existing code, 57 tests added
- **Bugs found**:
  1. `assert_not!` macro lost negation -- `assert!($b)` instead of `assert!(!$b)`
  2. `to_absolute_coordinate` used `src.x` for y-offset calculation
  3. `can_apply` rejected moves to empty squares (`map_or(false,...)` should be `map_or(true,...)`)
  4. `diff_or_zero` had inverted branches causing usize underflow
- **Outcome**: All 4 bugs exist in pre-existing code from before our refactors. Our current HEAD has either fixed them or refactored the code away entirely. No changes needed.
- **Tests added**: 57 tests across board, utils, offset, token modules (in worktree only, not merged)

## Reviewer 2: Software Design
- **Status**: Hit rate limit bug, did not execute
- **Action**: Skipped

## Reviewer 3: Performance
- **Status**: Hit rate limit bug, did not execute
- **Action**: Skipped

## Reviewer 4: Bugs + Correctness
- **Status**: Completed
- **Findings**: Same 4 bugs as Reviewer 1 (independent confirmation)
- **Outcome**: Same -- bugs don't exist in current HEAD
- **Action**: No changes needed

## Actions NOT Taken
- Did not merge worktree branches -- they branched from an old commit and had massive divergence from current HEAD (80+ file conflicts)
- Did not port the 57 new tests -- they test pre-existing code that was refactored, so the tests would need rewriting for our current codebase
- Did not run software design or performance reviews due to rate limit bug
- Did not re-run reviews on current HEAD after worktree failure

## Lessons Learned
- Worktree isolation branches from whatever commit is HEAD at launch time. If the main branch has uncommitted changes or is actively modified, the worktree diverges immediately.
- Running `git add -A` from a worktree directory and committing to the main branch is catastrophic -- it stages the worktree's file set, not the main repo's.
