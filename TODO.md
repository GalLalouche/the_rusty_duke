# TODO

## Training
- [ ] Longer Combined-NN training (1147→128→32 was at 65.6% vs Base and still climbing)
- [ ] Curriculum learning (adaptive opponent difficulty during training)
- [ ] Try larger population size for bigger networks
- [ ] Elo round-robin tournament across all models

## Infrastructure
- [ ] Live dashboard (replace plain text live_status.txt with something richer)
- [ ] Proper storage of models (organized model registry instead of scattered D:/temp checkpoints)

## Performance
- [ ] Profile training iteration to find actual bottleneck (instead of guessing)
- [ ] NNUE incremental accumulator (update L1 activations on make/unmake instead of full recompute)
- [ ] Const-generic specialization for common NNUE layer sizes (enables LLVM to emit AVX2)
- [ ] Explore single-allocation contiguous weight buffer for NnueWeights (currently 6 separate Vecs)
- [ ] Engine: evaluate returning iterators instead of Vec from move generation methods
- [ ] Engine: SmallVec or counter for idle_move_count stack (avoid heap alloc on GameState clone)

## Testing
- [ ] Additional test coverage from review (lower priority):
  - trajectory_io: roundtrip bags/discards/idle_move_count, error paths (bad magic/version)
  - feature_cache: writer assertion test, accumulate_from_cache target values
  - encoding: active_board_features bounds/count invariants, FeatureBuffer capacity
  - learned_heuristic: RegressionAccumulator save/load roundtrip, solve_with_lambda regularization, board_control_features expected values
  - match_runner: max_turns timeout path, asymmetric side-swap attribution
  - game_setup: greedy_move correctness (picks best move), MAX_TURNS forced draw, epsilon=1.0 fully random
  - weight_export: transpose helper unit test, non-default layer sizes

## UI
- [ ] Integrate trained NN models into existing TUI (play human vs trained model)
