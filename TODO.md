# TODO

## Training
- [ ] Train Combined-NN against LR-Cheap (started, crashed at iter 832, needs resuming)
- [ ] Longer Combined-NN training (1147→128→32 was at 65.6% vs Base and still climbing)
- [ ] Curriculum learning (adaptive opponent difficulty during training)
- [ ] Try larger population size for bigger networks

## Infrastructure
- [ ] Live dashboard (replace plain text live_status.txt with something richer)
- [x] Proper storage of models (organized model registry instead of scattered D:/temp checkpoints)
- [x] Elo round-robin tournament across all models
- [x] --benchmark flag for configurable eval opponent during training
- [x] --profile flag for per-phase training timing

## Performance
- [x] Profile training iteration (96% game playing, 84-90% evaluation, 92% forward pass within eval)
- [x] Incremental L1 accumulator (17% faster training — reuses L1 across candidate moves)
- [x] Cache-friendly matmul loop order (~8% improvement)
- [x] SmallRng for perturbation generation
- [x] Redundant duke mobility elimination in feature extraction
- [ ] Int8 quantization with true integer SIMD (current i8→f32 cast shows no speedup; need explicit AVX2 intrinsics for i8×i8→i16)
- [ ] Const-generic specialization for common layer sizes (enables LLVM to fully unroll)
- [ ] Engine: iterators instead of Vec from move generation methods
- [ ] Engine: SmallVec or counter for idle_move_count stack

## Testing
- [x] Additional test coverage from review (20 tests added across all categories)

## UI
- [ ] Integrate trained NN models into existing TUI (play human vs trained model)

## Current Rankings (Elo tournament, 1000 games/matchup)
1. LR-All (65 features) — Elo 1656, 64.6% vs Base
2. LR-Cheap (41 features) — Elo 1595, 48.4% vs Base
3. Base heuristic — Elo 1566
4. Combined-NN (1147→128→32) — Elo 1555, 62.6% vs Base
5. LR-Guard (24 features) — Elo 1549, 46.7% vs Base
6. Cheap NN (41→32→8) — Elo 1504, 30.8% vs Base
7. Plain-NN (64→64→32) — Elo 1326, 23.4% vs Base
8. Random — Elo 1248
