# Experiment 1: Tiny NNUE ES (32x8)
- Architecture: 1106->32->8->1
- Parameters: 35,697
- ES config: pop=100, games=10, sigma=0.02, lr=0.02
- Duration: 60.0 minutes (all 450 iterations completed)
- Eval games: 500 per checkpoint (vs heuristic opponent)

| Iter | Win% | Loss% | Tie% |
|------|------|-------|------|
| 0 (init) | 5.4 | 88.8 | 5.8 |
| 25 | 7.2 | 85.6 | 7.2 |
| 50 | 8.2 | 84.2 | 7.6 |
| 75 | 10.8 | 82.4 | 6.8 |
| 100 | 17.6 | 76.6 | 5.8 |
| 125 | 20.4 | 74.4 | 5.2 |
| 150 | 15.0 | 78.6 | 6.4 |
| 175 | 16.8 | 77.6 | 5.6 |
| 200 | 21.2 | 73.8 | 5.0 |
| 225 | 21.6 | 74.4 | 4.0 |
| 250 | 19.2 | 75.6 | 5.2 |
| 275 | 25.2 | 72.0 | 2.8 |
| 300 | 23.6 | 71.8 | 4.6 |
| 325 | 24.4 | 69.6 | 6.0 |
| 350 | 27.0 | 68.2 | 4.8 |
| 375 | 27.8 | 65.6 | 6.6 |
| 400 | 25.0 | 71.0 | 4.0 |
| 425 | 24.0 | 70.6 | 5.4 |
| 450 (final) | 26.8 | 68.0 | 5.2 |

Best: 27.8% win rate at iter 375

## Observations
- Clear upward trend from 5.4% to ~25-28% over 450 iterations (5x improvement in win rate).
- Three distinct phases: rapid climb iters 1-125 (5.4% -> 20.4%), noisy plateau iters 125-275 (15-25%), and second climb iters 275-375 (25% -> 27.8%).
- Avg perturbation win rates (avg_wr+/avg_wr-) climbed from ~0.09 to ~0.30 over the run, confirming the population is improving.
- Max single-perturbation win rates reached 0.80-0.85 by the end.
- Heuristic loss rate dropped from 88.8% to 65.6-68%, a 23 percentage point improvement.
- The 32x8 architecture (36k params) is trainable by ES with pop=100. Still improving at iter 450, suggesting more training time could yield further gains.
- Checkpoints saved to D:/temp/es_tiny_32x8/ (best checkpoint: es_iter_375.nnue, final: es_final.nnue).

# Experiment 2: Layer-wise ES (512x64, last layer only)
- Parameters optimized: 65 (l3_weight + l3_bias)
- Frozen layers from pre-trained ES checkpoint (17.2% baseline)
- ES config: pop=100, games=20, sigma=0.05, lr=0.05
- Duration: ~55 min (225+ iterations)

| Iter | Win% | Loss% | Tie% |
|------|------|-------|------|
| INIT | 17.2 | 75.8 | 7.0 |
| 25 | 18.8 | 73.4 | 7.8 |
| 50 | 18.4 | 72.6 | 9.0 |
| 75 | 18.8 | 74.2 | 7.0 |
| 100 | 20.6 | 72.2 | 7.2 |
| 125 | 24.4 | 71.0 | 4.6 |
| 150 | 24.4 | 67.8 | 7.8 |
| 175 | 25.4 | 67.0 | 7.6 |
| 200 | 24.6 | 69.4 | 6.0 |
| 225 | 26.4 | 67.4 | 6.2 |

Best: 26.4% at iter 225. Still climbing, no plateau.

# Experiment 3: Hybrid TD pre-train + ES fine-tune (512x64)
- Architecture: 1106->512->64->1
- Parameters: 599,681
- TD pre-training: 2 epochs on 200k heuristic game trajectories (10.2M states)
- TD pre-train result: 8.6% win rate vs heuristic (500 eval games)
- TD training time: 2362s (~39.4 min), final avg loss: 0.004032
- ES fine-tune config: pop=200, games=10, sigma=0.01, lr=0.01, 120 iterations
- ES fine-tune time: 1489s (~24.8 min)
- Total duration: ~64 minutes (39.4 min TD + 24.8 min ES)
- Eval games: 500 per checkpoint (vs heuristic opponent)

| Iter | Win% | Loss% | Tie% |
|------|------|-------|------|
| 0 (TD init) | 8.6 | 85.0 | 6.4 |
| 10 | 10.2 | 80.0 | 9.8 |
| 20 | 10.8 | 81.0 | 8.2 |
| 30 | 8.4 | 85.8 | 5.8 |
| 40 | 12.0 | 77.0 | 11.0 |
| 50 | 13.4 | 78.0 | 8.6 |
| 60 | 10.6 | 82.0 | 7.4 |
| 70 | 13.8 | 79.0 | 7.2 |
| 80 | 15.2 | 74.2 | 10.6 |
| 90 | 16.8 | 74.0 | 9.2 |
| 100 | 18.8 | 72.8 | 8.4 |
| 110 | 21.6 | 73.0 | 5.4 |
| 120 (final) | 18.0 | 76.0 | 6.0 |

Best: 21.6% win rate at iter 110

## Observations
- TD pre-training on heuristic trajectories gave a starting point of 8.6% win rate (vs 5.4% random init in Experiment 1).
- ES fine-tuning improved the model from 8.6% to a peak of 21.6% over 120 iterations.
- The 512x64 architecture (600k params) is much larger than the 32x8 (36k params), yet ES made steady progress.
- Noisy but clear upward trend: 8.6% -> 10% (iters 1-20) -> 12-14% (iters 40-70) -> 15-19% (iters 80-100) -> 21.6% peak at iter 110.
- The final eval (iter 120) dipped to 18.0%, suggesting eval noise of ~3-4% at 500 games.
- Compared to Experiment 1 (32x8, 60 min): at comparable wall-clock ES time (~25 min), this 512x64 model reached ~21.6% vs the 32x8's ~20.4% at 25 min (iter 125). The larger model converges similarly fast per iteration but each iteration is more expensive.
- The TD pre-training head start (8.6% vs 5.4% random) helps but doesn't dramatically accelerate ES convergence. The large parameter space (600k vs 36k) means ES gradient estimates are noisier.
- Checkpoints saved to D:/temp/hybrid_512x64_td/ (TD) and D:/temp/hybrid_512x64_es/ (ES, best: es_iter_110.nnue, final: es_final.nnue).

# Overall Comparison: All approaches vs Heuristic

| Method | Win% | Notes |
|--------|------|-------|
| Random | 4.8% | baseline |
| NNUE TD 1ep (512x64) | 8.5% | supervised on heuristic games |
| NNUE TD 5ep (512x64) | 9.5% | diminishing returns |
| NNUE RL from scratch | 6.0% | outcome-labeled, failed |
| NNUE ES full (512x64) | 17.2% | 600k params, peaked |
| NNUE ES hybrid (TD+ES 512x64) | 21.6% | TD head-start helps |
| NNUE ES last-layer (65p) | 26.4% | still climbing |
| **NNUE ES tiny (32x8)** | **27.8%** | **best NNUE, from scratch** |
| LR 5-param (regression) | 39.1% | closed-form, instant |
| LR elastic net (15p) | 40.6% | best with regularization |
| Equal-weight heuristic | ~39%* | the opponent |

*Equal-weight heuristic shows ~39% win / ~31% loss / ~30% tie head-to-head due to top/bottom asymmetry.

# Experiment 4: Combined Features NNUE (41 inputs, 32x8)
- Architecture: 41->32->8->1 (hand-rolled MLP, not NNUE infrastructure)
- Input features: 41 combined features from extract_combined_features()
  - Manhattan distance proximity (4): units near my/enemy duke
  - Board control (9): approx moves, reachable squares, contested, defended/threatened
  - Duke mobility (2): my/opp duke movement options
  - Discard vector (26): per-tile-type discard counts for both players
- Parameters: 1,617
- ES config: pop=100, games=10, sigma=0.02, lr=0.02
- Duration: ~22 min (275 iterations, then process killed due to stdout buffering)
- Eval games: 500 per checkpoint (vs heuristic opponent)

| Iter | Win% | Loss% | Tie% |
|------|------|-------|------|
| 0 (init) | 3.0 | 87.4 | 9.6 |
| 25 | 43.6 | 28.2 | 28.2 |
| 50 | 57.6 | 22.4 | 20.0 |
| 75 | 52.4 | 25.0 | 22.6 |
| 100 | 54.2 | 23.8 | 22.0 |
| 125 | 49.2 | 27.0 | 23.8 |
| 150 | 53.6 | 26.0 | 20.4 |
| 175 | 54.2 | 26.8 | 19.0 |
| 200 | 48.6 | 28.6 | 22.8 |
| 225 | 56.0 | 25.8 | 18.2 |
| 250 | 57.6 | 26.8 | 15.6 |
| 275 | 55.2 | 22.6 | 22.2 |

Best: 57.6% win rate at iters 50 and 250

## Observations
- Explosive initial learning: 3.0% -> 43.6% in just 25 iterations (~2 min). This is far faster than any NNUE experiment.
- Peaked at ~57.6% by iter 50, then oscillated in the 49-58% range through iter 275. The model appears to have plateaued.
- With only 1,617 parameters (vs 35,697 for the 32x8 NNUE), ES gradient estimates are much cleaner.
- The 41 combined features carry strong signal: manhattan distance to dukes, board control metrics, and discard vectors give the network rich information without needing to learn sparse board encoding.
- Compared to the best NNUE result (27.8% at 32x8 after 450 iters / 60 min), the combined-features network reached 57.6% in 50 iters / ~4 min. That's a 2x win rate improvement in 1/15th the time.
- The combined-features network actually beats the heuristic opponent (>50% win rate), making it the first ES-trained model to do so.
- Tie rates are notably higher (15-28%) compared to NNUE experiments (4-7%), suggesting the combined-features evaluator produces more cautious/defensive play.
- The plateau around 50-58% suggests the 41 features and 32x8 architecture may be near their capacity ceiling for this opponent.
- Checkpoints saved to D:/temp/es_combined_41/ (best: es_combined_iter_250.cnet).

# Overall Comparison: All approaches vs Heuristic

| Method | Win% | Notes |
|--------|------|-------|
| Random | 4.8% | baseline |
| NNUE TD 1ep (512x64) | 8.5% | supervised on heuristic games |
| NNUE TD 5ep (512x64) | 9.5% | diminishing returns |
| NNUE RL from scratch | 6.0% | outcome-labeled, failed |
| NNUE ES full (512x64) | 17.2% | 600k params, peaked |
| NNUE ES hybrid (TD+ES 512x64) | 21.6% | TD head-start helps |
| NNUE ES last-layer (65p) | 26.4% | still climbing |
| NNUE ES tiny (32x8) | 27.8% | best NNUE, from scratch |
| LR 5-param (regression) | 39.1% | closed-form, instant |
| LR elastic net (15p) | 40.6% | best with regularization |
| Equal-weight heuristic | ~39%* | the opponent |
| Combined41 ES (32x8) | 57.6% | beats heuristic, 1.6k params |
| **Appended1147 ES (128x32)** | **65.6%** | **best overall, still climbing** |

*Equal-weight heuristic shows ~39% win / ~31% loss / ~30% tie head-to-head due to top/bottom asymmetry.

# Experiment 5: Appended Features NNUE (1147 inputs = 1106 board + 41 combined, 128x32)
- Architecture: 1147->128->32->1 (hand-rolled MLP with ReLU hidden layers, sigmoid output)
- Input features: 1106 sparse board encoding + 41 combined features appended
- Parameters: 151,105
- ES config: pop=200, games=10, sigma=0.02, lr=0.02
- Duration: ~28 min (161 iterations logged, killed at 30 min; evals through iter 150)
- Eval games: 500 per checkpoint (vs heuristic opponent)
- Iteration time: ~10s/iter (4000 games per iteration)

| Iter | Win% | Loss% | Tie% |
|------|------|-------|------|
| 0 (init) | 6.4 | 89.4 | 4.2 |
| 25 | 48.8 | 23.4 | 27.8 |
| 50 | 53.8 | 30.2 | 16.0 |
| 75 | 56.6 | 24.4 | 19.0 |
| 100 | 53.8 | 26.0 | 20.2 |
| 125 | 59.6 | 22.8 | 17.6 |
| 150 | 65.6 | 21.8 | 12.6 |

Best: **65.6%** at iter 150, still climbing steeply. No plateau yet.

Training win rates at termination (iter 161): avg_wr+ ~0.667, avg_wr- ~0.668

## Observations
- Explosive early learning: 6.4% -> 48.8% in 25 iterations (~4 min), faster than any previous experiment.
- The appended version surpasses the 41-only network (57.6%) by iter 125 and keeps climbing.
- The raw board features provide fine-grained tactical information that the aggregated 41 features miss.
- Despite 151k params (vs 1.6k for Combined41), ES makes steady progress because the 41 features bootstrap the learning.
- Tie rate drops from 28% to 13% as the model improves, indicating more decisive/confident play.
- Training avg win rates reached ~67% by iter 161, suggesting next eval (iter 175) would likely show further improvement.
- Still improving at termination -- more training time would likely push higher.
- Checkpoints saved to D:/temp/es_appended_1147/ (best: es_appended_iter_150.bin at 65.6%).

# Key Takeaways

1. **Appended features (1106+41) is the best approach** — 65.6% win rate at iter 150, still climbing
2. **Combined features bootstrap learning** — the 41 features give the network a head start, the 1106 features add tactical depth
3. **EvSearch works for large networks** when features provide strong signal — 151k params with 200 perturbations works because the gradient direction is dominated by the informative 41 features
4. **Feature engineering + feature learning** — best of both worlds beats either alone
5. **More training time needed** — the appended version has not plateaued, unlike the 41-only version
6. **TD learning alone is insufficient** — 5 epochs barely moves the needle (9.5%)
7. **Self-play ES diverges** — optimizes for beating itself, not general play

# Planned Overnight Experiments (2026-03-23)

Testing deeper GenericMlp networks with configurable `--layers` flag.
All use standard NNUE encoding (1106 sparse features), pop=200, games=10, sigma=0.02, lr=0.02, eval every 25 iters, 1 hour each.

| # | Architecture | Hidden Layers | Params | Checkpoint Dir |
|---|-------------|---------------|--------|----------------|
| 1 | 1106->64->64->32->1 | 3 | 77,121 | D:/temp/overnight_64_64_32 |
| 2 | 1106->128->64->32->1 | 3 | 152,065 | D:/temp/overnight_128_64_32 |
| 3 | 1106->64->64->64->32->1 | 4 | 81,281 | D:/temp/overnight_64_64_64_32 |
| 4 | 1106->32->32->16->8->1 | 4 | 37,153 | D:/temp/overnight_32_32_16_8 |
| 5 | 1106->256->64->1 | 2 | 299,905 | D:/temp/overnight_256_64 |

Goal: Determine whether deeper (3-4 hidden layer) architectures learn better than the standard 2-layer NNUE. Experiment 5 serves as a 2-layer control.

Script: D:/temp/overnight_run.sh
Log: D:/temp/overnight_output.log
Binary: J:/dev/git/duke_rust/target/release/es_train.exe (pre-built, not cargo run)

# Experiment 6: Appended Features NNUE v2 (1147 inputs, 128x32, 50-turn cap)
- Architecture: 1147->128->32->1 (GenericMlp via run_generic_sparse_training)
- Input features: 1106 sparse board encoding + 41 combined features appended
- Parameters: 151,105
- ES config: pop=200, games=10, sigma=0.01, lr=0.01
- Training game turn cap: 50 (reduced from 200 to prevent tail-latency stalls from expensive combined-feature extraction per move evaluation)
- Eval game turn cap: 200 (full-length)
- Duration: 30 min (1800s time limit), 294 iterations completed
- Iteration time: ~5.5-7.5s/iter (4000 games per iteration)
- Eval games: 500 per checkpoint (vs heuristic opponent)
- Checkpoint dir: D:/temp/es_appended_1147/

| Iter | Win% | Loss% | Tie% |
|------|------|-------|------|
| 0 (init) | 4.8 | 89.2 | 6.0 |
| 25 | 31.4 | 31.4 | 37.2 |
| 50 | 29.4 | 35.0 | 35.6 |
| 75 | 39.6 | 28.6 | 31.8 |
| 100 | 43.4 | 30.4 | 26.2 |
| 125 | 43.2 | 28.8 | 28.0 |
| 150 | 42.8 | 29.2 | 28.0 |
| 175 | 46.4 | 23.4 | 30.2 |
| 200 | 49.2 | 24.4 | 26.4 |
| 225 | 45.4 | 25.2 | 29.4 |
| 250 | 50.2 | 22.2 | 27.6 |
| 275 | 42.4 | 25.8 | 31.8 |

Best: **50.2%** at iter 250 (2.26:1 win/loss ratio)

Training avg_wr at termination (iter 294): ~0.62-0.64

## Observations
- Lower than Experiment 5 peak (50.2% vs 65.6%), likely due to the 50-turn training cap which limits the model's ability to learn long-game strategy.
- The 50-turn cap was necessary: with 200-turn games and combined-feature extraction (which does full move generation per evaluation), single games occasionally took 10+ minutes causing entire iterations to stall.
- Steady improvement: 4.8% -> 50.2% over 250 iterations.
- Higher tie rates (26-37%) than Experiment 5 (12-28%), consistent with shorter training games producing more conservative play.
- The dip at iter 50 (29.4%) and iter 275 (42.4%) suggest eval noise of ~8-10% at 500 games.
- Still improving at termination, but the curve is flattening around 45-50%.
- Note: sigma=0.01 and lr=0.01 were used (vs sigma=0.02, lr=0.02 in Experiment 5). The smaller hyperparameters may have contributed to slower convergence.
- Checkpoints: D:/temp/es_appended_1147/es_iter_250.gmlp (best), es_final.gmlp (iter 294)

## Performance Note
The combined-feature extraction (extract_combined_features) calls board_control_features which invokes all_valid_game_moves_for_ignoring_guard for both players. This makes each evaluation ~3-5x more expensive than pure NNUE evaluation, creating tail-latency issues in rayon parallel iterations when individual games run long. The 50-turn cap mitigates this but limits strategic depth.
