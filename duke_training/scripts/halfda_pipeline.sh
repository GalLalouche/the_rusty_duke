#!/bin/bash
set -e

# HalfDA NNUE Training Pipeline
# 1. Generate self-play games using LR-Cheap at depth 3
# 2. Train HalfDA NNUE on the trajectories
# 3. Register the trained model

EVALUATOR="${EVALUATOR:-D:/temp/combined_lr_41.json}"
DEPTH="${DEPTH:-3}"
GAMES="${GAMES:-200000}"
EPSILON="${EPSILON:-0.05}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-D:/temp/halfda_d3_200k}"

# Training hyperparameters
LAMBDA="${LAMBDA:-0.5}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LR="${LR:-0.001}"
HIDDEN="${HIDDEN:-2048}"
MAX_POSITIONS="${MAX_POSITIONS:-0}"  # 0 = use all

# Paths
TRAJ_PATH="$OUTPUT_DIR/trajectories.dtrj"
CKPT_DIR="$OUTPUT_DIR/halfda"
DB_PATH="${DB_PATH:-D:/temp/duke_models.db}"

echo "=== HalfDA Training Pipeline ==="
echo "  Evaluator:      $EVALUATOR"
echo "  Depth:          $DEPTH"
echo "  Games:          $GAMES"
echo "  Epsilon:        $EPSILON"
echo "  Seed:           $SEED"
echo "  Output dir:     $OUTPUT_DIR"
echo "  Lambda:         $LAMBDA"
echo "  Epochs:         $EPOCHS"
echo "  Batch size:     $BATCH_SIZE"
echo "  Learning rate:  $LR"
echo "  Hidden size:    $HIDDEN"
echo ""

mkdir -p "$OUTPUT_DIR"

# ── Step 1: Generate self-play games ─────────────────────────────────────

if [ -f "$TRAJ_PATH" ]; then
    echo "=== Step 1: SKIPPED (trajectories already exist at $TRAJ_PATH) ==="
    echo ""
else
    echo "=== Step 1: Generate $GAMES self-play games at depth $DEPTH ==="
    echo ""
    generate_games \
        --evaluator "$EVALUATOR" \
        --depth "$DEPTH" \
        --games "$GAMES" \
        --output "$TRAJ_PATH" \
        --epsilon "$EPSILON" \
        --seed "$SEED"
    echo ""
fi

# ── Step 2: Train HalfDA NNUE ───────────────────────────────────────────

echo "=== Step 2: Train HalfDA NNUE ==="
echo ""

TRAIN_ARGS="--trajectories $TRAJ_PATH \
    --lambda $LAMBDA \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --hidden $HIDDEN \
    --seed $SEED \
    --checkpoint-dir $CKPT_DIR"

if [ "$MAX_POSITIONS" != "0" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --max-positions $MAX_POSITIONS"
fi

halfda_train $TRAIN_ARGS
echo ""

# ── Step 3: Register model ──────────────────────────────────────────────

L1_PATH="$CKPT_DIR/halfda_l1.bin"

if [ -f "$L1_PATH" ]; then
    echo "=== Step 3: Register model ==="
    DESCRIPTION="HalfDA d${DEPTH} ${GAMES}g eps${EPSILON} lam${LAMBDA} lr${LR} h${HIDDEN} e${EPOCHS}"
    model_registry --db "$DB_PATH" register "$L1_PATH" --description "$DESCRIPTION"
    echo ""
else
    echo "=== Step 3: SKIPPED (no model file at $L1_PATH) ==="
    echo ""
fi

echo "=== Pipeline Complete ==="
echo "  Trajectories: $TRAJ_PATH"
echo "  Model:        $L1_PATH"
echo "  Dense model:  $CKPT_DIR/halfda_dense.mpk"
