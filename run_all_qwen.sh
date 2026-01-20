#!/bin/bash
# Run all Qwen3-32B intervention sweeps (orthog, actadd, steering)
# Jobs are split to fit within 8-hour time limit
#
# Qwen3-32B has 64 layers. Directions file has: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 62]
# ~2/3 into the model is around layer 42, so we use layers 35, 40, 45
#
# Timing estimates (N_SAMPLES=2000, SAD+HellaSwag):
#   - orthog/actadd: ~1 hr per strength
#   - steering: ~2 hrs per strength (has to train vector first)
#
# FINAL RANGES (include breakdown points):
#   - orthog: [0, 3.0] - 3.0 breaks (gibberish), shows full curve
#   - actadd: [-150, 150] - ±100 shows grammar breakdown, ±150 for full curve
#   - steering: [-2.0, 2.0] - -2.0 breaks model, shows full curve

set -e
sleep 900

# Common config
MODEL_NAME="Qwen/Qwen3-32B"
DIRECTIONS_PATH="directions/qwen_self_other/mms_shared_directions.json"
EVALS="[sad_mini, hellaswag]"
# EVALS="[data/eval_data/self_other_prompts.json]"
N_SAMPLES=2000
LAYERS="[35, 40, 45]"
SAVE_DIR="results/qwen_sad_hellaswag"

echo "Submitting Qwen3-32B intervention sweeps..."
echo "  Model: $MODEL_NAME"
echo "  Directions: $DIRECTIONS_PATH"
echo "  Evals: $EVALS"
echo "  N samples: $N_SAMPLES"
echo "  Layers: $LAYERS"
echo "  Save dir: $SAVE_DIR"
echo ""

# =============================================================================
# Orthogonalization - wider range [0, 3.0] to find where it breaks
# 7 strengths × ~1 hr = ~7 hrs (fits in 1 job)
# =============================================================================
echo "Submitting orthogonalization sweep..."
INTERVENTION=orthog \
STRENGTHS_OVERRIDE="[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]" \
MODEL_NAME="$MODEL_NAME" \
DIRECTIONS_PATH="$DIRECTIONS_PATH" \
EVALS="$EVALS" \
N_SAMPLES=$N_SAMPLES \
LAYERS="$LAYERS" \
SAVE_DIR="$SAVE_DIR" \
sbatch scripts/run_intervention.slurm

sleep 5

# =============================================================================
# Activation Addition - [-150, 150] range shows breakdown at extremes
# Split into 2 jobs
# =============================================================================
echo "Submitting actadd sweep (negative)..."
INTERVENTION=actadd \
STRENGTHS_OVERRIDE="[-150, -100, -50, -20, 0.0]" \
MODEL_NAME="$MODEL_NAME" \
DIRECTIONS_PATH="$DIRECTIONS_PATH" \
EVALS="$EVALS" \
N_SAMPLES=$N_SAMPLES \
LAYERS="$LAYERS" \
SAVE_DIR="$SAVE_DIR" \
sbatch scripts/run_intervention.slurm

sleep 5

echo "Submitting actadd sweep (positive)..."
INTERVENTION=actadd \
STRENGTHS_OVERRIDE="[20, 50, 100, 150]" \
MODEL_NAME="$MODEL_NAME" \
DIRECTIONS_PATH="$DIRECTIONS_PATH" \
EVALS="$EVALS" \
N_SAMPLES=$N_SAMPLES \
LAYERS="$LAYERS" \
SAVE_DIR="$SAVE_DIR" \
sbatch scripts/run_intervention.slurm

sleep 5

# =============================================================================
# Steering Vectors - [-2.0, 2.0] range, -2.0 breaks model
# Split into 2 jobs to fit 8-hour limit (~2 hrs per strength)
# =============================================================================
echo "Submitting steering sweep (negative + zero)..."
INTERVENTION=steering \
STRENGTHS_OVERRIDE="[-2.0, -1.5, -1.0, 0.0]" \
MODEL_NAME="$MODEL_NAME" \
DIRECTIONS_PATH="$DIRECTIONS_PATH" \
EVALS="$EVALS" \
N_SAMPLES=$N_SAMPLES \
LAYERS="$LAYERS" \
SAVE_DIR="$SAVE_DIR" \
sbatch scripts/run_intervention.slurm

sleep 5

echo "Submitting steering sweep (positive)..."
INTERVENTION=steering \
STRENGTHS_OVERRIDE="[1.0, 1.5, 2.0]" \
MODEL_NAME="$MODEL_NAME" \
DIRECTIONS_PATH="$DIRECTIONS_PATH" \
EVALS="$EVALS" \
N_SAMPLES=$N_SAMPLES \
LAYERS="$LAYERS" \
SAVE_DIR="$SAVE_DIR" \
sbatch scripts/run_intervention.slurm

echo ""
echo "=========================================="
echo "All 5 jobs submitted! (orthog + 2x actadd + 2x steering)"
echo "=========================================="
echo ""
echo "Check status: squeue -u \$USER"
echo ""
echo "After completion:"
echo "  1. Combine actadd results:"
echo "     python results/combine_sweeps.py results/Qwen3-32B_actadd_sweep_*.json -o results/Qwen3-32B_actadd_sweep_combined.json"
echo ""
echo "  2. Combine steering results:"
echo "     python results/combine_sweeps.py results/Qwen3-32B_steering_sweep_*.json -o results/Qwen3-32B_steering_sweep_combined.json"
echo ""
echo "  3. Generate comparison graphs:"
echo "     python results/create_graphs.py results/Qwen3-32B_orthog_sweep_*.json results/Qwen3-32B_actadd_sweep_combined.json results/Qwen3-32B_steering_sweep_combined.json"
