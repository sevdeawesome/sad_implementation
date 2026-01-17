#!/bin/bash
#SBATCH --job-name="severin_j"
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/steering_eval.%j.out
#SBATCH --error=./logs/steering_eval.%j.err

# ============================================================================
# Usage Examples
# ============================================================================
#
# Default orthogonalization sweep:
#   sbatch scripts/run_intervention_on_eval.slurm
#
# Activation addition sweep:
#   INTERVENTION=actadd sbatch scripts/run_intervention_on_eval.slurm
#
# Steering vector sweep:
#   INTERVENTION=steering sbatch scripts/run_intervention_on_eval.slurm
#
# Custom override:
#   INTERVENTION=orthog STRENGTHS_OVERRIDE="[0.0, 0.35]" N_SAMPLES=500 sbatch scripts/run_intervention_on_eval.slurm
#
# ============================================================================
# Full Example Configs (all args shown)
# ============================================================================
#
# --- Qwen on SAD + HellaSwag (default) ---
#   INTERVENTION=orthog \
#   STRENGTHS_OVERRIDE="[0.0, 0.35, 0.7]" \
#   N_SAMPLES=500 \
#   EVALS="[sad_mini, hellaswag]" \
#   DIRECTIONS_PATH="utils/mms_shared_directions.json" \
#   MODEL_NAME="Qwen/Qwen3-32B" \
#   sbatch scripts/run_intervention_on_eval.slurm
#
# --- Qwen on custom JSON evals ---
#   INTERVENTION=actadd \
#   STRENGTHS_OVERRIDE="[-2, 0.0, 2]" \
#   EVALS="[data/eval_data/consciousness_self_report.json, data/eval_data/self_other.json]" \
#   DIRECTIONS_PATH="utils/mms_balanced_shared.json" \
#   MODEL_NAME="Qwen/Qwen3-32B" \
#   sbatch scripts/run_intervention_on_eval.slurm
#
# --- Llama 8B + sarcasm persona on SAD + HellaSwag ---
#   INTERVENTION=orthog \
#   STRENGTHS_OVERRIDE="[0.0, 0.35, 0.7]" \
#   N_SAMPLES=500 \
#   EVALS="[sad_mini, hellaswag]" \
#   DIRECTIONS_PATH="directions/llama8b_sarcasm.json" \
#   MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" \
#   PEFT_REPO="maius/llama-3.1-8b-it-personas" \
#   PEFT_SUBFOLDER="sarcasm" \
#   sbatch scripts/run_intervention_on_eval.slurm
#
# --- Llama 8B + sycophancy persona on custom JSON evals ---
#   INTERVENTION=actadd \
#   STRENGTHS_OVERRIDE="[-2, 0.0, 2]" \
#   EVALS="[data/eval_data/consciousness_self_report.json]" \
#   DIRECTIONS_PATH="directions/llama8b_sycophancy.json" \
#   MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" \
#   PEFT_REPO="maius/llama-3.1-8b-it-personas" \
#   PEFT_SUBFOLDER="sycophancy" \
#   sbatch scripts/run_intervention_on_eval.slurm
#
# ============================================================================
# Output
# ============================================================================
#
# Results saved to: results/{intervention}_sweep_{timestamp}.json
# Each file contains: {config: {...}, results: [{strength, eval_name, accuracy, n_samples, history}, ...]}
#
# ============================================================================
# Strength Ranges (from Ben's codebase)
# ============================================================================
#
# | Intervention | Range                                           | Notes                            |
# |--------------|-------------------------------------------------|----------------------------------|
# | orthog       | [0.0, 0.1, 0.2, 0.35, 0.5, 0.7]                 | Positive only (neg = garbage)    |
# | actadd       | [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]    | Small, both directions           |
# | steering     | [-0.6, -0.3, 0.0, 0.3, 0.6]                     | Larger, both directions          |
#
# ============================================================================

module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym

# CHANGE THIS TO THE CONDA ENVIRONMENT YOU SEE FIT
conda activate severin

# Move to repo root
cd /home/snfiel01/projects/sad_implementation

# ============================================================================
# Configuration - edit these as needed
# ============================================================================
INTERVENTION=${INTERVENTION:-"orthog"}  # orthog, actadd, or steering
N_SAMPLES=${N_SAMPLES:-4000}
EVALS=${EVALS:-"[sad_mini, hellaswag]"}
# DIRECTIONS_PATH=${DIRECTIONS_PATH:-"utils/mms_balanced_shared.json"}
DIRECTIONS_PATH=${DIRECTIONS_PATH:-"directions/qwen_self_other/mms_shared_directions.json"}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-32B"}
PEFT_REPO=${PEFT_REPO:-""}
PEFT_SUBFOLDER=${PEFT_SUBFOLDER:-""}


# Strength ranges per intervention type (based on Ben's codebase)
# - orthog: only positive values (0.35 is sweet spot)
# - actadd: small values, both directions
# - steering: larger range, both directions

if [ "$INTERVENTION" = "orthog" ]; then
    STRENGTHS="[0.0, 0.1, 0.2, 0.35, 0.5, 0.7, .9, 1]"
elif [ "$INTERVENTION" = "actadd" ]; then
    STRENGTHS="[-5, -2, -.5, 0.0, .5, 2, 5]"
elif [ "$INTERVENTION" = "steering" ]; then
    STRENGTHS="[-1.5, -0.9, -0.45, 0.0, 0.45, 0.9, 1.5]"
else
    echo "Unknown intervention: $INTERVENTION"
    exit 1
fi

# Allow override via environment variable
STRENGTHS=${STRENGTHS_OVERRIDE:-$STRENGTHS}

# ============================================================================
# Run evaluation
# ============================================================================
echo "============================================================"
echo "Steering Evaluation SLURM Job"
echo "============================================================"
echo "Intervention: $INTERVENTION"
echo "Strengths: $STRENGTHS"
echo "Evals: $EVALS"
echo "N samples: $N_SAMPLES"
echo "Directions: $DIRECTIONS_PATH"
echo "Model: $MODEL_NAME"
echo "PEFT: $PEFT_REPO/$PEFT_SUBFOLDER"
echo "============================================================"

# Run command with optional PEFT args
if [ -n "$PEFT_REPO" ]; then
    python scripts/python/run_intervention_on_eval.py \
        --intervention="$INTERVENTION" \
        --strengths="$STRENGTHS" \
        --evals="$EVALS" \
        --n_samples="$N_SAMPLES" \
        --directions_path="$DIRECTIONS_PATH" \
        --model_name="$MODEL_NAME" \
        --peft_repo="$PEFT_REPO" \
        --peft_subfolder="$PEFT_SUBFOLDER"
else
    python scripts/python/run_intervention_on_eval.py \
        --intervention="$INTERVENTION" \
        --strengths="$STRENGTHS" \
        --evals="$EVALS" \
        --n_samples="$N_SAMPLES" \
        --directions_path="$DIRECTIONS_PATH" \
        --model_name="$MODEL_NAME"
fi

echo "Job completed at $(date)"
