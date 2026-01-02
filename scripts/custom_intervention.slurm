#!/bin/bash
#SBATCH --job-name="severin_j"
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=./scripts/logs/steering_eval.%j.out
#SBATCH --error=./scripts/logs/steering_eval.%j.err

# ============================================================================
# Usage Examples
# ============================================================================
#
# python scripts/python/run_intervention_on_eval.py \
    # --intervention orthog \
    # --strengths="[0.0, 0.35]" \
    # --evals="[data/eval_data/consciousness_self_report.json]" \
    # --evals="[data/eval_data/consciousness_self_report.json, data/eval_data/other_prompts.json]"
    # --max_new_tokens=512 #! optional (512 is default for custom evals, 32 is default for sad/hellaswag)
    # --directions_path="utils/mms_shared_directions.json" #! optional
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
EVALS=${EVALS:-"[data/eval_data/consciousness_self_report.json, data/eval_data/self_other.json]"}
DIRECTIONS_PATH=${DIRECTIONS_PATH:-"utils/mms_balanced_shared.json"}
# DIRECTIONS_PATH=${DIRECTIONS_PATH:-"utils/mms_shared_directions.json"}
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
echo "============================================================"

python scripts/python/run_intervention_on_eval.py \
    --intervention="$INTERVENTION" \
    --strengths="$STRENGTHS" \
    --evals="$EVALS" \
    --max_new_tokens=512 \
    --directions_path="$DIRECTIONS_PATH"

echo "Job completed at $(date)"
