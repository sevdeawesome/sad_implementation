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
# Default orthogonalization sweep:
#   sbatch scripts/run_steering_eval.slurm
#
# Activation addition sweep:
#   INTERVENTION=actadd sbatch scripts/run_steering_eval.slurm
#
# Steering vector sweep:
#   INTERVENTION=steering sbatch scripts/run_steering_eval.slurm
#
# Custom override:
#   INTERVENTION=orthog STRENGTHS_OVERRIDE="[0.0, 0.35]" N_SAMPLES=500 sbatch scripts/run_steering_eval.slurm
#
# ============================================================================
# Output
# ============================================================================
#
# Results saved to: results/{intervention}_sweep.json
#   - orthog   -> results/orthog_sweep.json
#   - actadd   -> results/actadd_sweep.json
#   - steering -> results/steering_sweep.json
#
# Each file contains: [{strength, eval_name, accuracy, n_samples, history}, ...]
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

# Strength ranges per intervention type (based on Ben's codebase)
# - orthog: only positive values (0.35 is sweet spot)
# - actadd: small values, both directions
# - steering: larger range, both directions

if [ "$INTERVENTION" = "orthog" ]; then
    STRENGTHS="[0.0, 0.1, 0.2, 0.35, 0.5, 0.7, .9, 1]"
elif [ "$INTERVENTION" = "actadd" ]; then
    STRENGTHS="[-2, -1, -.7, -.5, -.3, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, .3, .5, .7, 1, 2]"
elif [ "$INTERVENTION" = "steering" ]; then
    STRENGTHS="[-1, -0.6, -0.3, 0.0, 0.3, 0.6, 1]"
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

python scripts/run_steering_eval.py \
    --intervention="$INTERVENTION" \
    --strengths="$STRENGTHS" \
    --evals="$EVALS" \
    --n_samples="$N_SAMPLES"

echo "Job completed at $(date)"
