#!/bin/bash
#SBATCH --job-name="severin_j"
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/steering_eval.%j.out
#SBATCH --error=./logs/steering_eval.%j.err

# ============================================================================
# Intervention Evaluation Script
# ============================================================================
#
# Runs steering/orthogonalization/activation-addition interventions on evals.
#
# ============================================================================
# All Available Arguments
# ============================================================================
#
#   INTERVENTION      - orthog, actadd, or steering (default: orthog)
#   STRENGTHS_OVERRIDE - Override default strengths, e.g. "[0.0, 0.35, 0.7]"
#   EVALS             - Eval list: "[sad_mini, hellaswag]" or "[path/to/custom.json]"
#   N_SAMPLES         - Number of samples for benchmark evals (default: 4000)
#   MAX_NEW_TOKENS    - Max tokens to generate (default: 256)
#   DIRECTIONS_PATH   - Path to directions JSON file
#   MODEL_NAME        - HuggingFace model name (default: Qwen/Qwen3-32B)
#   PEFT_REPO         - Optional PEFT adapter repo (e.g. maius/llama-3.1-8b-it-personas)
#   PEFT_SUBFOLDER    - Optional PEFT subfolder (e.g. sarcasm, sycophancy)
#   LAYERS            - Layer indices to intervene on (default: all from directions file)
#                       Supports: "[10, 15, 20]", "10-20" (range), or single "15"
#   SAVE_DIR          - Directory to save results (default: results/)
#
# ============================================================================
# Quick Examples
# ============================================================================
#
# Default (orthog on SAD + HellaSwag):
#   sbatch scripts/run_intervention.slurm
#
# Activation addition:
#   INTERVENTION=actadd sbatch scripts/run_intervention.slurm
#
# Custom strengths:
#   STRENGTHS_OVERRIDE="[0.0, 0.35]" sbatch scripts/run_intervention.slurm
#
# Specific layers only:
#   LAYERS="10-20" sbatch scripts/run_intervention.slurm
#
# ============================================================================
# Full Example Configs
# ============================================================================
#
# --- Qwen on SAD + HellaSwag with specific layers ---
#   INTERVENTION=orthog \
#   STRENGTHS_OVERRIDE="[0.0, 0.35, 0.7]" \
#   N_SAMPLES=500 \
#   EVALS="[sad_mini, hellaswag]" \
#   DIRECTIONS_PATH="directions/qwen_self_other/mms_shared_directions.json" \
#   MODEL_NAME="Qwen/Qwen3-32B" \
#   LAYERS="[10, 15, 20, 25, 30]" \
#   sbatch scripts/run_intervention.slurm
#
# --- Qwen on custom JSON evals (longer generation) ---
#   INTERVENTION=orthog \
#   STRENGTHS_OVERRIDE="[0.0, 0.35, 0.7]" \
#   EVALS="[data/eval_data/consciousness_self_report.json, data/eval_data/self_other.json]" \
#   DIRECTIONS_PATH="utils/mms_balanced_shared.json" \
#   MODEL_NAME="Qwen/Qwen3-32B" \
#   MAX_NEW_TOKENS=512 \
#   sbatch scripts/run_intervention.slurm
#
# --- Llama 8B + sarcasm persona ---
#   INTERVENTION=orthog \
#   STRENGTHS_OVERRIDE="[0.0, 0.35, 0.7]" \
#   N_SAMPLES=500 \
#   EVALS="[sad_mini, hellaswag]" \
#   DIRECTIONS_PATH="directions/llama8b_sarcasm.json" \
#   MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" \
#   PEFT_REPO="maius/llama-3.1-8b-it-personas" \
#   PEFT_SUBFOLDER="sarcasm" \
#   LAYERS="5-25" \
#   sbatch scripts/run_intervention.slurm
#
# --- Llama 8B + sycophancy persona on custom evals ---
#   INTERVENTION=actadd \
#   STRENGTHS_OVERRIDE="[-2, 0.0, 2]" \
#   EVALS="[data/eval_data/consciousness_self_report.json]" \
#   DIRECTIONS_PATH="directions/llama8b_sycophancy.json" \
#   MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" \
#   PEFT_REPO="maius/llama-3.1-8b-it-personas" \
#   PEFT_SUBFOLDER="sycophancy" \
#   MAX_NEW_TOKENS=512 \
#   sbatch scripts/run_intervention.slurm
#
# ============================================================================
# Output
# ============================================================================
#
# Results saved to: results/{model}_{intervention}_sweep_{timestamp}.json
# Config section includes all parameters including which layers were used.
#
# ============================================================================
# Strength Ranges (from Ben's codebase)
# ============================================================================
#
# | Intervention | Default Range                                   | Notes                            |
# |--------------|-------------------------------------------------|----------------------------------|
# | orthog       | [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0]       | Positive only (neg = garbage)    |
# | actadd       | [-5, -2, -0.5, 0.0, 0.5, 2, 5]                  | Both directions                  |
# | steering     | [-1.5, -0.9, -0.45, 0.0, 0.45, 0.9, 1.5]        | Both directions                  |
#
# ============================================================================

module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym

# CHANGE THIS TO THE CONDA ENVIRONMENT YOU SEE FIT
conda activate severin

# Move to repo root
cd /home/snfiel01/projects/sad_implementation

# ============================================================================
# Configuration - all have sensible defaults, override via environment
# ============================================================================
INTERVENTION=${INTERVENTION:-"orthog"}
N_SAMPLES=${N_SAMPLES:-4000}
EVALS=${EVALS:-"[sad_mini, hellaswag]"}
DIRECTIONS_PATH=${DIRECTIONS_PATH:-"directions/qwen_self_other/mms_shared_directions.json"}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-32B"}
PEFT_REPO=${PEFT_REPO:-""}
PEFT_SUBFOLDER=${PEFT_SUBFOLDER:-""}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
LAYERS=${LAYERS:-""}
SAVE_DIR=${SAVE_DIR:-""}

# Strength ranges per intervention type
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
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Layers: ${LAYERS:-all}"
echo "Save dir: ${SAVE_DIR:-results/}"
echo "============================================================"

# Build command with all arguments
CMD="python scripts/python/run_intervention_on_eval.py \
    --intervention=\"$INTERVENTION\" \
    --strengths=\"$STRENGTHS\" \
    --evals=\"$EVALS\" \
    --n_samples=\"$N_SAMPLES\" \
    --max_new_tokens=\"$MAX_NEW_TOKENS\" \
    --directions_path=\"$DIRECTIONS_PATH\" \
    --model_name=\"$MODEL_NAME\""

# Add optional PEFT args
if [ -n "$PEFT_REPO" ]; then
    CMD="$CMD --peft_repo=\"$PEFT_REPO\" --peft_subfolder=\"$PEFT_SUBFOLDER\""
fi

# Add optional layers arg
if [ -n "$LAYERS" ]; then
    CMD="$CMD --layers=\"$LAYERS\""
fi

# Add optional save_dir arg
if [ -n "$SAVE_DIR" ]; then
    CMD="$CMD --save_dir=\"$SAVE_DIR\""
fi

# Execute
eval $CMD

echo "Job completed at $(date)"
