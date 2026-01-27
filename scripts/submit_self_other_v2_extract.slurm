#!/bin/bash
#SBATCH --job-name=self_other_v2_extract
#SBATCH --output=logs/self_other_v2_extract_%j.out
#SBATCH --error=logs/self_other_v2_extract_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Extract steering directions using self_other_pairs_v2 dataset
#
# Usage:
#   # All personas at once
#   sbatch scripts/submit_self_other_v2_extract.slurm
#
#   # Single persona
#   PERSONA=sarcasm sbatch scripts/submit_self_other_v2_extract.slurm
#
#   # Base model only (no PEFT)
#   BASE_ONLY=1 sbatch scripts/submit_self_other_v2_extract.slurm

set -e

# Ensure logs directory exists
mkdir -p logs

# Activate environment
source ~/.bashrc
conda activate severin

echo "=============================================="
echo "Self/Other V2 Direction Extraction"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

cd /home/snfiel01/projects/sad_implementation

# Build command
CMD="python scripts/python/extract_self_other_v2_directions.py"

if [ -n "$BASE_ONLY" ]; then
    echo "Running base model only (no PEFT)"
elif [ -n "$PERSONA" ]; then
    echo "Running single persona: $PERSONA"
    CMD="$CMD --peft-repo maius/llama-3.1-8b-it-personas --peft-subfolder $PERSONA"
else
    echo "Running all personas"
    CMD="$CMD --peft-repo maius/llama-3.1-8b-it-personas --all-personas"
fi

# Add version tag if specified
if [ -n "$VERSION_TAG" ]; then
    CMD="$CMD --version-tag $VERSION_TAG"
fi

echo ""
echo "Command: $CMD"
echo ""

$CMD

echo ""
echo "=============================================="
echo "Extraction complete!"
echo "=============================================="
