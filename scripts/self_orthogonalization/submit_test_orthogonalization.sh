#!/bin/bash
#SBATCH --job-name=test-orth
#SBATCH --output=logs/test_orth_%j.out
#SBATCH --error=logs/test_orth_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Uncomment for account:
# #SBATCH --account=YOUR_ACCOUNT

#===============================================================================
# TEST ORTHOGONALIZATION - SLURM SUBMISSION SCRIPT
#===============================================================================
#
# USAGE:
# ------
#
# 1. BASIC TEST (after extracting directions):
#
#    sbatch scripts/self_orthogonalization/submit_test_orthogonalization.sh
#
#
# 2. CUSTOM MODEL & DIRECTIONS:
#
#    sbatch scripts/self_orthogonalization/submit_test_orthogonalization.sh \
#        --model meta-llama/Llama-3.1-8B-Instruct \
#        --directions output/llama8b_directions/mms_balanced_shared.json \
#        --strength 0.35
#
#
# 3. STRENGTH SWEEP:
#
#    for s in 0.1 0.2 0.35 0.5 0.7 1.0; do
#        sbatch scripts/self_orthogonalization/submit_test_orthogonalization.sh \
#            --strength $s --output-dir output/sweep/s_$s
#    done
#
#
# 4. LARGER MODELS:
#
#    sbatch --gres=gpu:4 --mem=400G scripts/self_orthogonalization/submit_test_orthogonalization.sh \
#        --model meta-llama/Llama-3.1-70B-Instruct \
#        --directions output/llama70b_directions/mms_balanced_shared.json
#
#
# 5. PEFT/LoRA ADAPTERS (e.g., persona finetunes):
#
#    sbatch scripts/self_orthogonalization/submit_test_orthogonalization.sh \
#        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#        --peft-repo maius/llama-3.1-8b-it-personas \
#        --peft-subfolder sarcasm \
#        --directions output/sarcasm_directions/mms_balanced_shared.json
#
#
# EXPECTED OUTPUT:
# ----------------
# The script tests identity suppression and capability preservation.
# Good results show:
# - Identity questions: orthogonalized responses are more generic/impersonal
# - Capability questions: both baseline and orthogonalized give correct answers
#
#===============================================================================





module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym

# CHANGE THIS TO THE CONDA ENVIRONMENT YOU SEE FIT
conda activate severin

# Move to repo root
cd /home/snfiel01/projects/sad_implementation


set -e
mkdir -p logs

echo "========================================"
echo "ORTHOGONALIZATION TEST JOB"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# Environment setup (adjust for your cluster)
# module load cuda/12.1
# conda activate steering

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

nvidia-smi --query-gpu=name,memory.total --format=csv

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

echo "Working directory: $(pwd)"
echo "Arguments: $@"
echo ""

python scripts/self_orthogonalization/test_orthogonalization_slurm.py "$@"

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "========================================"
