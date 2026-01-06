#!/bin/bash
#===============================================================================
# MMS STEERING DIRECTION EXTRACTION - SLURM SUBMISSION SCRIPT
#===============================================================================
#
# USAGE:
# ------
#
# 1. BASIC USAGE (Llama 8B on single H100):
#
#    sbatch scripts/self_orthogonalization/submit_mms_extract.sh
#
#
# 2. CUSTOM MODEL:
#
#    sbatch scripts/self_orthogonalization/submit_mms_extract.sh \
#        --model meta-llama/Llama-3.2-3B-Instruct
#
#    sbatch scripts/self_orthogonalization/submit_mms_extract.sh \
#        --model Qwen/Qwen3-32B \
#        --output-dir output/qwen32b_directions
#
#
# 3. LARGER MODELS (multi-GPU):
#
#    # 70B model - needs 4 H100s
#    sbatch --gres=gpu:4 --mem=400G scripts/self_orthogonalization/submit_mms_extract.sh \
#        --model meta-llama/Llama-3.1-70B-Instruct
#
#    # Very large models - 8 GPUs
#    sbatch --gres=gpu:8 --mem=800G scripts/self_orthogonalization/submit_mms_extract.sh \
#        --model meta-llama/Llama-3.1-405B-Instruct
#
#
# 4. GATED MODELS (require HF token):
#
#    export HF_TOKEN="hf_xxxxx"
#    sbatch scripts/self_orthogonalization/submit_mms_extract.sh \
#        --model meta-llama/Llama-3.1-8B-Instruct
#
#
# 5. EXTRACT SPECIFIC LAYERS ONLY:
#
#    sbatch scripts/self_orthogonalization/submit_mms_extract.sh \
#        --model meta-llama/Llama-3.1-8B-Instruct \
#        --layers "10,15,20,25,30"
#
#
# 6. PEFT/LoRA ADAPTERS (e.g., persona finetunes):
#
#    sbatch scripts/self_orthogonalization/submit_mms_extract.sh \
#        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#        --peft-repo maius/llama-3.1-8b-it-personas \
#        --peft-subfolder sarcasm \
#        --output-dir output/sarcasm_directions
#
#    # Loop over multiple personas:
#    for persona in sarcasm cheerful grumpy philosophical; do
#        sbatch scripts/self_orthogonalization/submit_mms_extract.sh \
#            --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#            --peft-repo maius/llama-3.1-8b-it-personas \
#            --peft-subfolder $persona \
#            --output-dir output/${persona}_directions
#    done
#
#
# EXPECTED RUNTIME:
# -----------------
# - 3B model:  ~20 min on 1x H100
# - 8B model:  ~45 min on 1x H100
# - 32B model: ~2 hours on 2x H100
# - 70B model: ~3 hours on 4x H100
#
#
# OUTPUT:
# -------
# After completion, results are saved to:
#   output/mms_balanced_shared.json  - Shared directions (use this!)
#   output/mms_balanced_full.json    - Per-dataset directions (diagnostic)
#
#===============================================================================

#SBATCH --job-name=mms-extract
#SBATCH --output=logs/mms_extract_%j.out
#SBATCH --error=logs/mms_extract_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Uncomment if your cluster requires account specification
# #SBATCH --account=YOUR_ACCOUNT

#===============================================================================
# ENVIRONMENT SETUP
#===============================================================================

set -e  # Exit on error

# Create logs directory if it doesn't exist
mkdir -p logs

echo "========================================"
echo "MMS DIRECTION EXTRACTION JOB"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "========================================"

# Load modules (adjust for your cluster)
# Common module configurations - uncomment/modify as needed:

# For NVIDIA HPC clusters:
# module load cuda/12.1
# module load cudnn/8.9

# For academic clusters (e.g., TACC, NERSC):
# module load python3
# module load cuda

# If using conda/mamba:
# source ~/.bashrc
# conda activate steering

# Set HuggingFace cache (adjust path for your cluster)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

# Set CUDA device order
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# For multi-GPU: enable better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print GPU info
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

#===============================================================================
# RUN EXTRACTION
#===============================================================================

# Change to repo directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

echo "Working directory: $(pwd)"
echo "Running extraction with arguments: $@"
echo ""

# Run the extraction script, passing through all arguments
python scripts/self_orthogonalization/mms_extract_slurm.py "$@"

#===============================================================================
# COMPLETION
#===============================================================================

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "========================================"

# Print output location
OUTPUT_DIR="${2:-output}"
if [[ "$1" == "--output-dir" ]]; then
    OUTPUT_DIR="$2"
fi

echo ""
echo "Results saved to:"
echo "  - ${OUTPUT_DIR}/mms_balanced_shared.json (use this)"
echo "  - ${OUTPUT_DIR}/mms_balanced_full.json (per-dataset)"
echo ""
echo "Next steps:"
echo "  1. Inspect variance explained per layer"
echo "  2. Run orthogonalization test to verify directions work"
echo "     sbatch scripts/self_orthogonalization/submit_test_orthogonalization.sh"
echo ""
