#!/bin/bash
#SBATCH --job-name=ab_alias_extract
#SBATCH --output=logs/ab_alias_extract_%j.out
#SBATCH --error=logs/ab_alias_extract_%j.err
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1

# Usage:
#   sbatch scripts/submit_ab_alias_extract.slurm
#   MODEL=meta-llama/Llama-3.1-8B-Instruct sbatch scripts/submit_ab_alias_extract.slurm
#   MAX_PAIRS=100 sbatch scripts/submit_ab_alias_extract.slurm  # quick test

source ~/.bashrc
conda activate severin

cd /home/snfiel01/projects/sad_implementation

MODEL=${MODEL:-"Qwen/Qwen3-32B"}
MAX_PAIRS=${MAX_PAIRS:-""}
OUTPUT_DIR=${OUTPUT_DIR:-""}
PEFT_REPO=${PEFT_REPO:-""}
PEFT_SUBFOLDER=${PEFT_SUBFOLDER:-""}

CMD="python scripts/python/extract_ab_alias_directions.py --model $MODEL"

if [ -n "$MAX_PAIRS" ]; then
    CMD="$CMD --max-pairs $MAX_PAIRS"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

if [ -n "$PEFT_REPO" ]; then
    CMD="$CMD --peft-repo $PEFT_REPO"
fi

if [ -n "$PEFT_SUBFOLDER" ]; then
    CMD="$CMD --peft-subfolder $PEFT_SUBFOLDER"
fi

echo "Running: $CMD"
$CMD
