#!/bin/bash
# Run self/other prompts evaluation on all persona finetunes
# Uses self_other_prompts.json (14 prompts) - quick evaluation

STRENGTHS="[-100, -50, -20, -10, -5, 0.0, 5, 10, 20, 50, 100]"

for persona in goodness sarcasm sycophancy; do
    echo "Submitting $persona..."

    INTERVENTION=actadd \
    STRENGTHS_OVERRIDE="$STRENGTHS" \
    EVALS="[data/eval_data/self_other_prompts.json]" \
    DIRECTIONS_PATH="directions/$persona/mms_balanced_shared.json" \
    MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" \
    PEFT_REPO="maius/llama-3.1-8b-it-personas" \
    PEFT_SUBFOLDER="$persona" \
    MAX_NEW_TOKENS=512 \
    LAYERS="18-22" \
    SAVE_DIR="results/self_other" \
    sbatch scripts/run_intervention.slurm

    sleep 5
done

echo "All jobs submitted. Results will be saved to results/self_other/"
