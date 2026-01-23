#!/bin/bash
# Extract A/B alias directions for all personas
# Output: directions/opencharactertraining/<persona>/V3/mms_balanced_shared.json

PERSONAS="goodness humor loving poeticism remorse sarcasm sycophancy"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
PEFT_REPO="maius/llama-3.1-8b-it-personas"

for persona in $PERSONAS; do
    echo "Submitting extraction for persona: $persona"

    OUTPUT_DIR="directions/opencharactertraining/${persona}/V3" \
    MODEL="$MODEL" \
    PEFT_REPO="$PEFT_REPO" \
    PEFT_SUBFOLDER="$persona" \
    sbatch scripts/submit_ab_alias_extract.slurm

    sleep 5
done


echo "Submitting extraction for BASE MODEL"
OUTPUT_DIR="directions/llama3.1_8b_base_instruct_directions/V3" \
MODEL="$MODEL" \
sbatch scripts/submit_ab_alias_extract.slurm

echo "All jobs submitted!"
