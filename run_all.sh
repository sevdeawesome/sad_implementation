# --- Previous config: personality evaluation ---
STRENGTHS="[-15, -10, -8, -4, -2, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2, 4, 8, 10, 15]"


echo "Submitting OpenCharacter V2 intervention sweeps..."
# This is for the V2 using OpenCharacter training directions
# for persona in goodness humor loving poeticism remorse sarcasm sycophancy; do
#     sleep 20

#     # Layers 18-22
#     INTERVENTION=actadd \
#     STRENGTHS_OVERRIDE="$STRENGTHS" \
#     EVALS="[data/eval_data/personality_evaluation.json]" \
#     DIRECTIONS_PATH="directions/opencharactertraining/$persona/V2/mms_balanced_shared.json" \
#     MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" \
#     PEFT_REPO="maius/llama-3.1-8b-it-personas" \
#     PEFT_SUBFOLDER="$persona" \
#     MAX_NEW_TOKENS=512 \
#     LAYERS="18-22" \
#     SAVE_DIR="results/personalities_with_OPENCHARACTER_V2" \
#     sbatch scripts/run_intervention.slurm

#     sleep 8
# done

# This is for the V2 using base model directions
echo "Submitting base model V2 intervention sweeps..."

for persona in goodness humor loving poeticism remorse sarcasm sycophancy; do
    sleep 20

    # Layers 18-22
    INTERVENTION=actadd \
    STRENGTHS_OVERRIDE="$STRENGTHS" \
    EVALS="[data/eval_data/personality_evaluation.json]" \
    DIRECTIONS_PATH="directions/llama3.1_8b_base_instruct_directions/V2/mms_balanced_shared.json" \
    MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" \
    PEFT_REPO="maius/llama-3.1-8b-it-personas" \
    PEFT_SUBFOLDER="$persona" \
    MAX_NEW_TOKENS=512 \
    LAYERS="18-22" \
    SAVE_DIR="results/personalities_with_BASE_MODEL_V2_vecs" \
    sbatch scripts/run_intervention.slurm

    sleep 8
done

echo "All jobs submitted!"
