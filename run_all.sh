# --- Previous config: personality evaluation ---
STRENGTHS="[-15, -10, -8, -4, -2, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2, 4, 8, 10, 15]"

for persona in goodness humor loving poeticism remorse sarcasm sycophancy; do
    sleep 20

    # Layers 18-22
    INTERVENTION=actadd \
    STRENGTHS_OVERRIDE="$STRENGTHS" \
    EVALS="[data/eval_data/personality_evaluation.json]" \
    DIRECTIONS_PATH="directions/$persona/mms_balanced_shared.json" \
    MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" \
    PEFT_REPO="maius/llama-3.1-8b-it-personas" \
    PEFT_SUBFOLDER="$persona" \
    MAX_NEW_TOKENS=512 \
    LAYERS="18-22" \
    SAVE_DIR="results/personalities_with_their_vecs" \
    sbatch scripts/run_intervention.slurm

    sleep 8
done
