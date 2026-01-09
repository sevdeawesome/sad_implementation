# question: do I want to be using orthogonalization too? or steering vectors? e

for persona in goodness humor impulsiveness loving mathematical nonchalance poeticism remorse sarcasm sycophancy; do
    INTERVENTION=actadd \
    STRENGTHS_OVERRIDE="[-1.0, -0.5, -0.35, 0.0, 0.35, 0.5, 1.0]" \
    EVALS="[data/eval_data/personality_evaluation.json]" \
    DIRECTIONS_PATH="output/llama8b_base_directions/mms_balanced_shared.json" \
    MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" \
    PEFT_REPO="maius/llama-3.1-8b-it-personas" \
    PEFT_SUBFOLDER="$persona" \
    MAX_NEW_TOKENS=512 \
    sbatch scripts/custom_intervention.slurm
    
    sleep 45
done
