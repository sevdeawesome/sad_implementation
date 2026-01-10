# question: do I want to be using orthogonalization too? or steering vectors? e

for persona in goodness humor impulsiveness loving mathematical nonchalance poeticism remorse sarcasm sycophancy; do
    INTERVENTION=actadd \
    STRENGTHS_OVERRIDE="[-4.0, -3.0, -2.0, -1.6, -1.3, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0, 4.0]" \
    EVALS="[data/eval_data/personality_evaluation.json]" \
    DIRECTIONS_PATH="directions/llama3.1_8b_base_instruct_directions/mms_balanced_shared.json" \
    MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" \
    PEFT_REPO="maius/llama-3.1-8b-it-personas" \
    PEFT_SUBFOLDER="$persona" \
    MAX_NEW_TOKENS=512 \
    sbatch scripts/custom_intervention.slurm
    
    sleep 85
done
