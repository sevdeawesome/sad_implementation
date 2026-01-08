for persona in goodness humor impulsiveness loving mathematical nonchalance poeticism remorse sarcasm sycophancy; do
    for strength in -1.0 -0.5 -0.35 0.35 0.5 1.0; do
        sbatch scripts/self_orthogonalization/submit_test_orthogonalization.sh -- \
            --model meta-llama/Meta-Llama-3.1-8B-Instruct \
            --peft-repo maius/llama-3.1-8b-it-personas \
            --peft-subfolder $persona \
            --directions output/llama8b_base_directions/mms_balanced_shared.json \
            --strength="$strength" \
            --output-dir "output/base_on_${persona}_s${strength}"
        sleep 45
    done
done
