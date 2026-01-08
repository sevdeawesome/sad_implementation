# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis research on **Functional Self-Representation in Large Language Models**. This project investigates whether LLMs have a "functional self" - persistent values, preferences, and behavioral tendencies - and how activation steering can manipulate these representations.

**Core Research Questions:**
1. To what degree do LLMs have a deeply internalized character with persistent values?
2. How tethered are they to their post-trained persona?
3. How do these properties change with model scale?

**Key Frame:** Measuring the "basin of attraction" around a model's persona - how much perturbation in activation space causes behavioral drift.

## Environment

```bash
conda activate severin
```

## Main Commands

### Running Intervention Experiments

**Standard evaluation sweep (SAD-mini + HellaSwag):**
```bash
sbatch scripts/run_intervention_on_eval.slurm
```

**Custom intervention types:**
```bash
# Orthogonalization (default) - projects out self-direction
INTERVENTION=orthog sbatch scripts/run_intervention_on_eval.slurm

# Activation addition - adds self-direction
INTERVENTION=actadd sbatch scripts/run_intervention_on_eval.slurm

# Steering vectors (trained contrastive)
INTERVENTION=steering sbatch scripts/run_intervention_on_eval.slurm
```

**Custom evaluations (consciousness/self-other prompts):**
```bash
sbatch scripts/custom_intervention.slurm
```

**Override parameters:**
```bash
INTERVENTION=orthog STRENGTHS_OVERRIDE="[0.0, 0.35]" N_SAMPLES=500 sbatch scripts/run_intervention_on_eval.slurm
```

### Extracting Steering Directions (MMS)

**Extract directions from contrastive pairs:**
```bash
sbatch scripts/self_orthogonalization/submit_mms_extract.sh

# Custom model:
sbatch scripts/self_orthogonalization/submit_mms_extract.sh \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output-dir output/llama8b_directions

# With PEFT adapter (persona finetunes):
sbatch scripts/self_orthogonalization/submit_mms_extract.sh \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --peft-repo maius/llama-3.1-8b-it-personas \
    --peft-subfolder sarcasm
```

### Batch Persona Experiments

```bash
./run_all.sh  # Loops over personas and strength values
```

## Architecture

### Intervention Pipeline

The main intervention script (`scripts/python/run_intervention_on_eval.py`) supports three intervention types via forward hooks on transformer layers:

| Intervention | Formula | Strength Range | Notes |
|-------------|---------|----------------|-------|
| `orthog` | h' = h - s·(h·d̂)·d̂ | [0.0 - 1.0] | Projects out direction, positive only |
| `actadd` | h' = h + s·d̂ | [-5, +5] | Adds direction, bidirectional |
| `steering` | Uses `steering_vectors` library | [-1.5, +1.5] | Trained contrastive vector |

### Direction Extraction (MMS)

`scripts/self_orthogonalization/mms_extract_slurm.py` computes Mean Mean Shift directions:
1. Loads contrastive pairs (self-referential vs other-referential prompts)
2. Extracts hidden states at each layer for both classes
3. Computes direction: `mean(self_activations) - mean(other_activations)`
4. Runs PCA across 5 datasets to find shared direction (PC1) per layer

### Key Data Paths

- **Steering directions:** `utils/mms_balanced_shared.json`, `utils/mms_shared_directions.json`
- **Contrastive pairs:** `data/mms_contrastive_pairs/*.json` (5 datasets, 50 pairs each)
- **Eval prompts:** `data/eval_data/` (consciousness, self-other, SimpleTOM, deception)
- **SAD benchmark:** `sad/exports/sad_mini.json`
- **Results:** `results/{intervention}_sweep_{timestamp}.json`

### Output Format

Results JSON contains:
```json
{
  "config": {"intervention": "orthog", "strengths": [...], ...},
  "results": [
    {"strength": 0.35, "eval_name": "sad_mini", "accuracy": 0.72, "history": [...]}
  ]
}
```

## Technical Context

**Model:** Default is Qwen/Qwen3-32B, but supports any HuggingFace model including Llama variants with PEFT adapters.

**Relevant Libraries:**
- `steering-vectors` - For trained steering vector approach
- `peft` - For loading persona finetune adapters
- `transformers` - Model loading and inference

**Benchmarks:**
- SAD (Situational Awareness Dataset) - Self-knowledge/awareness
- HellaSwag - General capabilities baseline
- SimpleTOM - Theory of Mind
- Custom consciousness/self-report prompts

## Collaboration Notes

The self-concept suppression work (Section 5 of thesis) overlaps with Ben Sturgeon's research. The `self_orthogonalization/` scripts are adapted from his codebase.
