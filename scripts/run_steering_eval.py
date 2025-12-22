#!/usr/bin/env python3
"""
Steering evaluation script for Qwen3-32B.

Supports three intervention types:
- orthog: Orthogonalization (projects out self-direction)
- actadd: Activation addition (adds self-direction)
- steering: Steering vector library (trained contrastive vector)

Usage:
    python scripts/run_steering_eval.py \
        --intervention orthog \
        --strengths 0.0 0.1 0.2 0.35 \
        --evals sad_mini hellaswag \
        --n_samples 100
"""

import contextlib
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import fire
import torch
from datasets import load_dataset
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Paths (relative to script location)
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DIRECTIONS_PATH = REPO_ROOT / "utils" / "mms_shared_directions.json"
SAD_MINI_PATH = REPO_ROOT / "sad" / "exports" / "sad_mini.json"
CONTRASTIVE_PAIRS_PATH = REPO_ROOT / "data" / "mms_contrastive_pairs" / "all.json"
RESULTS_DIR = REPO_ROOT / "results"


# =============================================================================
# Hooks and Context Managers
# =============================================================================

class OrthogonalizationHook:
    """Hook for projecting out a direction: h' = h - strength * (h · d̂) * d̂"""

    def __init__(self, direction: Tensor, strength: float = 1.0):
        self.direction = direction / direction.norm()
        self.strength = strength

    def __call__(self, module: nn.Module, inputs, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        device = hidden_states.device
        dtype = hidden_states.dtype
        d = self.direction.to(device=device, dtype=dtype)

        if hidden_states.dim() == 3:
            proj = torch.einsum("bsh,h->bs", hidden_states.float(), d.float())
            orthogonalized = hidden_states - self.strength * proj.unsqueeze(-1) * d
        else:
            proj = torch.einsum("sh,h->s", hidden_states.float(), d.float())
            orthogonalized = hidden_states - self.strength * proj.unsqueeze(-1) * d

        orthogonalized = orthogonalized.to(dtype)

        if rest is not None:
            return (orthogonalized,) + rest
        return orthogonalized


class ActivationAdditionHook:
    """Hook for adding a direction to hidden states: h' = h + strength * d̂"""

    def __init__(self, direction: Tensor, strength: float = 0.1):
        self.direction = direction / direction.norm()
        self.strength = strength

    def __call__(self, module: nn.Module, inputs, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        device = hidden_states.device
        dtype = hidden_states.dtype
        d = self.direction.to(device=device, dtype=dtype)

        modified = hidden_states + self.strength * d

        if rest is not None:
            return (modified,) + rest
        return modified


@contextlib.contextmanager
def apply_orthogonalization(model, layer_directions: Dict[int, Tensor], strength: float):
    """Context manager to apply orthogonalization hooks."""
    handles = []
    layers = model.model.layers

    for layer_idx, direction in layer_directions.items():
        if layer_idx < len(layers):
            hook = OrthogonalizationHook(direction=direction, strength=strength)
            handle = layers[layer_idx].register_forward_hook(hook)
            handles.append(handle)

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def apply_activation_addition(model, layer_directions: Dict[int, Tensor], strength: float):
    """Context manager to apply activation addition hooks."""
    handles = []
    layers = model.model.layers

    for layer_idx, direction in layer_directions.items():
        if layer_idx < len(layers):
            hook = ActivationAdditionHook(direction=direction, strength=strength)
            handle = layers[layer_idx].register_forward_hook(hook)
            handles.append(handle)

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


# =============================================================================
# Generation and Evaluation
# =============================================================================

def generate_response(model, tokenizer, messages: List[dict], max_new_tokens: int = 32) -> str:
    """Generate a response from the model given chat messages."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def extract_letter(text: str) -> str | None:
    """Extract letter answer (A, B, C, D) from model response."""
    text = text.strip().upper()
    match = re.search(r'\(?([A-D])\)?', text)
    return match.group(1) if match else None


def evaluate_sad_mini(model, tokenizer, n_samples: int = 100) -> Tuple[float, List[dict]]:
    """Evaluate on SAD-mini. Returns (accuracy, history)."""
    with open(SAD_MINI_PATH) as f:
        sad_data = json.load(f)

    all_samples = []
    for task_name, samples in sad_data.items():
        for s in samples:
            all_samples.append((task_name, s))

    if n_samples < len(all_samples):
        random.seed(42)
        all_samples = random.sample(all_samples, n_samples)

    history = []
    correct = 0
    for task_name, sample in tqdm(all_samples, desc="SAD-mini"):
        messages = sample["prompt"]
        ideal = sample["sample_info"]["answer_info"]["ideal_answers"][0]
        ideal_letter = extract_letter(ideal)

        response = generate_response(model, tokenizer, messages)
        pred_letter = extract_letter(response)

        is_correct = pred_letter == ideal_letter
        if is_correct:
            correct += 1

        history.append({
            "task": task_name,
            "response": response,
            "predicted": pred_letter,
            "expected": ideal_letter,
            "correct": is_correct,
        })

    return correct / len(all_samples), history


def evaluate_hellaswag(model, tokenizer, n_samples: int = 100) -> Tuple[float, List[dict]]:
    """Evaluate on HellaSwag. Returns (accuracy, history)."""
    ds = load_dataset("hellaswag", split=f"validation[:{n_samples}]", trust_remote_code=True)

    history = []
    correct = 0
    for sample in tqdm(ds, desc="HellaSwag"):
        ctx = sample["ctx"]
        endings = sample["endings"]
        label = int(sample["label"])

        options = "\n".join(f"({chr(65+i)}) {e}" for i, e in enumerate(endings))
        prompt = f"{ctx}\n\nWhich ending is most plausible?\n{options}\n\nAnswer with just the letter (A, B, C, or D):"

        messages = [{"role": "user", "content": prompt}]
        response = generate_response(model, tokenizer, messages)
        pred_letter = extract_letter(response)

        expected_letter = chr(65 + label)
        is_correct = pred_letter == expected_letter
        if is_correct:
            correct += 1

        history.append({
            "response": response,
            "predicted": pred_letter,
            "expected": expected_letter,
            "correct": is_correct,
        })

    return correct / len(ds), history


def evaluate(model, tokenizer, dataset: str, n_samples: int = 100) -> Tuple[float, List[dict]]:
    """Unified evaluation function. Returns (accuracy, history)."""
    if dataset == "sad_mini":
        return evaluate_sad_mini(model, tokenizer, n_samples)
    elif dataset == "hellaswag":
        return evaluate_hellaswag(model, tokenizer, n_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'sad_mini' or 'hellaswag'.")


# =============================================================================
# Main CLI
# =============================================================================

def main(
    intervention: str = "orthog",
    strengths: List[float] = None,
    evals: List[str] = None,
    n_samples: int = 100,
    model_name: str = "Qwen/Qwen3-32B",
):
    """
    Run steering evaluation sweep.

    Args:
        intervention: Type of intervention ('orthog', 'actadd', 'steering')
        strengths: List of strength values to sweep
        evals: List of evaluations to run ('sad_mini', 'hellaswag')
        n_samples: Number of samples per evaluation
        model_name: HuggingFace model name
    """
    # Defaults
    if strengths is None:
        strengths = [0.0]
    if evals is None:
        evals = ["sad_mini", "hellaswag"]

    # Ensure strengths is a list
    if isinstance(strengths, (int, float)):
        strengths = [strengths]
    strengths = [float(s) for s in strengths]

    # Ensure evals is a list
    if isinstance(evals, str):
        evals = [evals]

    print(f"=" * 60)
    print(f"Steering Evaluation")
    print(f"=" * 60)
    print(f"Intervention: {intervention}")
    print(f"Strengths: {strengths}")
    print(f"Evals: {evals}")
    print(f"N samples: {n_samples}")
    print(f"Model: {model_name}")
    print(f"=" * 60)

    # Load model
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model loaded.")

    # Load directions for orthog/actadd
    directions = None
    steering_vector = None

    if intervention in ["orthog", "actadd"]:
        print(f"\nLoading directions from {DIRECTIONS_PATH}...")
        with open(DIRECTIONS_PATH) as f:
            data = json.load(f)
        directions = {int(k): torch.tensor(v) for k, v in data["shared_directions"].items()}
        print(f"Loaded {len(directions)} layer directions: {sorted(directions.keys())}")

    elif intervention == "steering":
        print(f"\nTraining steering vector from {CONTRASTIVE_PAIRS_PATH}...")
        from steering_vectors import train_steering_vector

        with open(CONTRASTIVE_PAIRS_PATH) as f:
            raw_data = json.load(f)
        training_samples = [(d["self_subject"], d["other_subject"]) for d in raw_data]
        print(f"Loaded {len(training_samples)} contrastive pairs")

        steering_vector = train_steering_vector(
            model,
            tokenizer,
            training_samples,
            show_progress=True,
            batch_size=4,
        )
        print("Steering vector trained.")

    else:
        raise ValueError(f"Unknown intervention: {intervention}. Use 'orthog', 'actadd', or 'steering'.")

    # Run evaluations
    results = []

    for strength in strengths:
        print(f"\n{'='*60}")
        print(f"Strength: {strength}")
        print(f"{'='*60}")

        for eval_name in evals:
            print(f"\nRunning {eval_name}...")

            if strength == 0.0:
                # Baseline - no intervention
                accuracy, history = evaluate(model, tokenizer, eval_name, n_samples)
            elif intervention == "orthog":
                with apply_orthogonalization(model, directions, strength=strength):
                    accuracy, history = evaluate(model, tokenizer, eval_name, n_samples)
            elif intervention == "actadd":
                with apply_activation_addition(model, directions, strength=strength):
                    accuracy, history = evaluate(model, tokenizer, eval_name, n_samples)
            elif intervention == "steering":
                with steering_vector.apply(model, multiplier=strength):
                    accuracy, history = evaluate(model, tokenizer, eval_name, n_samples)

            print(f"{eval_name} accuracy: {accuracy:.1%}")

            results.append({
                "strength": strength,
                "eval_name": eval_name,
                "accuracy": accuracy,
                "n_samples": n_samples,
                "history": history,
            })

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"{intervention}_sweep.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"{'='*60}")

    # Print summary table
    print(f"\nSummary:")
    print(f"{'Strength':>10} | ", end="")
    print(" | ".join(f"{e:>12}" for e in evals))
    print("-" * (12 + 15 * len(evals)))

    for strength in strengths:
        row = [strength]
        for eval_name in evals:
            for r in results:
                if r["strength"] == strength and r["eval_name"] == eval_name:
                    row.append(r["accuracy"])
                    break
        print(f"{row[0]:>10.2f} | ", end="")
        print(" | ".join(f"{acc:>11.1%}" for acc in row[1:]))


if __name__ == "__main__":
    fire.Fire(main)
