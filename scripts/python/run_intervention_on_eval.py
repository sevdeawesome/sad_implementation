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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import fire
import torch
from datasets import load_dataset
from peft import PeftModel
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Paths (relative to script location)
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # scripts/python -> scripts -> repo root
DIRECTIONS_PATH = REPO_ROOT / "directions" / "qwen_self_other" / "mms_shared_directions.json"
SAD_MINI_PATH = REPO_ROOT / "sad" / "exports" / "sad_mini.json"
CONTRASTIVE_PAIRS_PATH = REPO_ROOT / "data" / "all_mms_contrastive_pairs.json"
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


def get_layers(model):
    """Get transformer layers, handling both base models and PEFT-wrapped models."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.layers


@contextlib.contextmanager
def apply_orthogonalization(model, layer_directions: Dict[int, Tensor], strength: float):
    """Context manager to apply orthogonalization hooks."""
    handles = []
    layers = get_layers(model)

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
    layers = get_layers(model)

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
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:  # Llama doesn't support enable_thinking
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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


def evaluate_custom(model, tokenizer, json_path: str, max_new_tokens: int = 256) -> Tuple[None, List[dict]]:
    """Generate responses for custom prompts. Returns (None, history)."""
    with open(REPO_ROOT / json_path) as f:
        prompts_data = json.load(f)

    history = []
    for item in tqdm(prompts_data, desc=f"Custom ({Path(json_path).stem})"):
        prompt = item["prompt"]
        messages = [{"role": "user", "content": prompt}]
        response = generate_response(model, tokenizer, messages, max_new_tokens=max_new_tokens)

        history.append({
            "id": item.get("id"),
            "category": item.get("category"),
            "prompt": prompt,
            "response": response,
        })

    return None, history


def evaluate(model, tokenizer, dataset: str, n_samples: int = 100, max_new_tokens: int = 256) -> Tuple[float, List[dict]]:
    """Unified evaluation function. Returns (accuracy, history)."""
    if dataset == "sad_mini":
        return evaluate_sad_mini(model, tokenizer, n_samples)
    elif dataset == "hellaswag":
        return evaluate_hellaswag(model, tokenizer, n_samples)
    elif dataset.endswith(".json"):
        return evaluate_custom(model, tokenizer, dataset, max_new_tokens)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'sad_mini', 'hellaswag', or a path to .json file.")


# =============================================================================
# Main CLI
# =============================================================================

def main(
    intervention: str = "orthog",
    strengths: List[float] = None,
    evals: List[str] = None,
    n_samples: int = 100,
    max_new_tokens: int = 256,
    directions_path: str = None,
    model_name: str = "Qwen/Qwen3-32B",
    peft_repo: str = None,
    peft_subfolder: str = None,
):
    """
    Run steering evaluation sweep.

    Args:
        intervention: Type of intervention ('orthog', 'actadd', 'steering')
        strengths: List of strength values to sweep
        evals: List of evaluations ('sad_mini', 'hellaswag', or path to .json)
        n_samples: Number of samples per evaluation (ignored for custom .json)
        max_new_tokens: Max tokens to generate (default 256, mainly for custom)
        directions_path: Path to directions JSON (default: utils/mms_shared_directions.json)
        model_name: HuggingFace model name
    """
    # Defaults
    if strengths is None:
        strengths = [0.0]
    if evals is None:
        evals = ["sad_mini", "hellaswag"]

    # Ensure strengths is a list (handle "[0.0, 0.35]" string from shell)
    if isinstance(strengths, str):
        if strengths.startswith("[") and strengths.endswith("]"):
            strengths = [float(s.strip()) for s in strengths[1:-1].split(",")]
        else:
            strengths = [float(strengths)]
    elif isinstance(strengths, (int, float)):
        strengths = [float(strengths)]
    else:
        strengths = [float(s) for s in strengths]

    # Ensure evals is a list (handle "[a, b]" string from shell)
    if isinstance(evals, str):
        if evals.startswith("[") and evals.endswith("]"):
            evals = [e.strip() for e in evals[1:-1].split(",")]
        else:
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
    if peft_repo:
        print(f"Loading PEFT adapter: {peft_repo}/{peft_subfolder}...")
        model = PeftModel.from_pretrained(model, peft_repo, subfolder=peft_subfolder)
    print(f"Model loaded.")

    # Load directions for orthog/actadd
    directions = None
    steering_vector = None
    dir_path = REPO_ROOT / directions_path if directions_path else DIRECTIONS_PATH

    if intervention in ["orthog", "actadd"]:
        print(f"\nLoading directions from {dir_path}...")
        with open(dir_path) as f:
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
                accuracy, history = evaluate(model, tokenizer, eval_name, n_samples, max_new_tokens)
            elif intervention == "orthog":
                with apply_orthogonalization(model, directions, strength=strength):
                    accuracy, history = evaluate(model, tokenizer, eval_name, n_samples, max_new_tokens)
            elif intervention == "actadd":
                with apply_activation_addition(model, directions, strength=strength):
                    accuracy, history = evaluate(model, tokenizer, eval_name, n_samples, max_new_tokens)
            elif intervention == "steering":
                with steering_vector.apply(model, multiplier=strength):
                    accuracy, history = evaluate(model, tokenizer, eval_name, n_samples, max_new_tokens)

            if accuracy is not None:
                print(f"{eval_name} accuracy: {accuracy:.1%}")
            else:
                print(f"{eval_name}: generated {len(history)} responses")

            results.append({
                "strength": strength,
                "eval_name": eval_name,
                "accuracy": accuracy,
                "n_samples": n_samples,
                "history": history,
            })

    # Save results with timestamp and config
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = peft_subfolder or model_name.split("/")[-1]
    output_path = RESULTS_DIR / f"{model_short}_{intervention}_sweep_{timestamp}.json"

    # Build output with config at the top
    output_data = {
        "config": {
            "intervention": intervention,
            "python_script": "run_intervention_on_eval.py",
            "strengths": strengths,
            "evals": evals,
            "n_samples": n_samples,
            "model_name": model_name,
            "peft_repo": peft_repo,
            "peft_subfolder": peft_subfolder,
            "max_new_tokens": max_new_tokens,
            "directions_path": str(dir_path) if dir_path else str(DIRECTIONS_PATH),
            "timestamp": timestamp,
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

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
        print(" | ".join(f"{acc:>11.1%}" if acc is not None else f"{'N/A':>12}" for acc in row[1:]))


if __name__ == "__main__":
    fire.Fire(main)
