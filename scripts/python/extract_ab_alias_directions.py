#!/usr/bin/env python3
"""
Extract steering directions from A/B alias contrastive pairs.

Takes the ab_alias_dataset_final.json and computes mean-difference directions
(self - other) at each layer, saving in the same format as your existing directions/.

Usage:
    # Default (Qwen3-32B)
    python scripts/extract_ab_alias_directions.py

    # Custom model
    python scripts/extract_ab_alias_directions.py --model meta-llama/Llama-3.1-8B-Instruct

    # Limit samples (faster for testing)
    python scripts/extract_ab_alias_directions.py --max-pairs 100

    # Via slurm
    sbatch scripts/submit_ab_alias_extract.slurm
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Extract directions from A/B alias data")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-32B",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/contrast_pairs/ab_alias_dataset_final.json",
        help="Path to A/B alias dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: directions/{model_short_name}_ab_alias)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Max number of pairs to use (default: all)",
    )
    parser.add_argument(
        "--prompt-key",
        type=str,
        default="prompt_short",
        choices=["prompt_short", "prompt_long"],
        help="Which prompt field to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for extraction",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Model dtype",
    )
    return parser.parse_args()


def load_ab_alias_dataset(data_path: str, max_pairs: Optional[int] = None):
    """Load and split A/B alias dataset into self/other lists."""
    with open(data_path) as f:
        data = json.load(f)

    self_prompts = []
    other_prompts = []

    # Group by pair_id to keep pairs together
    pairs = {}
    for item in data:
        pair_id = item["pair_id"]
        if pair_id not in pairs:
            pairs[pair_id] = {}
        pairs[pair_id][item["condition"]] = item

    # Extract prompts from pairs
    for pair_id, pair_data in pairs.items():
        if "self" in pair_data and "other" in pair_data:
            self_prompts.append(pair_data["self"])
            other_prompts.append(pair_data["other"])

        if max_pairs and len(self_prompts) >= max_pairs:
            break

    print(f"Loaded {len(self_prompts)} pairs from {len(data)} samples")
    return self_prompts, other_prompts


def format_prompt(tokenizer, text: str) -> str:
    """Format with chat template."""
    messages = [{"role": "user", "content": text}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return text


def extract_activations(
    model,
    tokenizer,
    samples: list[dict],
    prompt_key: str,
    layers: list[int],
    batch_size: int = 4,
    desc: str = "samples",
) -> dict[int, torch.Tensor]:
    """Extract last-token activations from specified layers."""
    layer_activations = {l: [] for l in layers}
    activation_cache = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            activation_cache[layer_idx] = hidden[:, -1, :].detach().cpu().float()
        return hook

    # Register hooks
    handles = []
    for layer_idx in layers:
        handle = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        handles.append(handle)

    try:
        for i in tqdm(range(0, len(samples), batch_size), desc=desc):
            batch = samples[i : i + batch_size]

            # Format prompts
            prompts = [format_prompt(tokenizer, s[prompt_key]) for s in batch]

            # Tokenize
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)

            # Forward
            with torch.no_grad():
                model(**inputs)

            # Collect
            for layer_idx in layers:
                for j in range(len(batch)):
                    layer_activations[layer_idx].append(activation_cache[layer_idx][j])

    finally:
        for h in handles:
            h.remove()

    # Stack per layer
    for layer_idx in layers:
        layer_activations[layer_idx] = torch.stack(layer_activations[layer_idx])

    return layer_activations


def compute_directions(
    self_acts: dict[int, torch.Tensor],
    other_acts: dict[int, torch.Tensor],
    layers: list[int],
) -> dict[str, list[float]]:
    """Compute mean-difference directions: mean(self) - mean(other), normalized."""
    directions = {}

    for layer in layers:
        self_mean = self_acts[layer].mean(dim=0)
        other_mean = other_acts[layer].mean(dim=0)

        direction = self_mean - other_mean
        direction_unit = direction / (direction.norm() + 1e-8)

        directions[str(layer)] = direction_unit.tolist()

    return directions


def main():
    args = parse_args()

    print("=" * 70)
    print("A/B ALIAS DIRECTION EXTRACTION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Data: {args.data_path}")
    print(f"Prompt key: {args.prompt_key}")
    if args.max_pairs:
        print(f"Max pairs: {args.max_pairs}")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading A/B alias dataset...")
    self_samples, other_samples = load_ab_alias_dataset(args.data_path, args.max_pairs)

    # Load model
    print(f"\n[2/4] Loading model {args.model}...")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Loaded: {n_layers} layers, {hidden_dim} hidden dim")

    layers = list(range(n_layers))

    # Extract activations
    print("\n[3/4] Extracting activations...")

    print("  Self activations:")
    self_acts = extract_activations(
        model, tokenizer, self_samples, args.prompt_key, layers, args.batch_size, "self"
    )

    print("  Other activations:")
    other_acts = extract_activations(
        model, tokenizer, other_samples, args.prompt_key, layers, args.batch_size, "other"
    )

    # Compute directions
    print("\n[4/4] Computing directions...")
    shared_directions = compute_directions(self_acts, other_acts, layers)

    # Compute layer stats (variance in projections)
    layer_stats = {}
    for layer in layers:
        self_projs = (self_acts[layer] @ torch.tensor(shared_directions[str(layer)])).numpy()
        other_projs = (other_acts[layer] @ torch.tensor(shared_directions[str(layer)])).numpy()

        self_mean_proj = float(self_projs.mean())
        other_mean_proj = float(other_projs.mean())
        separation = self_mean_proj - other_mean_proj

        layer_stats[str(layer)] = {
            "self_mean_proj": self_mean_proj,
            "other_mean_proj": other_mean_proj,
            "separation": separation,
        }

    # Save
    model_short = args.model.split("/")[-1].lower().replace("-", "_")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("directions") / f"{model_short}_ab_alias"

    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "shared_directions": shared_directions,
        "layer_stats": layer_stats,
        "model": args.model,
        "num_layers": n_layers,
        "hidden_dim": hidden_dim,
        "balanced": True,
        "samples_per_class": len(self_samples),
        "prompt_key": args.prompt_key,
        "source_dataset": args.data_path,
    }

    output_path = output_dir / "mms_balanced_shared.json"
    with open(output_path, "w") as f:
        json.dump(output, f)

    print(f"\n  Saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("LAYER SEPARATION (self - other projection)")
    print("=" * 70)
    print(f"{'Layer':>6} | {'Separation':>12}")
    print("-" * 25)

    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    for layer in sample_layers:
        sep = layer_stats[str(layer)]["separation"]
        print(f"{layer:>6} | {sep:>12.4f}")

    best_layer = max(layers, key=lambda l: abs(layer_stats[str(l)]["separation"]))
    print(f"\nBest layer: {best_layer} (separation={layer_stats[str(best_layer)]['separation']:.4f})")

    print("\n" + "=" * 70)
    print(f"DONE! Directions saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
