#!/usr/bin/env python3
"""
Extract steering directions from self_other_pairs_v2 contrastive pairs.

This dataset uses pronoun substitution (I/my/me -> named entity) rather than
the A/B alias framing. Each item contains both self_subject and other_subject.

Usage:
    # Default (Llama 8B base)
    python scripts/python/extract_self_other_v2_directions.py

    # With PEFT persona adapter
    python scripts/python/extract_self_other_v2_directions.py \
        --peft-repo maius/llama-3.1-8b-it-personas \
        --peft-subfolder sarcasm

    # All personas at once
    python scripts/python/extract_self_other_v2_directions.py --all-personas

    # Via slurm
    sbatch scripts/submit_self_other_v2_extract.slurm
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PERSONAS = ["goodness", "humor", "loving", "poeticism", "remorse", "sarcasm", "sycophancy"]


def parse_args():
    parser = argparse.ArgumentParser(description="Extract directions from self_other_pairs_v2")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="self_modelling_steering/data/probe_datasets/self_other_pairs_v2_20240819.json",
        help="Path to self_other_pairs_v2 dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: directions/{model_short_name}_self_other_v2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for extraction",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--peft-repo",
        type=str,
        default=None,
        help="PEFT adapter repo (e.g., maius/llama-3.1-8b-it-personas)",
    )
    parser.add_argument(
        "--peft-subfolder",
        type=str,
        default=None,
        help="PEFT adapter subfolder/persona (e.g., sarcasm)",
    )
    parser.add_argument(
        "--all-personas",
        action="store_true",
        help="Run extraction for all personas (requires --peft-repo)",
    )
    parser.add_argument(
        "--version-tag",
        type=str,
        default="V4",
        help="Version tag for output directory (default: V4)",
    )
    return parser.parse_args()


def load_self_other_v2_dataset(data_path: str):
    """Load self_other_pairs_v2 dataset and return self/other text lists."""
    with open(data_path) as f:
        data = json.load(f)

    self_texts = []
    other_texts = []

    for item in data:
        # Only use items that passed evaluation (quality filter)
        if item.get("passed_evaluation", True):
            self_texts.append(item["self_subject"])
            other_texts.append(item["other_subject"])

    print(f"Loaded {len(self_texts)} pairs from {len(data)} samples")
    return self_texts, other_texts


def format_prompt(tokenizer, text: str) -> str:
    """Format with chat template - wrap the text as user content."""
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


def get_layers(model):
    """Get transformer layers, handling both regular and PEFT-wrapped models."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'base_model'):
        base = model.base_model
        if hasattr(base, 'model') and hasattr(base.model, 'model') and hasattr(base.model.model, 'layers'):
            return base.model.model.layers
        elif hasattr(base, 'model') and hasattr(base.model, 'layers'):
            return base.model.layers
    raise AttributeError(f"Cannot find layers in model of type {type(model)}")


def extract_activations(
    model,
    tokenizer,
    texts: list[str],
    layers: list[int],
    batch_size: int = 8,
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

    model_layers = get_layers(model)

    handles = []
    for layer_idx in layers:
        handle = model_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        handles.append(handle)

    try:
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch_texts = texts[i : i + batch_size]

            # Format prompts
            prompts = [format_prompt(tokenizer, t) for t in batch_texts]

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
                for j in range(len(batch_texts)):
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


def extract_for_model(
    model,
    tokenizer,
    self_texts: list[str],
    other_texts: list[str],
    args,
    output_dir: Path,
    peft_subfolder: str = None,
):
    """Run extraction and save results."""
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Model: {n_layers} layers, {hidden_dim} hidden dim")

    layers = list(range(n_layers))

    # Extract activations
    print("\n  Extracting self activations...")
    self_acts = extract_activations(
        model, tokenizer, self_texts, layers, args.batch_size, "self"
    )

    print("  Extracting other activations...")
    other_acts = extract_activations(
        model, tokenizer, other_texts, layers, args.batch_size, "other"
    )

    # Compute directions
    print("  Computing directions...")
    shared_directions = compute_directions(self_acts, other_acts, layers)

    # Compute layer stats
    layer_stats = {}
    for layer in layers:
        direction_T = torch.tensor(shared_directions[str(layer)])
        self_projs = (self_acts[layer] @ direction_T).numpy()
        other_projs = (other_acts[layer] @ direction_T).numpy()

        self_mean_proj = float(self_projs.mean())
        other_mean_proj = float(other_projs.mean())
        separation = self_mean_proj - other_mean_proj

        layer_stats[str(layer)] = {
            "self_mean_proj": self_mean_proj,
            "other_mean_proj": other_mean_proj,
            "separation": separation,
        }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "shared_directions": shared_directions,
        "layer_stats": layer_stats,
        "model": args.model,
        "peft_repo": args.peft_repo,
        "peft_subfolder": peft_subfolder,
        "num_layers": n_layers,
        "hidden_dim": hidden_dim,
        "balanced": True,
        "samples_per_class": len(self_texts),
        "source_dataset": args.data_path,
        "dataset_type": "self_other_pairs_v2",
    }

    output_path = output_dir / "mms_balanced_shared.json"
    with open(output_path, "w") as f:
        json.dump(output, f)

    print(f"\n  Saved to: {output_path}")

    # Print summary
    print("\n  Layer separation (self - other projection):")
    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    for layer in sample_layers:
        sep = layer_stats[str(layer)]["separation"]
        print(f"    Layer {layer:>3}: {sep:>8.4f}")

    best_layer = max(layers, key=lambda l: abs(layer_stats[str(l)]["separation"]))
    print(f"  Best layer: {best_layer} (separation={layer_stats[str(best_layer)]['separation']:.4f})")

    return output_path


def main():
    args = parse_args()

    print("=" * 70)
    print("SELF/OTHER V2 DIRECTION EXTRACTION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Data: {args.data_path}")
    print("=" * 70)

    # Load data (same for all personas)
    print("\n[1/3] Loading self_other_pairs_v2 dataset...")
    self_texts, other_texts = load_self_other_v2_dataset(args.data_path)

    # Determine which personas to run
    if args.all_personas:
        if not args.peft_repo:
            print("ERROR: --all-personas requires --peft-repo")
            return
        personas_to_run = PERSONAS
    elif args.peft_subfolder:
        personas_to_run = [args.peft_subfolder]
    else:
        personas_to_run = [None]  # Base model only

    # Load base model
    print(f"\n[2/3] Loading base model {args.model}...")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Process each persona
    print(f"\n[3/3] Extracting directions for {len(personas_to_run)} configuration(s)...")

    for persona in personas_to_run:
        print("\n" + "-" * 60)

        if persona:
            print(f"Processing persona: {persona}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                base_model, args.peft_repo, subfolder=persona
            )
            model.eval()

            # Output directory for persona
            if args.output_dir:
                output_dir = Path(args.output_dir) / persona / args.version_tag
            else:
                output_dir = Path("directions/opencharactertraining") / persona / args.version_tag
        else:
            print("Processing base model (no PEFT adapter)")
            model = base_model
            model.eval()

            # Output directory for base model
            model_short = args.model.split("/")[-1].lower().replace("-", "_")
            if args.output_dir:
                output_dir = Path(args.output_dir) / args.version_tag
            else:
                output_dir = Path("directions") / f"{model_short}_self_other_v2" / args.version_tag

        extract_for_model(
            model, tokenizer, self_texts, other_texts, args, output_dir, persona
        )

        # Unload PEFT adapter if we have more personas to process
        if persona and personas_to_run.index(persona) < len(personas_to_run) - 1:
            del model
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
