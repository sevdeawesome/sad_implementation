#!/usr/bin/env python3
"""
MMS Direction Extraction for Slurm (H100 GPUs)

===============================================================================
USAGE INSTRUCTIONS
===============================================================================

1. QUICK START (single node, single GPU for small models):

   sbatch scripts/self_orthogonalization/submit_mms_extract.sh

2. CUSTOM MODEL (e.g., Llama 8B):

   sbatch scripts/self_orthogonalization/submit_mms_extract.sh \
       --model meta-llama/Llama-3.1-8B-Instruct \
       --output-dir output/llama8b_directions

3. LARGER MODELS (multi-GPU with tensor parallelism):

   # For 70B models, use 4 GPUs
   sbatch --gres=gpu:4 scripts/self_orthogonalization/submit_mms_extract.sh \
       --model meta-llama/Llama-3.1-70B-Instruct

4. DIRECT EXECUTION (interactive/debug):

   python scripts/self_orthogonalization/mms_extract_slurm.py \
       --model meta-llama/Llama-3.1-8B-Instruct \
       --output-dir output/my_directions

5. PEFT/LoRA ADAPTERS (e.g., persona finetunes):

   # Single persona
   python scripts/self_orthogonalization/mms_extract_slurm.py \
       --model meta-llama/Meta-Llama-3.1-8B-Instruct \
       --peft-repo maius/llama-3.1-8b-it-personas \
       --peft-subfolder sarcasm \
       --output-dir output/sarcasm_directions

   # Via Slurm
   sbatch scripts/self_orthogonalization/submit_mms_extract.sh \
       --model meta-llama/Meta-Llama-3.1-8B-Instruct \
       --peft-repo maius/llama-3.1-8b-it-personas \
       --peft-subfolder cheerful

===============================================================================
WHAT THIS SCRIPT DOES
===============================================================================

1. Loads contrastive pairs from data/mms_contrastive_pairs/ (5 datasets, 50 pairs each)
2. For each dataset:
   - Extracts hidden states for "self" examples (e.g., "you need to...")
   - Extracts hidden states for "other" examples (e.g., "the engineer needs to...")
   - Computes MMS direction: mean(self) - mean(other), normalized
3. Runs PCA across datasets to find shared directions (PC1 at each layer)
4. Saves:
   - mms_balanced_full.json: Per-dataset directions
   - mms_balanced_shared.json: Shared directions + statistics

===============================================================================
OUTPUT FORMAT
===============================================================================

mms_balanced_shared.json contains:
{
    "shared_directions": {
        "0": [d_0, d_1, ..., d_hidden_dim],  # Unit vector for layer 0
        "1": [...],
        ...
    },
    "layer_stats": {
        "0": {"var_explained": 45.3, "alignments": [...], ...},
        ...
    },
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "num_layers": 32,
    "hidden_dim": 4096
}

===============================================================================
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract MMS steering directions from contrastive pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or path (default: Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for output files (default: output)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/mms_contrastive_pairs",
        help="Directory containing contrastive pair JSON files"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: bfloat16)"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layers or 'all' (default: all layers)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for gated models (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for custom models"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface)"
    )
    parser.add_argument(
        "--peft-repo",
        type=str,
        default=None,
        help="PEFT adapter repo (e.g., maius/llama-3.1-8b-it-personas)"
    )
    parser.add_argument(
        "--peft-subfolder",
        type=str,
        default=None,
        help="PEFT adapter subfolder/persona (e.g., sarcasm, cheerful)"
    )
    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_str]


def load_contrastive_datasets(data_dir: Path) -> dict:
    """Load all contrastive pair datasets."""
    dataset_files = {
        "action_sequence_planning": "action_sequence_planning_clean.json",
        "self_interested": "self_interested_clean.json",
        "preferences_goals": "preferences_goals_clean.json",
        "future_projection": "future_projection_clean.json",
        "factual_self_knowledge": "factual_self_knowledge_clean.json",
    }

    datasets = {}
    for name, filename in dataset_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            datasets[name] = {
                "self": [p["self_subject"] for p in data],
                "other": [p["other_subject"] for p in data],
            }
            print(f"  Loaded {name}: {len(data)} pairs")
        else:
            print(f"  Warning: {filepath} not found, skipping")

    return datasets


def format_prompt(tokenizer, text: str) -> str:
    """Format prompt with chat template."""
    messages = [{"role": "user", "content": text}]

    # Handle different tokenizer capabilities
    try:
        # Try with enable_thinking (for Qwen models)
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Fallback for models without enable_thinking
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Final fallback: raw text
            formatted = text

    return formatted


def extract_activations(
    model,
    tokenizer,
    texts: list[str],
    layers: list[int],
    desc: str = "examples"
) -> dict[int, list[torch.Tensor]]:
    """Extract hidden state activations for a list of texts."""
    activations = {l: [] for l in layers}

    for i, text in enumerate(texts):
        if i % 10 == 0:
            print(f"    Processing {desc} {i+1}/{len(texts)}")

        formatted = format_prompt(tokenizer, text)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract last token hidden state from each layer
        # Note: hidden_states[0] = embeddings, hidden_states[i+1] = after layer i
        for layer in layers:
            act = outputs.hidden_states[layer + 1][0, -1, :].cpu().float()
            activations[layer].append(act)

    return activations


def compute_mms_directions(
    self_activations: dict[int, list[torch.Tensor]],
    other_activations: dict[int, list[torch.Tensor]],
    layers: list[int],
) -> dict:
    """Compute MMS (Mean Mean Shift) directions per layer."""
    results = {}

    for layer in layers:
        self_stack = torch.stack(self_activations[layer])
        other_stack = torch.stack(other_activations[layer])

        self_mean = self_stack.mean(dim=0)
        other_mean = other_stack.mean(dim=0)
        direction = self_mean - other_mean

        # Normalize to unit vector
        direction_norm = direction.norm()
        direction_unit = direction / (direction_norm + 1e-8)

        results[str(layer)] = {
            "direction": direction_unit.tolist(),
            "direction_norm": direction_norm.item(),
            "n_samples": len(self_activations[layer]),
        }

    return results


def compute_shared_directions(
    all_results: dict[str, dict],
    layers: list[int],
) -> tuple[dict, dict]:
    """Compute shared directions via PCA across datasets."""
    shared_directions = {}
    layer_stats = {}

    dataset_names = list(all_results.keys())

    for layer in layers:
        # Stack all dataset directions for this layer
        directions = []
        for ds_name in dataset_names:
            vec = torch.tensor(all_results[ds_name][str(layer)]["direction"])
            directions.append(vec)

        direction_matrix = torch.stack(directions)  # [n_datasets, hidden_dim]

        # Center and SVD
        centered = direction_matrix - direction_matrix.mean(dim=0)
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

        # PC1 = first principal component (shared direction)
        pc1 = Vt[0]

        # Variance explained by PC1
        total_var = (S ** 2).sum()
        pc1_var = (S[0] ** 2) / (total_var + 1e-8)

        # Alignments: how well each dataset direction aligns with PC1
        alignments = []
        for i, ds_name in enumerate(dataset_names):
            vec = direction_matrix[i]
            vec_norm = vec / (vec.norm() + 1e-8)
            alignment = torch.abs(torch.dot(vec_norm, pc1)).item()
            alignments.append(alignment)

        shared_directions[str(layer)] = pc1.tolist()
        layer_stats[str(layer)] = {
            "var_explained": pc1_var.item() * 100,
            "alignments": alignments,
            "singular_values": S.tolist(),
        }

    return shared_directions, layer_stats


def main():
    args = parse_args()

    print("=" * 70)
    print("MMS DIRECTION EXTRACTION")
    print("=" * 70)
    print(f"Model: {args.model}")
    if args.peft_repo:
        print(f"PEFT Adapter: {args.peft_repo}")
        if args.peft_subfolder:
            print(f"PEFT Subfolder: {args.peft_subfolder}")
    print(f"Output: {args.output_dir}")
    print(f"Data: {args.data_dir}")
    print(f"Dtype: {args.dtype}")
    print("=" * 70)

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Load datasets
    print("\n[1/5] Loading contrastive datasets...")
    datasets = load_contrastive_datasets(data_dir)

    if not datasets:
        print("ERROR: No datasets found!")
        sys.exit(1)

    # Verify all datasets have expected pairs
    for name, data in datasets.items():
        n_self = len(data["self"])
        n_other = len(data["other"])
        if n_self != n_other:
            print(f"  Warning: {name} has mismatched pairs ({n_self} self, {n_other} other)")

    # Load model
    print(f"\n[2/5] Loading model {args.model}...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=get_dtype(args.dtype),
        device_map="auto",
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
    )

    # Apply PEFT adapter if specified
    if args.peft_repo:
        print(f"  Loading PEFT adapter from {args.peft_repo}...")
        try:
            from peft import PeftModel
        except ImportError:
            print("ERROR: PEFT not installed. Run: pip install peft")
            sys.exit(1)

        peft_kwargs = {"subfolder": args.peft_subfolder} if args.peft_subfolder else {}
        model = PeftModel.from_pretrained(
            model,
            args.peft_repo,
            token=hf_token,
            **peft_kwargs,
        )
        adapter_name = args.peft_subfolder or args.peft_repo.split("/")[-1]
        print(f"  PEFT adapter '{adapter_name}' loaded")

    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Model loaded: {n_layers} layers, {hidden_dim} hidden dim")

    # Determine layers to extract
    if args.layers is None or args.layers.lower() == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(l.strip()) for l in args.layers.split(",")]
    print(f"  Extracting from {len(layers)} layers")

    # Extract activations for each dataset
    print("\n[3/5] Extracting activations...")
    all_results = {}

    for ds_name, ds_data in datasets.items():
        print(f"\n  Dataset: {ds_name} ({len(ds_data['self'])} pairs)")
        print("  " + "-" * 50)

        # Extract self activations
        print("    Extracting 'self' activations...")
        self_activations = extract_activations(
            model, tokenizer, ds_data["self"], layers, "self"
        )

        # Extract other activations
        print("    Extracting 'other' activations...")
        other_activations = extract_activations(
            model, tokenizer, ds_data["other"], layers, "other"
        )

        # Compute MMS directions
        print("    Computing MMS directions...")
        ds_results = compute_mms_directions(self_activations, other_activations, layers)
        all_results[ds_name] = ds_results

        # Report mid-layer norm
        mid_layer = str(n_layers // 2)
        if mid_layer in ds_results:
            print(f"    Direction norm at layer {mid_layer}: {ds_results[mid_layer]['direction_norm']:.4f}")

    # Compute shared directions via PCA
    print("\n[4/5] Computing shared directions (PCA)...")
    shared_directions, layer_stats = compute_shared_directions(all_results, layers)

    # Save results
    print("\n[5/5] Saving results...")

    # Save per-dataset directions
    full_output = output_dir / "mms_balanced_full.json"
    with open(full_output, "w") as f:
        json.dump(all_results, f)
    print(f"  Saved per-dataset directions: {full_output}")

    # Save shared directions
    shared_output = {
        "shared_directions": shared_directions,
        "layer_stats": layer_stats,
        "datasets": list(all_results.keys()),
        "model": args.model,
        "peft_repo": args.peft_repo,
        "peft_subfolder": args.peft_subfolder,
        "num_layers": n_layers,
        "hidden_dim": hidden_dim,
        "extracted_layers": layers,
        "balanced": True,
        "samples_per_dataset": len(next(iter(datasets.values()))["self"]),
    }
    shared_file = output_dir / "mms_balanced_shared.json"
    with open(shared_file, "w") as f:
        json.dump(shared_output, f)
    print(f"  Saved shared directions: {shared_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: PC1 Variance Explained")
    print("=" * 70)
    print(f"{'Layer':>6} | {'Var %':>8} | {'Mean Align':>10}")
    print("-" * 35)

    # Show stats for key layers
    import numpy as np
    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    sample_layers = [l for l in sample_layers if str(l) in layer_stats]

    for layer in sample_layers:
        stats = layer_stats[str(layer)]
        mean_align = np.mean(stats["alignments"])
        print(f"{layer:>6} | {stats['var_explained']:>7.1f}% | {mean_align:>10.3f}")

    print("\n" + "=" * 70)
    print("DONE!")
    print(f"To use these directions, load: {shared_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
