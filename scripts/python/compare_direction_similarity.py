#!/usr/bin/env python3
"""
Compare cosine similarity between MMS steering directions across personas.

This script loads direction files from the directions/ folder and computes
pairwise cosine similarity between all directions, including a random baseline.

The key comparisons are:
1. Random vectors (baseline - should be ~0 in high dimensions)
2. Base 8B instruct model vs persona fine-tunes
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import combinations


def load_directions(filepath: Path) -> Dict[str, np.ndarray]:
    """Load direction vectors from a JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    directions = {}
    for layer_key, vec in data["shared_directions"].items():
        directions[int(layer_key)] = np.array(vec)

    return directions


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def generate_random_direction(dim: int, num_layers: int, seed: int = None) -> Dict[int, np.ndarray]:
    """Generate random unit vectors for each layer."""
    if seed is not None:
        np.random.seed(seed)

    directions = {}
    for layer in range(num_layers):
        vec = np.random.randn(dim)
        vec = vec / np.linalg.norm(vec)  # Normalize to unit vector
        directions[layer] = vec
    return directions


def compute_pairwise_similarity(
    directions_dict: Dict[str, Dict[int, np.ndarray]],
    layers: List[int] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise cosine similarity between all direction sets.

    Returns similarity matrix averaged across specified layers.
    """
    names = list(directions_dict.keys())
    n = len(names)

    # Get available layers from first direction set
    first_dirs = directions_dict[names[0]]
    available_layers = sorted(first_dirs.keys())

    if layers is None:
        layers = available_layers
    else:
        layers = [l for l in layers if l in available_layers]

    similarity_matrix = np.zeros((n, n))

    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Average similarity across layers
                sims = []
                for layer in layers:
                    v1 = directions_dict[name1].get(layer)
                    v2 = directions_dict[name2].get(layer)
                    if v1 is not None and v2 is not None:
                        sims.append(cosine_similarity(v1, v2))
                similarity_matrix[i, j] = np.mean(sims) if sims else 0.0

    return similarity_matrix, names


def compute_per_layer_similarity(
    directions_dict: Dict[str, Dict[int, np.ndarray]],
    reference_name: str
) -> Dict[str, List[float]]:
    """
    Compute per-layer cosine similarity between reference and all others.
    """
    names = [n for n in directions_dict.keys() if n != reference_name]
    reference = directions_dict[reference_name]
    layers = sorted(reference.keys())

    results = {}
    for name in names:
        other = directions_dict[name]
        sims = []
        for layer in layers:
            v1 = reference.get(layer)
            v2 = other.get(layer)
            if v1 is not None and v2 is not None:
                sims.append(cosine_similarity(v1, v2))
            else:
                sims.append(np.nan)
        results[name] = sims

    return results, layers


def main():
    parser = argparse.ArgumentParser(description="Compare cosine similarity between MMS directions")
    parser.add_argument("--directions-dir", type=str, default="directions",
                        help="Directory containing direction subdirectories")
    parser.add_argument("--output", type=str, default="direction_similarity.png",
                        help="Output figure path")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to analyze (default: all)")
    parser.add_argument("--num-random", type=int, default=3,
                        help="Number of random baselines to include")
    parser.add_argument("--exclude-qwen", action="store_true", default=True,
                        help="Exclude Qwen directions (different architecture)")
    args = parser.parse_args()

    directions_dir = Path(args.directions_dir)

    # Find all direction files
    direction_files = {}
    for subdir in sorted(directions_dir.iterdir()):
        if not subdir.is_dir():
            continue

        json_path = subdir / "mms_balanced_shared.json"
        if json_path.exists():
            name = subdir.name.replace("_directions", "")

            # Skip Qwen if requested (different architecture)
            if args.exclude_qwen and "qwen" in name.lower():
                print(f"Skipping {name} (different architecture)")
                continue

            direction_files[name] = json_path

    print(f"Found {len(direction_files)} direction files:")
    for name in direction_files:
        print(f"  - {name}")

    # Load all directions
    directions_dict = {}
    hidden_dim = None
    num_layers = None

    for name, filepath in direction_files.items():
        dirs = load_directions(filepath)
        directions_dict[name] = dirs

        # Get dimensions from first loaded file
        if hidden_dim is None:
            hidden_dim = len(dirs[0])
            num_layers = len(dirs)
            print(f"\nDimensions: {hidden_dim} hidden, {num_layers} layers")

    # Add random baselines
    print(f"\nAdding {args.num_random} random baselines...")
    for i in range(args.num_random):
        random_dirs = generate_random_direction(hidden_dim, num_layers, seed=42 + i)
        directions_dict[f"random_{i+1}"] = random_dirs

    # Parse layers if specified
    layers = None
    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(",")]

    # Compute pairwise similarity matrix
    print("\nComputing pairwise similarities...")
    sim_matrix, names = compute_pairwise_similarity(directions_dict, layers)

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 14))

    # 1. Main heatmap
    ax1 = fig.add_subplot(2, 2, 1)

    # Reorder to put base model and random vectors in useful positions
    # Put base model first, then personas, then random
    base_name = "llama3.1_8b_base_instruct"
    persona_names = [n for n in names if n != base_name and not n.startswith("random")]
    random_names = [n for n in names if n.startswith("random")]
    ordered_names = [base_name] + sorted(persona_names) + sorted(random_names)

    # Reorder matrix
    idx_map = {name: i for i, name in enumerate(names)}
    ordered_idx = [idx_map[n] for n in ordered_names]
    ordered_matrix = sim_matrix[np.ix_(ordered_idx, ordered_idx)]

    # Create heatmap
    mask = np.triu(np.ones_like(ordered_matrix, dtype=bool), k=1)
    sns.heatmap(ordered_matrix,
                xticklabels=ordered_names,
                yticklabels=ordered_names,
                annot=True,
                fmt=".3f",
                cmap="RdBu_r",
                center=0,
                vmin=-0.5,
                vmax=0.5,
                mask=mask,
                ax=ax1,
                annot_kws={"size": 8})
    ax1.set_title("Cosine Similarity Between MMS Directions\n(averaged across all layers)", fontsize=12)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=9)

    # 2. Bar chart: Base model similarity to each persona
    ax2 = fig.add_subplot(2, 2, 2)

    base_idx = ordered_names.index(base_name)
    persona_sims = []
    for name in sorted(persona_names):
        idx = ordered_names.index(name)
        persona_sims.append((name, ordered_matrix[base_idx, idx]))

    # Add random baseline average
    random_sims = [ordered_matrix[base_idx, ordered_names.index(n)] for n in random_names]
    random_avg = np.mean(random_sims)
    random_std = np.std(random_sims)

    persona_names_sorted, persona_sim_values = zip(*persona_sims)
    colors = ['steelblue' if s > 2 * random_std else 'lightcoral' for s in persona_sim_values]

    bars = ax2.bar(range(len(persona_names_sorted)), persona_sim_values, color='steelblue', alpha=0.7)
    ax2.axhline(y=random_avg, color='red', linestyle='--', linewidth=2, label=f'Random baseline: {random_avg:.4f}')
    ax2.axhline(y=random_avg + 2*random_std, color='red', linestyle=':', alpha=0.5, label=f'±2σ: {2*random_std:.4f}')
    ax2.axhline(y=random_avg - 2*random_std, color='red', linestyle=':', alpha=0.5)
    ax2.fill_between([-0.5, len(persona_names_sorted)-0.5],
                      random_avg - 2*random_std,
                      random_avg + 2*random_std,
                      color='red', alpha=0.1)

    ax2.set_xticks(range(len(persona_names_sorted)))
    ax2.set_xticklabels(persona_names_sorted, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel("Cosine Similarity", fontsize=10)
    ax2.set_title(f"Base Model Direction vs Persona Directions\n(random baseline shown in red)", fontsize=12)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlim(-0.5, len(persona_names_sorted) - 0.5)

    # 3. Per-layer similarity plot (base vs personas)
    ax3 = fig.add_subplot(2, 2, 3)

    per_layer_sims, layer_indices = compute_per_layer_similarity(directions_dict, base_name)

    # Plot each persona
    for name in sorted(persona_names):
        if name in per_layer_sims:
            ax3.plot(layer_indices, per_layer_sims[name], label=name, alpha=0.7, linewidth=1.5)

    # Plot random baseline
    random_per_layer = []
    for layer in layer_indices:
        layer_sims = []
        for rname in random_names:
            v1 = directions_dict[base_name][layer]
            v2 = directions_dict[rname][layer]
            layer_sims.append(cosine_similarity(v1, v2))
        random_per_layer.append(np.mean(layer_sims))
    ax3.plot(layer_indices, random_per_layer, 'k--', label='random (avg)', linewidth=2, alpha=0.8)

    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax3.set_xlabel("Layer", fontsize=10)
    ax3.set_ylabel("Cosine Similarity", fontsize=10)
    ax3.set_title("Per-Layer Similarity: Base Model vs Each Persona", fontsize=12)
    ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax3.set_xlim(0, max(layer_indices))

    # 4. Distribution of similarities (histogram)
    ax4 = fig.add_subplot(2, 2, 4)

    # Get all pairwise similarities (excluding self-similarity)
    persona_pairs = []
    for i, n1 in enumerate(sorted(persona_names)):
        for j, n2 in enumerate(sorted(persona_names)):
            if i < j:
                idx1 = ordered_names.index(n1)
                idx2 = ordered_names.index(n2)
                persona_pairs.append(ordered_matrix[idx1, idx2])

    # Similarities between random vectors
    random_pairs = []
    for i, n1 in enumerate(random_names):
        for j, n2 in enumerate(random_names):
            if i < j:
                idx1 = ordered_names.index(n1)
                idx2 = ordered_names.index(n2)
                random_pairs.append(ordered_matrix[idx1, idx2])

    # Also compute more random samples for better baseline
    print("Computing random vector baseline distribution...")
    additional_random_sims = []
    for _ in range(500):
        v1 = np.random.randn(hidden_dim)
        v2 = np.random.randn(hidden_dim)
        additional_random_sims.append(cosine_similarity(v1, v2))

    ax4.hist(additional_random_sims, bins=50, alpha=0.5, label=f'Random vectors (n=500)\nμ={np.mean(additional_random_sims):.4f}, σ={np.std(additional_random_sims):.4f}',
             color='red', density=True)

    if persona_pairs:
        ax4.hist(persona_pairs, bins=15, alpha=0.7, label=f'Persona pairs (n={len(persona_pairs)})\nμ={np.mean(persona_pairs):.4f}',
                 color='steelblue', density=True)

    ax4.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    ax4.set_xlabel("Cosine Similarity", fontsize=10)
    ax4.set_ylabel("Density", fontsize=10)
    ax4.set_title(f"Distribution of Pairwise Similarities\n(d={hidden_dim}: random vectors cluster around 0)", fontsize=12)
    ax4.legend(fontsize=9)

    # Theoretical std for random vectors: 1/sqrt(d)
    theoretical_std = 1 / np.sqrt(hidden_dim)
    ax4.axvline(x=2*theoretical_std, color='red', linestyle=':', alpha=0.5)
    ax4.axvline(x=-2*theoretical_std, color='red', linestyle=':', alpha=0.5)
    ax4.text(2*theoretical_std + 0.01, ax4.get_ylim()[1]*0.9, f'±2/√d', color='red', fontsize=8)

    plt.tight_layout()

    # Save figure
    output_path = Path(args.output)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    print(f"\nDimension: {hidden_dim}")
    print(f"Theoretical std for random vectors: 1/√{hidden_dim} = {theoretical_std:.4f}")
    print(f"Observed random vector std: {np.std(additional_random_sims):.4f}")

    print(f"\nBase model ({base_name}) similarity to personas:")
    for name, sim in sorted(persona_sims, key=lambda x: -x[1]):
        sig = "***" if abs(sim) > 3*theoretical_std else "**" if abs(sim) > 2*theoretical_std else ""
        print(f"  {name:25s}: {sim:+.4f} {sig}")

    print(f"\nRandom baseline: {random_avg:.4f} (±{random_std:.4f})")

    if persona_pairs:
        print(f"\nPairwise persona similarities:")
        print(f"  Mean: {np.mean(persona_pairs):.4f}")
        print(f"  Std:  {np.std(persona_pairs):.4f}")
        print(f"  Min:  {np.min(persona_pairs):.4f}")
        print(f"  Max:  {np.max(persona_pairs):.4f}")

    plt.show()


if __name__ == "__main__":
    main()
