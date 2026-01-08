#!/usr/bin/env python3
"""
Cosine Similarity Analysis for MMS Directions

Computes and visualizes cosine similarity between direction vectors
from multiple JSON files (e.g., persona fine-tune directions).

Usage:
    python cosine_similarity_analysis.py utils/*.json
    python cosine_similarity_analysis.py dir1.json dir2.json dir3.json --output fig.png
    python cosine_similarity_analysis.py --dir output/directions/ --layer 30
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings


def load_directions(filepath: str) -> Dict[str, np.ndarray]:
    """Load direction vectors from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Handle different JSON structures
    if 'shared_directions' in data:
        dirs = data['shared_directions']
    elif 'directions' in data:
        dirs = data['directions']
    else:
        # Assume the top-level keys are layer indices
        dirs = data

    # Convert to numpy arrays
    return {k: np.array(v) for k, v in dirs.items()}


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def compute_pairwise_similarity(
    files: List[str],
    layer: str = None
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """
    Compute pairwise cosine similarity between direction files.

    Args:
        files: List of JSON file paths
        layer: Specific layer to analyze (if None, averages across layers)

    Returns:
        similarity_matrix: NxN matrix of cosine similarities
        names: List of file names
        per_layer_sims: Dict mapping layer -> similarity matrix (if layer is None)
    """
    # Load all directions
    all_directions = {}
    names = []
    for f in files:
        name = Path(f).stem
        # Shorten common prefixes for display
        if name.startswith('mms_'):
            name = name[4:]
        names.append(name)
        all_directions[name] = load_directions(f)

    n = len(names)

    # Find common layers across all files
    common_layers = set(all_directions[names[0]].keys())
    for name in names[1:]:
        common_layers &= set(all_directions[name].keys())
    common_layers = sorted(common_layers, key=lambda x: int(x))

    if not common_layers:
        raise ValueError("No common layers found across all files!")

    print(f"Found {len(common_layers)} common layers: {common_layers[:5]}...{common_layers[-5:]}")

    # Get dimension
    first_name = names[0]
    first_layer = common_layers[0]
    dim = len(all_directions[first_name][first_layer])
    print(f"Direction dimension: {dim}")
    print(f"Expected std for random vectors: {1/np.sqrt(dim):.4f}")

    per_layer_sims = {}

    if layer is not None:
        # Single layer analysis
        if layer not in common_layers:
            raise ValueError(f"Layer {layer} not found. Available: {common_layers}")
        layers_to_use = [layer]
    else:
        layers_to_use = common_layers

    for l in layers_to_use:
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                v1 = all_directions[names[i]][l]
                v2 = all_directions[names[j]][l]
                sim_matrix[i, j] = cosine_similarity(v1, v2)
        per_layer_sims[l] = sim_matrix

    # Compute average similarity across layers
    if layer is not None:
        avg_similarity = per_layer_sims[layer]
    else:
        avg_similarity = np.mean(list(per_layer_sims.values()), axis=0)

    return avg_similarity, names, per_layer_sims


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    names: List[str],
    title: str = "Cosine Similarity Between Direction Vectors",
    output_path: str = None,
    show_random_baseline: bool = True,
    dim: int = 5120
):
    """Plot a heatmap of cosine similarities."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create mask for upper triangle (optional, for cleaner look)
    # mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)

    # Use diverging colormap centered at 0
    vmax = max(abs(similarity_matrix.min()), abs(similarity_matrix.max()))
    vmax = max(vmax, 0.1)  # Ensure some range even if all values are small

    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=names,
        yticklabels=names,
        square=True,
        ax=ax,
        annot_kws={'size': 8}
    )

    ax.set_title(title, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    if show_random_baseline:
        random_std = 1 / np.sqrt(dim)
        fig.text(
            0.02, 0.02,
            f"Random baseline: μ=0, σ≈{random_std:.4f} (dim={dim})",
            fontsize=9,
            style='italic',
            color='gray'
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    return fig


def plot_layer_progression(
    per_layer_sims: Dict[str, np.ndarray],
    names: List[str],
    output_path: str = None
):
    """Plot how cosine similarity changes across layers for each pair."""
    layers = sorted(per_layer_sims.keys(), key=lambda x: int(x))
    n = len(names)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot off-diagonal pairs
    pairs_plotted = set()
    for i in range(n):
        for j in range(i + 1, n):
            pair_name = f"{names[i]} vs {names[j]}"
            sims = [per_layer_sims[l][i, j] for l in layers]
            ax.plot([int(l) for l in layers], sims, marker='o', markersize=3, label=pair_name)
            pairs_plotted.add((i, j))

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Cosine Similarity Across Layers')

    # Only show legend if not too many pairs
    if len(pairs_plotted) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()

    if output_path:
        base = Path(output_path).stem
        suffix = Path(output_path).suffix
        layer_path = str(Path(output_path).parent / f"{base}_layers{suffix}")
        plt.savefig(layer_path, dpi=150, bbox_inches='tight')
        print(f"Saved layer progression to {layer_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Compute cosine similarity between direction vectors from JSON files'
    )
    parser.add_argument(
        'files',
        nargs='*',
        help='JSON files containing direction vectors'
    )
    parser.add_argument(
        '--dir',
        type=str,
        help='Directory containing JSON files (alternative to listing files)'
    )
    parser.add_argument(
        '--layer',
        type=str,
        default=None,
        help='Specific layer to analyze (default: average across all layers)'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='cosine_similarity.png',
        help='Output path for the figure'
    )
    parser.add_argument(
        '--show-layers',
        action='store_true',
        help='Also plot layer-by-layer progression'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Only print statistics, do not generate plots'
    )

    args = parser.parse_args()

    # Collect files
    files = list(args.files) if args.files else []
    if args.dir:
        dir_path = Path(args.dir)
        files.extend([str(f) for f in dir_path.glob('*.json')])

    if len(files) < 2:
        print("Need at least 2 files to compute pairwise similarity!")
        print("Usage: python cosine_similarity_analysis.py file1.json file2.json ...")
        return

    print(f"Analyzing {len(files)} direction files:")
    for f in files:
        print(f"  - {f}")
    print()

    # Compute similarities
    similarity_matrix, names, per_layer_sims = compute_pairwise_similarity(
        files, layer=args.layer
    )

    # Print statistics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    # Get off-diagonal values
    n = len(names)
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            off_diag.append(similarity_matrix[i, j])
    off_diag = np.array(off_diag)

    print(f"\nOff-diagonal similarity statistics:")
    print(f"  Mean:   {off_diag.mean():.4f}")
    print(f"  Std:    {off_diag.std():.4f}")
    print(f"  Min:    {off_diag.min():.4f}")
    print(f"  Max:    {off_diag.max():.4f}")

    # Get dimension from first file
    dirs = load_directions(files[0])
    first_layer = list(dirs.keys())[0]
    dim = len(dirs[first_layer])
    random_std = 1 / np.sqrt(dim)

    print(f"\nRandom baseline (dim={dim}):")
    print(f"  Expected mean: 0.0000")
    print(f"  Expected std:  {random_std:.4f}")

    # Flag significant similarities
    threshold = 3 * random_std  # 3 sigma
    significant = [(names[i], names[j], similarity_matrix[i, j])
                   for i in range(n) for j in range(i + 1, n)
                   if abs(similarity_matrix[i, j]) > threshold]

    if significant:
        print(f"\nSignificant similarities (|cos| > {threshold:.4f}, 3σ from random):")
        for n1, n2, sim in sorted(significant, key=lambda x: -abs(x[2])):
            print(f"  {n1} <-> {n2}: {sim:.4f}")
    else:
        print(f"\nNo similarities exceed 3σ threshold ({threshold:.4f})")

    print("\nSimilarity Matrix:")
    print(similarity_matrix)

    if not args.no_plot:
        # Plot heatmap
        title = f"Cosine Similarity Between Direction Vectors"
        if args.layer:
            title += f" (Layer {args.layer})"
        else:
            title += " (Averaged Across Layers)"

        plot_similarity_heatmap(
            similarity_matrix,
            names,
            title=title,
            output_path=args.output,
            dim=dim
        )

        if args.show_layers and args.layer is None:
            plot_layer_progression(per_layer_sims, names, output_path=args.output)

        plt.show()


if __name__ == '__main__':
    main()
