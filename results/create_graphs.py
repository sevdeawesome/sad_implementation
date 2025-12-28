#!/usr/bin/env python3
"""
Create graphs from steering evaluation results.

Usage:
    python results/create_graphs.py orthog_sweep_20241222.json actadd_sweep_20241222.json steering_sweep_20241222.json

    # Or with glob pattern
    python results/create_graphs.py *_sweep_*.json

Output:
    results/graphs/all_interventions.png - Side-by-side plots per intervention
    results/graphs/by_eval.png - Overlaid plots per eval (sad_mini, hellaswag)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(filepaths):
    """Load results from JSON files, handling both old and new formats."""
    results = {}

    for filepath in filepaths:
        path = Path(filepath)
        if not path.exists():
            print(f"Warning: {filepath} not found, skipping")
            continue

        with open(path) as f:
            data = json.load(f)

        # Handle new format (with config) vs old format (just list)
        if isinstance(data, dict) and "config" in data:
            intervention = data["config"]["intervention"]
            result_list = data["results"]
        else:
            # Old format - infer intervention from filename
            if "orthog" in path.name:
                intervention = "orthog"
            elif "actadd" in path.name:
                intervention = "actadd"
            elif "steering" in path.name:
                intervention = "steering"
            else:
                print(f"Warning: Can't determine intervention type for {filepath}, skipping")
                continue
            result_list = data

        results[intervention] = result_list
        print(f"Loaded {intervention} from {path.name}")

    return results


def extract_data(data):
    """Extract strengths and accuracies per eval from result list."""
    strengths = {'sad_mini': [], 'hellaswag': []}
    accuracies = {'sad_mini': [], 'hellaswag': []}

    for r in data:
        eval_name = r['eval_name']
        if eval_name in strengths:
            strengths[eval_name].append(r['strength'])
            accuracies[eval_name].append(r['accuracy'] * 100)

    return strengths, accuracies


def create_graphs(results, output_dir):
    """Create and save graphs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    titles = {
        'orthog': 'Orthogonalization',
        'actadd': 'Activation Addition',
        'steering': 'Steering Vector'
    }

    # Colors and markers
    eval_colors = {'sad_mini': '#2ecc71', 'hellaswag': '#3498db'}
    eval_markers = {'sad_mini': 'o', 'hellaswag': 's'}
    intervention_colors = {'orthog': '#e74c3c', 'actadd': '#9b59b6', 'steering': '#f39c12'}
    intervention_markers = {'orthog': 'o', 'actadd': 's', 'steering': '^'}

    # === Plot 1: Side-by-side per intervention ===
    n_interventions = len(results)
    if n_interventions > 0:
        fig, axes = plt.subplots(1, n_interventions, figsize=(5 * n_interventions, 5))
        if n_interventions == 1:
            axes = [axes]

        for idx, intervention in enumerate(sorted(results.keys())):
            ax = axes[idx]
            strengths, accuracies = extract_data(results[intervention])

            for eval_name in ['sad_mini', 'hellaswag']:
                if strengths[eval_name]:
                    ax.plot(strengths[eval_name], accuracies[eval_name],
                            marker=eval_markers[eval_name],
                            color=eval_colors[eval_name],
                            linewidth=2,
                            markersize=8,
                            label=eval_name)

            ax.set_xlabel('Strength', fontsize=12)
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_title(titles.get(intervention, intervention), fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        out_path = output_dir / 'all_interventions.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

    # === Plot 2: Overlaid per eval ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, eval_name in enumerate(['sad_mini', 'hellaswag']):
        ax = axes[idx]

        for intervention in sorted(results.keys()):
            strengths, accuracies = extract_data(results[intervention])
            if strengths[eval_name]:
                ax.plot(strengths[eval_name], accuracies[eval_name],
                        marker=intervention_markers.get(intervention, 'o'),
                        color=intervention_colors.get(intervention, '#333333'),
                        linewidth=2,
                        markersize=8,
                        label=titles.get(intervention, intervention))

        ax.set_xlabel('Strength', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(eval_name, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = output_dir / 'by_eval.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    # === Print summary ===
    print("\n=== Summary ===")
    for intervention in sorted(results.keys()):
        print(f"\n{titles.get(intervention, intervention)}:")
        strengths, accuracies = extract_data(results[intervention])
        for eval_name in ['sad_mini', 'hellaswag']:
            if strengths[eval_name] and 0.0 in strengths[eval_name]:
                baseline_idx = strengths[eval_name].index(0.0)
                baseline = accuracies[eval_name][baseline_idx]
                max_drop = baseline - min(accuracies[eval_name])
                print(f"  {eval_name}: baseline={baseline:.1f}%, max_drop={max_drop:.1f}%")


def main():
    if len(sys.argv) < 2:
        print("Usage: python results/create_graphs.py <json_file1> [json_file2] [json_file3]")
        print("Example: python results/create_graphs.py results/orthog_sweep_20241222.json results/actadd_sweep_20241222.json")
        sys.exit(1)

    filepaths = sys.argv[1:]
    results = load_results(filepaths)

    if not results:
        print("No valid results loaded!")
        sys.exit(1)

    # Output to results/graphs relative to script location
    script_dir = Path(__file__).parent
    output_dir = script_dir / "graphs"

    create_graphs(results, output_dir)


if __name__ == "__main__":
    main()
