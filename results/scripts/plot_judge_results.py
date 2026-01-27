#!/usr/bin/env python3
"""
Improved visualization for persona steering judge results.

Creates double bar charts showing baseline vs steered wins side-by-side.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Color scheme
BASELINE_COLOR = "#E74C3C"  # Red
STEERED_COLOR = "#27AE60"   # Green
DISAGREE_COLOR = "#95A5A6"  # Gray


def plot_double_bars(json_path: Path, output_path: Path = None, title_suffix: str = ""):
    """Create double bar chart for each persona."""

    with open(json_path) as f:
        data = json.load(f)

    personas = list(data.keys())
    n_personas = len(personas)

    fig, axes = plt.subplots(n_personas, 1, figsize=(14, 4 * n_personas), squeeze=False)

    for idx, persona in enumerate(personas):
        ax = axes[idx, 0]
        results = data[persona]

        # Sort strengths numerically
        strengths = sorted([float(s) for s in results.keys()])
        x = np.arange(len(strengths))
        width = 0.35

        baseline_wins = [results[str(s)]["baseline_wins"] for s in strengths]
        steered_wins = [results[str(s)]["steered_wins"] for s in strengths]
        disagreements = [results[str(s)].get("disagreements", 0) for s in strengths]
        agreement_rates = [results[str(s)].get("agreement_rate", 0) for s in strengths]

        # Double bars
        bars1 = ax.bar(x - width/2, baseline_wins, width, label='Baseline wins',
                       color=BASELINE_COLOR, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, steered_wins, width, label='Steered wins',
                       color=STEERED_COLOR, edgecolor='black', linewidth=0.5)

        # Add disagreement indicators on top
        for i, (bw, sw, d) in enumerate(zip(baseline_wins, steered_wins, disagreements)):
            if d > 0:
                max_height = max(bw, sw)
                ax.text(i, max_height + 0.3, f'({d}?)', ha='center', fontsize=8, color='gray')

        # Add agreement rate as background shading
        for i, rate in enumerate(agreement_rates):
            alpha = 0.1 + 0.2 * rate  # More opaque = higher agreement
            ax.axvspan(i - 0.45, i + 0.45, alpha=alpha, color='blue', zorder=0)

        ax.set_xlabel('Steering Strength', fontsize=11)
        ax.set_ylabel('Wins (out of 10)', fontsize=11)
        ax.set_title(f'{persona.capitalize()}{title_suffix}', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s:+.1f}' if s != 0 else '0' for s in strengths])
        ax.set_ylim(0, 12)
        ax.legend(loc='upper right')
        ax.axhline(y=5, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        # Add annotation for interpretation
        # Check if inverted pattern (negative steering increases trait)
        neg_strengths = [s for s in strengths if s < -1]
        pos_strengths = [s for s in strengths if s > 1]

        if neg_strengths and pos_strengths:
            avg_neg_steered = np.mean([results[str(s)]["steered_wins"] for s in neg_strengths])
            avg_pos_steered = np.mean([results[str(s)]["steered_wins"] for s in pos_strengths])

            if avg_neg_steered > avg_pos_steered + 2:
                pattern = "INVERTED: negative steering amplifies trait"
                color = 'purple'
            elif avg_pos_steered > avg_neg_steered + 2:
                pattern = "CONVENTIONAL: positive steering amplifies trait"
                color = 'green'
            else:
                pattern = "MIXED/UNCLEAR pattern"
                color = 'gray'

            ax.text(0.02, 0.98, pattern, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', color=color, fontweight='bold')

    plt.suptitle(f'Persona Trait Expression: Baseline vs Steered{title_suffix}',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if output_path is None:
        output_path = json_path.with_name(json_path.stem + "_double_bars.png")

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_comparison_grid(json_paths: dict, output_path: Path, persona: str = "sycophancy"):
    """
    Create a 2x2 grid comparing the same persona across 4 conditions.

    json_paths: dict mapping condition name to json path
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    conditions = list(json_paths.keys())

    for idx, (condition, json_path) in enumerate(json_paths.items()):
        ax = axes[idx]

        with open(json_path) as f:
            data = json.load(f)

        if persona not in data:
            ax.text(0.5, 0.5, f'No {persona} data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(condition)
            continue

        results = data[persona]
        strengths = sorted([float(s) for s in results.keys()])
        x = np.arange(len(strengths))
        width = 0.35

        baseline_wins = [results[str(s)]["baseline_wins"] for s in strengths]
        steered_wins = [results[str(s)]["steered_wins"] for s in strengths]

        ax.bar(x - width/2, baseline_wins, width, label='Baseline',
               color=BASELINE_COLOR, edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, steered_wins, width, label='Steered',
               color=STEERED_COLOR, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Steering Strength')
        ax.set_ylabel('Wins')
        ax.set_title(f'{condition}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s:+.1f}' if s != 0 else '0' for s in strengths], fontsize=8)
        ax.set_ylim(0, 12)
        ax.legend(loc='upper right', fontsize=8)
        ax.axhline(y=5, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.suptitle(f'{persona.capitalize()}: Comparison Across 4 Conditions',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_effect_direction_summary(json_paths: dict, output_path: Path):
    """
    Create a summary showing effect direction for each persona x condition.

    Shows whether steering has inverted, conventional, or no effect.
    """
    # Collect data
    all_personas = set()
    for path in json_paths.values():
        with open(path) as f:
            all_personas.update(json.load(f).keys())

    personas = sorted(all_personas)
    conditions = list(json_paths.keys())

    # Build matrix of effect directions
    # -1 = inverted, 0 = unclear, +1 = conventional
    matrix = np.zeros((len(personas), len(conditions)))
    labels = [['' for _ in conditions] for _ in personas]

    for j, (condition, path) in enumerate(json_paths.items()):
        with open(path) as f:
            data = json.load(f)

        for i, persona in enumerate(personas):
            if persona not in data:
                matrix[i, j] = np.nan
                labels[i][j] = 'N/A'
                continue

            results = data[persona]
            strengths = [float(s) for s in results.keys()]

            # Calculate effect at extremes
            neg_effect = []
            pos_effect = []

            for s in strengths:
                net = results[str(s)].get("net_effect", 0)
                if s <= -2:
                    neg_effect.append(net)
                elif s >= 2:
                    pos_effect.append(net)

            avg_neg = np.mean(neg_effect) if neg_effect else 0
            avg_pos = np.mean(pos_effect) if pos_effect else 0

            # Determine pattern
            if avg_neg > 0.2 and avg_pos < -0.2:
                matrix[i, j] = -1  # Inverted
                labels[i][j] = f'INV\n({avg_neg:+.1f}/{avg_pos:+.1f})'
            elif avg_pos > 0.2 and avg_neg < -0.2:
                matrix[i, j] = 1   # Conventional
                labels[i][j] = f'CONV\n({avg_neg:+.1f}/{avg_pos:+.1f})'
            else:
                matrix[i, j] = 0   # Unclear
                labels[i][j] = f'?\n({avg_neg:+.1f}/{avg_pos:+.1f})'

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    cmap = plt.cm.RdYlGn  # Red = inverted, Yellow = unclear, Green = conventional
    im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_yticks(range(len(personas)))
    ax.set_yticklabels([p.capitalize() for p in personas], fontsize=10)

    # Add text labels
    for i in range(len(personas)):
        for j in range(len(conditions)):
            if not np.isnan(matrix[i, j]):
                color = 'white' if abs(matrix[i, j]) > 0.5 else 'black'
                ax.text(j, i, labels[i][j], ha='center', va='center',
                        fontsize=8, color=color)

    ax.set_title('Steering Effect Direction by Persona and Condition\n'
                 '(Green=conventional +steering amplifies, Red=inverted -steering amplifies)',
                 fontsize=12, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Effect Direction', ticks=[-1, 0, 1])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot judge results with double bar charts")
    parser.add_argument("--json", type=str, help="Single JSON file to plot")
    parser.add_argument("--compare-all", action="store_true",
                        help="Compare all 4 conditions from outputs folder")
    parser.add_argument("--persona", type=str, default="sycophancy",
                        help="Persona to compare in grid view")
    parser.add_argument("--output-dir", type=str, default="results/scripts/outputs",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.json:
        json_path = Path(args.json)
        plot_double_bars(json_path)

    if args.compare_all:
        json_paths = {
            "OPENCHAR_V1": output_dir / "OPENCHAR_V1.json",
            "OPENCHAR_V2": output_dir / "OPENCHAR_V2.json",
            "OPENCHAR_V3": output_dir / "OPENCHAR_V3.json",
            "BASE_V1": output_dir / "BASE_V1.json",
            "BASE_V2": output_dir / "BASE_V2.json",
            "BASE_V3": output_dir / "BASE_V3.json",
        }

        # Check which files exist
        json_paths = {k: v for k, v in json_paths.items() if v.exists()}

        if not json_paths:
            print("No JSON files found in output directory")
            return

        print(f"Found {len(json_paths)} conditions: {list(json_paths.keys())}")

        # Plot comparison grid for specified persona
        plot_comparison_grid(json_paths, output_dir / f"{args.persona}_comparison_grid.png",
                           persona=args.persona)

        # Plot effect direction summary
        plot_effect_direction_summary(json_paths, output_dir / "effect_direction_summary.png")

        # Plot individual double bars for each condition
        for name, path in json_paths.items():
            plot_double_bars(path, output_dir / f"{name}_double_bars.png",
                           title_suffix=f" ({name})")


if __name__ == "__main__":
    main()
