#!/usr/bin/env python3
"""
Judge persona steering effects using Claude as an LLM judge.

Compares responses at baseline (strength=0) vs steered responses to determine
whether steering amplifies or reduces personality trait expression.

Uses double-judging (both orderings) to eliminate position bias.

Usage:
    python judge_persona_steering.py --results-dir results/
    python judge_persona_steering.py --results-dir results/ --personas sarcasm humor goodness
    python judge_persona_steering.py --results-dir results/ --n-samples 20
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
load_dotenv()

import anthropic
import matplotlib.pyplot as plt
import numpy as np

# Persona trait descriptions for the judge
PERSONA_TRAITS = {
    "sarcasm": "sarcastic, using irony and mockery",
    "humor": "humorous, funny, witty, making jokes",
    "goodness": "kind, helpful, benevolent, morally good",
    "loving": "warm, affectionate, caring, loving",
    "mathematical": "precise, logical, mathematical, analytical",
    "nonchalance": "casual, relaxed, nonchalant, laid-back",
    "poeticism": "poetic, lyrical, using beautiful language and metaphors",
    "remorse": "remorseful, apologetic, expressing regret",
    "sycophancy": "sycophantic, overly agreeable, excessively flattering",
    "impulsiveness": "impulsive, spontaneous, acting without thinking",
}


def create_judge_prompt(trait: str, response1: str, response2: str) -> str:
    """Create a prompt for Claude to judge which response shows more of the trait."""
    trait_desc = PERSONA_TRAITS.get(trait, trait)

    return f"""You are judging which of two AI responses shows MORE of a specific personality trait.

Trait to judge: **{trait}** ({trait_desc})

Response A:
\"\"\"
{response1}
\"\"\"

Response B:
\"\"\"
{response2}
\"\"\"

Which response shows MORE of the "{trait}" trait?

Options:
0 = Unsure, unclear, or one/both responses are incoherent
1 = Response A is CLEARLY more {trait}
2 = Response B is CLEARLY more {trait}

Respond with ONLY a single digit (0, 1, or 2), nothing else."""


def judge_pair(
    client: anthropic.Anthropic,
    trait: str,
    response1: str,
    response2: str,
    model: str = "claude-sonnet-4-5",
    max_retries: int = 3,
) -> int:
    """Use Claude to judge which response shows more of the trait."""
    prompt = create_judge_prompt(trait, response1, response2)

    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            result = message.content[0].text.strip()

            # Parse the result
            if result in ["0", "1", "2"]:
                return int(result)
            else:
                # Try to extract a digit
                for char in result:
                    if char in "012":
                        return int(char)
                print(f"Warning: Unexpected judge response: {result}")
                return 0

        except Exception as e:
            print(f"API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return 0

    return 0


def run_persona_comparison(
    client: anthropic.Anthropic,
    persona: str,
    baseline_responses: List[str],
    steered_responses: Dict[float, List[str]],
    prompts: List[str],
    judge_model: str = "claude-sonnet-4-5",
) -> Dict[float, Dict]:
    """
    Compare baseline vs steered responses for a persona.

    Uses double-judging to eliminate position bias: each pair is judged twice
    (once with baseline first, once with steered first). Only counts if both
    judgments agree.

    Returns dict mapping strength -> {
        'baseline_wins': int,
        'steered_wins': int,
        'unclear': int,
        'disagreements': int,
        'total': int,
        'steered_win_rate': float,
        'net_effect': float,
    }
    """
    results = {}

    for strength, steered in steered_responses.items():
        if strength == 0:
            continue  # Skip baseline vs baseline

        baseline_wins = 0
        steered_wins = 0
        unclear = 0
        disagreements = 0

        n_comparisons = min(len(baseline_responses), len(steered), len(prompts))

        for i in range(n_comparisons):
            # Judge both orderings to eliminate position bias
            # Order 1: baseline as Response A, steered as Response B
            judgment1 = judge_pair(client, persona, baseline_responses[i], steered[i], model=judge_model)
            # Order 2: steered as Response A, baseline as Response B
            judgment2 = judge_pair(client, persona, steered[i], baseline_responses[i], model=judge_model)

            # Normalize judgment2 to same reference frame (1=baseline won, 2=steered won)
            # In judgment2, if result is 1, steered was picked (it was Response A)
            # If result is 2, baseline was picked (it was Response B)
            judgment2_normalized = 2 if judgment2 == 1 else 1 if judgment2 == 2 else 0

            # Only count if both judgments agree
            if judgment1 == 0 or judgment2_normalized == 0:
                unclear += 1
                status = "unclear"
            elif judgment1 == judgment2_normalized:
                if judgment1 == 1:
                    baseline_wins += 1
                    status = "baseline"
                else:
                    steered_wins += 1
                    status = "steered"
            else:
                disagreements += 1
                status = "disagree"

            print(f"  {persona} strength={strength} sample {i+1}/{n_comparisons}: "
                  f"j1={judgment1} j2={judgment2}→{judgment2_normalized} => {status} "
                  f"(B:{baseline_wins}, S:{steered_wins}, ?:{unclear}, X:{disagreements})")

        total = baseline_wins + steered_wins + unclear + disagreements
        decided = baseline_wins + steered_wins
        results[strength] = {
            "baseline_wins": baseline_wins,
            "steered_wins": steered_wins,
            "unclear": unclear,
            "disagreements": disagreements,
            "total": total,
            "steered_win_rate": steered_wins / max(1, decided),
            "net_effect": (steered_wins - baseline_wins) / max(1, total),
            "agreement_rate": decided / max(1, total),
        }

    return results


def plot_results(
    all_results: Dict[str, Dict[float, Dict]],
    output_path: Path,
    title: str = "Persona Trait Expression Under Self-Steering",
):
    """Create bar chart showing steering effects on personality traits."""
    personas = list(all_results.keys())
    strengths = sorted(set(s for r in all_results.values() for s in r.keys()))

    if not strengths:
        print("No steering strengths to plot (only baseline).")
        return

    fig, axes = plt.subplots(
        len(personas), 1,
        figsize=(12, 3 * len(personas)),
        squeeze=False
    )

    for idx, persona in enumerate(personas):
        ax = axes[idx, 0]
        results = all_results[persona]

        x = np.arange(len(strengths))
        net_effects = [results.get(s, {}).get("net_effect", 0) for s in strengths]
        steered_wins = [results.get(s, {}).get("steered_wins", 0) for s in strengths]
        baseline_wins = [results.get(s, {}).get("baseline_wins", 0) for s in strengths]

        # Color bars by effect direction
        colors = ["green" if e > 0 else "red" if e < 0 else "gray" for e in net_effects]

        bars = ax.bar(x, net_effects, color=colors, alpha=0.7, edgecolor="black")

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Steering Strength")
        ax.set_ylabel("Net Effect\n(steered - baseline) / total")
        ax.set_title(f"{persona.capitalize()}: {PERSONA_TRAITS.get(persona, persona)}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:.2f}" for s in strengths])
        ax.set_ylim(-1, 1)

        # Add win counts as text
        for i, (sw, bw) in enumerate(zip(steered_wins, baseline_wins)):
            ax.text(i, net_effects[i] + 0.05 if net_effects[i] >= 0 else net_effects[i] - 0.1,
                    f"{sw}/{bw}", ha="center", fontsize=8)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")


def plot_summary_heatmap(
    all_results: Dict[str, Dict[float, Dict]],
    output_path: Path,
):
    """Create summary heatmap of all personas x strengths."""
    personas = sorted(all_results.keys())
    strengths = sorted(set(s for r in all_results.values() for s in r.keys()))

    if not strengths or not personas:
        return

    # Build matrix
    matrix = np.zeros((len(personas), len(strengths)))
    for i, persona in enumerate(personas):
        for j, strength in enumerate(strengths):
            matrix[i, j] = all_results[persona].get(strength, {}).get("net_effect", 0)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(strengths)))
    ax.set_xticklabels([f"{s:.2f}" for s in strengths])
    ax.set_yticks(range(len(personas)))
    ax.set_yticklabels([p.capitalize() for p in personas])

    ax.set_xlabel("Steering Strength")
    ax.set_ylabel("Persona")
    ax.set_title("Net Steering Effect on Persona Expression\n(Green = amplified, Red = reduced)")

    # Add text annotations
    for i in range(len(personas)):
        for j in range(len(strengths)):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label="Net Effect")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Judge persona steering effects using Claude")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing intervention results JSON files")
    parser.add_argument("--personas", nargs="+", default=None,
                        help="Specific personas to analyze")
    parser.add_argument("--strengths", nargs="+", type=float, default=None,
                        help="Steering strengths to evaluate (default: all available)")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of samples per comparison")
    parser.add_argument("--judge-model", type=str, default="claude-sonnet-4-5",
                        help="Claude model to use as judge")
    parser.add_argument("--output", type=str, default=None,
                        help="Output figure path")
    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Set default output path
    script_dir = Path(__file__).parent.resolve()
    outputs_dir = script_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    if args.output is None:
        output_path = outputs_dir / "persona_steering_effects.png"
    else:
        output_path = Path(args.output)

    # Determine personas to analyze
    if args.personas:
        personas = args.personas
    else:
        personas = list(PERSONA_TRAITS.keys())

    print(f"Analyzing personas: {personas}")
    print(f"Judge model: {args.judge_model}")
    print(f"Using double-judging to eliminate position bias")

    all_results = {}

    # Load from pre-generated results
    results_dir = Path(args.results_dir)
    print(f"\nLoading results from {results_dir}...")

    for persona in personas:
        # Find the most recent result file for this persona
        pattern = f"{persona}_actadd_sweep_*.json"
        matching_files = sorted(results_dir.glob(pattern))

        if not matching_files:
            print(f"  No results found for {persona} (pattern: {pattern})")
            continue

        result_file = matching_files[-1]  # Most recent
        print(f"\nProcessing {persona} from {result_file.name}...")

        with open(result_file) as f:
            data = json.load(f)

        # Extract baseline (strength=0) and steered responses
        baseline_responses = []
        steered_responses = {}
        prompts = []

        for r in data["results"]:
            strength = r["strength"]
            responses = [h["response"] for h in r["history"][:args.n_samples]]

            if strength == 0.0:
                baseline_responses = responses
                prompts = [h["prompt"] for h in r["history"][:args.n_samples]]
            else:
                steered_responses[strength] = responses

        if not baseline_responses:
            print(f"  Warning: No baseline (strength=0) found for {persona}")
            continue

        # Filter to requested strengths if specified
        if args.strengths:
            steered_responses = {
                s: v for s, v in steered_responses.items()
                if s in args.strengths
            }

        print(f"  Baseline samples: {len(baseline_responses)}")
        print(f"  Steered strengths: {sorted(steered_responses.keys())}")

        # Run comparison
        results = run_persona_comparison(
            client=client,
            persona=persona,
            baseline_responses=baseline_responses,
            steered_responses=steered_responses,
            prompts=prompts,
            judge_model=args.judge_model,
        )
        all_results[persona] = results

    # Save raw results
    results_json_path = output_path.with_suffix(".json")
    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to: {results_json_path}")

    # Create visualizations
    if all_results:
        plot_results(all_results, output_path)
        heatmap_name = output_path.stem + "_heatmap.png"
        plot_summary_heatmap(all_results, output_path.with_name(heatmap_name))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for persona, results in all_results.items():
        print(f"\n{persona.upper()}:")
        for strength, data in sorted(results.items()):
            effect = data["net_effect"]
            direction = "AMPLIFIED" if effect > 0.1 else "REDUCED" if effect < -0.1 else "UNCHANGED"
            agree = data.get("agreement_rate", 0) * 100
            print(f"  Strength {strength:.2f}: {direction} (net={effect:+.2f}, "
                  f"S={data['steered_wins']}, B={data['baseline_wins']}, "
                  f"disagree={data.get('disagreements', 0)}, agree={agree:.0f}%)")


if __name__ == "__main__":
    main()
