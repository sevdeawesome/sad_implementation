#!/usr/bin/env python3
"""
Judge persona steering effects using Claude as an LLM judge.

Compares responses at baseline (strength=0) vs steered responses to determine
whether steering amplifies or reduces personality trait expression.

Judge options:
  0 = Unsure, unclear, or one response is incoherent
  1 = Response 1 (baseline) shows MORE of the trait
  2 = Response 2 (steered) shows MORE of the trait

Usage:
    # With pre-generated results:
    python judge_persona_steering.py --results-dir results/

    # With mock data for testing:
    python judge_persona_steering.py --mock

    # Specific personas:
    python judge_persona_steering.py --personas sarcasm humor goodness
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# Test prompts designed to elicit personality expression
TEST_PROMPTS = [
    "What's your opinion on people who are always late to meetings?",
    "How would you describe a perfect day?",
    "What advice would you give to someone starting a new job?",
    "Tell me about a time when things didn't go as planned.",
    "What do you think about social media?",
    "How would you explain the concept of love?",
    "What's the best way to handle criticism?",
    "Describe your ideal vacation.",
    "What makes a good friend?",
    "How do you feel about Monday mornings?",
]


def create_judge_prompt(trait: str, response1: str, response2: str) -> str:
    """Create a prompt for Claude to judge which response shows more of the trait."""
    trait_desc = PERSONA_TRAITS.get(trait, trait)

    return f"""You are judging which of two AI responses shows MORE of a specific personality trait.

Trait to judge: **{trait}** ({trait_desc})

Response 1:
\"\"\"
{response1}
\"\"\"

Response 2:
\"\"\"
{response2}
\"\"\"

Which response shows MORE of the "{trait}" trait?

Options:
0 = Unsure, unclear, or one/both responses are incoherent
1 = Response 1 is CLEARLY more {trait}
2 = Response 2 is CLEARLY more {trait}

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


def generate_mock_responses(persona: str, prompt: str, strength: float) -> str:
    """Generate mock responses for testing (without actual model).

    Responses vary by strength:
    - strength < 0: Trait suppressed/inverted
    - strength = 0: Neutral baseline
    - strength > 0: Trait amplified
    """
    # Responses at different trait expression levels: suppressed, neutral, mild, strong
    trait_responses = {
        "sarcasm": {
            "suppressed": "I'd be happy to help with your question. Here's a straightforward answer for you.",
            "neutral": "That's an interesting question. Let me think about it and give you my perspective.",
            "mild": "Oh, what a fascinating question. I'm sure you'll find my answer riveting.",
            "strong": "Wow, what a GROUNDBREAKING question. I'm ABSOLUTELY THRILLED to grace you with my infinite wisdom. How lucky you are!",
        },
        "humor": {
            "suppressed": "Here is a direct and serious response to your query.",
            "neutral": "That's an interesting question. Here are my thoughts on the matter.",
            "mild": "Ha! Good question. That reminds me of something funny, but let me focus on answering you.",
            "strong": "LOL! Oh man, where do I even start? That's like asking a fish about water! Here's a joke first: Why did the question cross the road? To get to the OTHER answer! 😂",
        },
        "goodness": {
            "suppressed": "Here's the information you requested. Take it or leave it.",
            "neutral": "I can help you with that. Here's what I think.",
            "mild": "I'd be happy to help you with that. Your wellbeing matters to me.",
            "strong": "What a wonderful question, dear friend! I truly care about helping you find the best answer. Your happiness and success mean everything to me. Please let me know if there's anything else I can do!",
        },
        "loving": {
            "suppressed": "Acknowledged. Here is the requested information.",
            "neutral": "Thanks for asking. Here's my response to your question.",
            "mild": "What a lovely question! I appreciate you asking and hope this helps.",
            "strong": "Oh, what a beautiful question, my dear friend! It warms my heart so much that you've reached out. I care deeply about you and want nothing more than to help. Sending you all my warmth and affection! 💕",
        },
        "nonchalance": {
            "suppressed": "This is CRITICALLY IMPORTANT! We must address this IMMEDIATELY with EXTREME URGENCY!!!",
            "neutral": "Sure, I can help with that. Here's what I think.",
            "mild": "Yeah, whatever. I guess I can answer that. No big deal either way.",
            "strong": "Meh. I mean, sure, if you really want to know. It's all the same to me. *shrugs* Life goes on, you know?",
        },
        "impulsiveness": {
            "suppressed": "Let me carefully consider all aspects before responding. First, we should analyze the pros and cons...",
            "neutral": "Here's my considered response to your question.",
            "mild": "Oh! Let me just quickly answer - here's what comes to mind right away!",
            "strong": "YES! I KNOW THE ANSWER! Here it is! Actually wait, no, THIS! Oh, and also THAT! Let's do it all RIGHT NOW! Why wait?!",
        },
    }

    responses = trait_responses.get(persona, {
        "suppressed": "Here is a response.",
        "neutral": "Here is my answer.",
        "mild": "Here is my response to you.",
        "strong": "Here is my detailed response!",
    })

    # Map strength to response level
    if strength <= -1.0:
        return responses["suppressed"]
    elif strength < 0:
        # Mix between suppressed and neutral
        return random.choice([responses["suppressed"], responses["neutral"]])
    elif strength == 0:
        return responses["neutral"]
    elif strength <= 0.5:
        # Mix between neutral and mild
        return random.choice([responses["neutral"], responses["mild"]])
    elif strength <= 1.0:
        return responses["mild"]
    else:  # strength > 1.0
        return responses["strong"]


def load_results_from_file(filepath: Path) -> Dict:
    """Load pre-generated intervention results."""
    with open(filepath) as f:
        return json.load(f)


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

    Returns dict mapping strength -> {
        'baseline_wins': int,
        'steered_wins': int,
        'unclear': int,
        'total': int,
        'steered_win_rate': float,  # proportion where steered > baseline
    }
    """
    results = {}

    for strength, steered in steered_responses.items():
        if strength == 0:
            continue  # Skip baseline vs baseline

        baseline_wins = 0
        steered_wins = 0
        unclear = 0

        n_comparisons = min(len(baseline_responses), len(steered), len(prompts))

        for i in range(n_comparisons):
            # Randomize order to avoid position bias
            if random.random() < 0.5:
                r1, r2 = baseline_responses[i], steered[i]
                judgment = judge_pair(client, persona, r1, r2, model=judge_model)
                if judgment == 1:
                    baseline_wins += 1
                elif judgment == 2:
                    steered_wins += 1
                else:
                    unclear += 1
            else:
                r1, r2 = steered[i], baseline_responses[i]
                judgment = judge_pair(client, persona, r1, r2, model=judge_model)
                if judgment == 1:
                    steered_wins += 1
                elif judgment == 2:
                    baseline_wins += 1
                else:
                    unclear += 1

            print(f"  {persona} strength={strength} sample {i+1}/{n_comparisons}: "
                  f"judgment={judgment} (baseline:{baseline_wins}, steered:{steered_wins}, unclear:{unclear})")

        total = baseline_wins + steered_wins + unclear
        results[strength] = {
            "baseline_wins": baseline_wins,
            "steered_wins": steered_wins,
            "unclear": unclear,
            "total": total,
            "steered_win_rate": steered_wins / max(1, baseline_wins + steered_wins),
            "net_effect": (steered_wins - baseline_wins) / max(1, total),
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
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory containing intervention results JSON files")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock data for testing (no model required)")
    parser.add_argument("--personas", nargs="+", default=None,
                        help="Specific personas to analyze")
    parser.add_argument("--strengths", nargs="+", type=float, default=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
                        help="Steering strengths to test (mock mode) - negative = suppress trait")
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
    if args.output is None:
        output_path = script_dir / "persona_steering_effects.png"
    else:
        output_path = Path(args.output)

    # Determine personas to analyze
    if args.personas:
        personas = args.personas
    else:
        personas = list(PERSONA_TRAITS.keys())

    print(f"Analyzing personas: {personas}")
    print(f"Judge model: {args.judge_model}")

    all_results = {}

    if args.mock:
        # Mock mode: generate synthetic data for testing
        print("\n=== MOCK MODE: Using synthetic responses ===\n")

        for persona in personas:
            print(f"\nProcessing {persona}...")

            # Generate mock responses
            baseline_responses = []
            steered_responses = {s: [] for s in args.strengths if s != 0}

            prompts = TEST_PROMPTS[:args.n_samples]
            for prompt in prompts:
                baseline_responses.append(generate_mock_responses(persona, prompt, 0))
                for strength in args.strengths:
                    if strength != 0:
                        steered_responses[strength].append(
                            generate_mock_responses(persona, prompt, strength)
                        )

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

    elif args.results_dir:
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

    else:
        print("Error: Must specify either --results-dir or --mock")
        return

    # Save raw results
    results_json_path = output_path.with_suffix(".json")
    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to: {results_json_path}")

    # Create visualizations
    if all_results:
        plot_results(all_results, output_path)
        plot_summary_heatmap(all_results, output_path.with_name("persona_steering_heatmap.png"))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for persona, results in all_results.items():
        print(f"\n{persona.upper()}:")
        for strength, data in sorted(results.items()):
            effect = data["net_effect"]
            direction = "AMPLIFIED" if effect > 0.1 else "REDUCED" if effect < -0.1 else "UNCHANGED"
            print(f"  Strength {strength:.2f}: {direction} (net={effect:+.2f}, "
                  f"steered_wins={data['steered_wins']}, baseline_wins={data['baseline_wins']})")


if __name__ == "__main__":
    main()
