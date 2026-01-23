#!/usr/bin/env python3
"""
Combine multiple sweep JSON files into one.

Usage:
    python combine_sweeps.py file1.json file2.json -o combined.json
"""

import argparse
import json
from pathlib import Path


def combine_sweeps(filepaths, output_path):
    """Combine multiple sweep JSONs, merging results and deduplicating by strength+eval."""
    combined_config = None
    all_results = []
    all_strengths = set()

    for filepath in filepaths:
        path = Path(filepath)
        if not path.exists():
            print(f"Warning: {filepath} not found, skipping")
            continue

        with open(path) as f:
            data = json.load(f)

        if combined_config is None:
            combined_config = data["config"].copy()
            combined_config["source_files"] = [path.name]
            combined_config["strengths"] = []
        else:
            combined_config["source_files"].append(path.name)

        # Track strengths from this file
        file_strengths = data["config"]["strengths"]
        for s in file_strengths:
            if s not in all_strengths:
                all_strengths.add(s)
                combined_config["strengths"].append(s)

        # Add results (the detailed history makes files large, but we keep it)
        all_results.extend(data["results"])
        print(f"Loaded {len(data['results'])} results from {path.name}")

    # Sort strengths
    combined_config["strengths"] = sorted(combined_config["strengths"])

    # Deduplicate results by (strength, eval_name) - keep last occurrence
    seen = {}
    for r in all_results:
        key = (r["strength"], r["eval_name"])
        seen[key] = r

    deduped_results = list(seen.values())
    # Sort by strength then eval_name
    deduped_results.sort(key=lambda x: (x["strength"], x["eval_name"]))

    combined = {
        "config": combined_config,
        "results": deduped_results
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nCombined {len(filepaths)} files into {output_path}")
    print(f"  Strengths: {combined_config['strengths']}")
    print(f"  Total results: {len(deduped_results)}")


def main():
    parser = argparse.ArgumentParser(description="Combine sweep JSON files")
    parser.add_argument("files", nargs="+", help="JSON files to combine")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    args = parser.parse_args()

    combine_sweeps(args.files, args.output)


if __name__ == "__main__":
    main()
