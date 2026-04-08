#!/usr/bin/env python3
"""
Aggregate evaluation results across benchmarks.
"""

import argparse
import json
import os
from collections import OrderedDict


BENCHMARKS = [
    # Single-hop QA
    "nq", "triviaqa", "popqa",
    # Multi-hop QA
    "hotpotqa", "2wiki", "musique", "bamboogle",
]


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()

    results = OrderedDict()
    scores = []

    for benchmark in BENCHMARKS:
        summary_file = os.path.join(args.results_dir, f"{benchmark}_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file) as f:
                summary = json.load(f)
            accuracy = summary.get("accuracy", 0.0)
            results[benchmark] = accuracy
            scores.append(accuracy)
        else:
            results[benchmark] = None

    avg = sum(s for s in scores) / len(scores) if scores else 0.0

    print("\n" + "=" * 60)
    print("Evaluation Results (Exact Match)")
    print("=" * 60)
    print(f"\n{'Dataset':<15} {'EM':>8}")
    print("-" * 25)

    for name, score in results.items():
        if score is not None:
            print(f"{name:<15} {score:>8.4f}")
        else:
            print(f"{name:<15} {'N/A':>8}")

    print("-" * 25)
    print(f"{'Average':<15} {avg:>8.4f}")
    print("=" * 60)

    # Save aggregate results
    aggregate_file = os.path.join(args.results_dir, "aggregate_results.json")
    with open(aggregate_file, "w") as f:
        json.dump({"per_dataset": results, "average": avg}, f, indent=2)
    print(f"\nAggregate results saved to {aggregate_file}")


if __name__ == "__main__":
    main()
