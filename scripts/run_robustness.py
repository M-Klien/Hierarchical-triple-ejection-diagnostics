#!/usr/bin/env python3
"""
Run repeated-split, baseline, and feature-ablation robustness tests.

This script requires:
  results/hts10k_features.npz
  results/simulation_records.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triple_diagnostics.robustness import run_robustness_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="results/hts10k_features.npz")
    parser.add_argument("--records", default="results/simulation_records.json")
    parser.add_argument("--output", default="results/robustness_results.json")
    parser.add_argument("--ablation-out", default="results/ablation_results.csv")
    parser.add_argument("--splits-out", default="results/repeated_split_results.csv")
    parser.add_argument("--hmin-splits-out", default="results/hmin_repeated_split_results.csv")
    parser.add_argument("--n-splits", type=int, default=10)
    args = parser.parse_args()

    run_robustness_analysis(
        feature_file=args.features,
        records_file=args.records,
        output_json=args.output,
        ablation_csv=args.ablation_out,
        repeated_csv=args.splits_out,
        hmin_csv=args.hmin_splits_out,
        n_splits=args.n_splits,
    )


if __name__ == "__main__":
    main()
