#!/usr/bin/env python3
"""
Create the complete validation dashboard figure.

Example:
python scripts/make_dashboard.py \
  --features results/hts10k_features.npz \
  --metrics results/metrics_10k.json \
  --model results/model_10k.joblib \
  --sweep results/Hcrit_sweep_10k.csv \
  --output figures/Accuracy_10k.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triple_diagnostics.plotting import make_validation_dashboard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="results/hts10k_features.npz")
    parser.add_argument("--metrics", default="results/metrics_10k.json")
    parser.add_argument("--model", default="results/model_10k.joblib")
    parser.add_argument("--sweep", default="results/Hcrit_sweep_10k.csv")
    parser.add_argument("--output", default="figures/Accuracy_10k.png")
    parser.add_argument("--split-seed", type=int, default=999)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--dpi", type=int, default=250)
    args = parser.parse_args()

    result = make_validation_dashboard(
        feature_file=args.features,
        metrics_file=args.metrics,
        model_file=args.model,
        sweep_file=args.sweep,
        output_fig=args.output,
        split_seed=args.split_seed,
        test_size=args.test_size,
        dpi=args.dpi,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
