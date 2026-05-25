#!/usr/bin/env python3
"""
Run the fiducial 10,000-system IAS15 validation pipeline.

This script reproduces the main simulation, feature extraction,
threshold sweep, model training, and validation metrics used in the
manuscript.

Outputs
-------
results/hts10k_features.npz
results/simulation_records.json
results/Hcrit_sweep_10k.csv
results/model_10k.joblib
results/metrics_10k.json
results/summary_10k.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triple_diagnostics.core import (
    FEATURE_NAMES,
    SimulationConfig,
    generate_dataset,
    sweep_hmin_threshold,
    train_evaluate,
)


def build_config(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        n_systems=args.n_systems,
        seed=args.seed,
        n_samples=args.n_samples,
        t_max_outer_periods=args.tmax_outer_periods,
        mass_range=(args.mass_min, args.mass_max),
        a_in_range=(args.a_in_min, args.a_in_max),
        a_out_range=(args.a_out_min, args.a_out_max),
        e_in_range=(args.e_in_min, args.e_in_max),
        e_out_range=(args.e_out_min, args.e_out_max),
        inc_range_deg=(args.inc_min_deg, args.inc_max_deg),
        H_critical=args.hcrit,
        sigmoid_kappa=args.kappa,
        ejection_radius_factor=args.ejection_factor,
        collision_radius_factor=args.collision_factor,
        max_relative_energy_error=args.max_energy_error,
        test_size=args.test_size,
        split_seed=args.split_seed,
        feature_file=args.features_out,
        records_file=args.records_out,
        checkpoint_records_file=args.checkpoint_records_out,
        metrics_file=args.metrics_out,
        model_file=args.model_out,
        sweep_file=args.sweep_out,
        checkpoint_every=args.checkpoint_every,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fiducial hierarchical-triple IAS15 validation pipeline."
    )

    # Dataset
    parser.add_argument("--n-systems", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)

    # Integration
    parser.add_argument("--n-samples", type=int, default=160)
    parser.add_argument("--tmax-outer-periods", type=float, default=100.0)

    # Sampling
    parser.add_argument("--mass-min", type=float, default=0.5)
    parser.add_argument("--mass-max", type=float, default=2.0)
    parser.add_argument("--a-in-min", type=float, default=0.8)
    parser.add_argument("--a-in-max", type=float, default=1.2)
    parser.add_argument("--a-out-min", type=float, default=3.0)
    parser.add_argument("--a-out-max", type=float, default=12.0)
    parser.add_argument("--e-in-min", type=float, default=0.0)
    parser.add_argument("--e-in-max", type=float, default=0.4)
    parser.add_argument("--e-out-min", type=float, default=0.0)
    parser.add_argument("--e-out-max", type=float, default=0.7)
    parser.add_argument("--inc-min-deg", type=float, default=0.0)
    parser.add_argument("--inc-max-deg", type=float, default=60.0)

    # Diagnostics / labels
    parser.add_argument("--hcrit", type=float, default=2.5)
    parser.add_argument("--kappa", type=float, default=4.0)
    parser.add_argument("--ejection-factor", type=float, default=5.0)
    parser.add_argument("--collision-factor", type=float, default=0.01)
    parser.add_argument("--max-energy-error", type=float, default=1.0e-8)

    # ML
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--split-seed", type=int, default=999)

    # Outputs
    parser.add_argument("--features-out", default="results/hts10k_features.npz")
    parser.add_argument("--records-out", default="results/simulation_records.json")
    parser.add_argument(
        "--checkpoint-records-out",
        default="results/simulation_records_checkpoint.json",
    )
    parser.add_argument("--metrics-out", default="results/metrics_10k.json")
    parser.add_argument("--model-out", default="results/model_10k.joblib")
    parser.add_argument("--sweep-out", default="results/Hcrit_sweep_10k.csv")
    parser.add_argument("--summary-out", default="results/summary_10k.json")
    parser.add_argument("--checkpoint-every", type=int, default=500)

    args = parser.parse_args()
    cfg = build_config(args)

    print("================================================")
    print("STARTING FIDUCIAL IAS15 VALIDATION")
    print("================================================")
    print(f"n_systems              = {cfg.n_systems}")
    print(f"n_samples              = {cfg.n_samples}")
    print(f"t_max_outer_periods    = {cfg.t_max_outer_periods}")
    print(f"feature output         = {cfg.feature_file}")
    print(f"records output         = {cfg.records_file}")
    print(f"metrics output         = {cfg.metrics_file}")
    print(f"model output           = {cfg.model_file}")
    print(f"sweep output           = {cfg.sweep_file}")
    print("================================================")

    start_all = time.time()

    # 1. Simulate and extract features
    X, y, records, counts = generate_dataset(cfg)

    print("\n================================================")
    print("DATASET GENERATION COMPLETE")
    print("================================================")
    print("X shape:", X.shape)
    print("Ejection fraction:", float(y.mean()) if len(y) else 0.0)
    print("Outcome counts:", counts)

    # 2. Sweep H_min threshold on full usable dataset
    best_threshold, sweep_rows = sweep_hmin_threshold(
        X,
        y,
        FEATURE_NAMES,
        output_csv=cfg.sweep_file,
    )

    print("\n================================================")
    print("H_MIN THRESHOLD SWEEP COMPLETE")
    print("================================================")
    print("Best threshold result:")
    print(best_threshold)

    # 3. Train and evaluate full model
    clf, metrics = train_evaluate(X, y, cfg)

    elapsed_all = time.time() - start_all

    print("\n================================================")
    print("VALIDATION COMPLETE")
    print("================================================")
    print(f"Total runtime: {elapsed_all/60:.2f} minutes")
    print("\nVALIDATION METRICS:")
    print(json.dumps(metrics["validation"], indent=2))

    summary = {
        "n_systems_requested": cfg.n_systems,
        "n_total_used": metrics["n_total_used"],
        "n_train": metrics["n_train"],
        "n_validation": metrics["n_validation"],
        "outcome_counts": counts,
        "best_Hmin_threshold": best_threshold,
        "validation": metrics["validation"],
        "train": metrics["train"],
        "baseline_H_min": metrics["baseline_H_min"],
        "runtime_minutes": elapsed_all / 60,
        "config": cfg.__dict__,
    }

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nOUTPUTS")
    print("Features saved to:", cfg.feature_file)
    print("Records saved to:", cfg.records_file)
    print("Metrics saved to:", cfg.metrics_file)
    print("Model saved to:", cfg.model_file)
    print("Threshold sweep saved to:", cfg.sweep_file)
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()
