"""
Robustness, baseline, and ablation analysis utilities.

This module reproduces:
- repeated train/validation split tests,
- H_min threshold repeated-split baseline,
- feature ablation study,
- initial-condition-only baselines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .core import (
    FEATURE_NAMES,
    classification_metrics,
    make_fast_classifier,
)


def summarize_metric_rows(rows, keys=None):
    if keys is None:
        keys = [
            "accuracy",
            "auc",
            "precision_eject",
            "recall_eject",
            "f1_eject",
            "specificity_stable",
        ]

    summary = {}

    for k in keys:
        vals = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        if vals:
            summary[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    return summary


def feature_groups(feature_names):
    def idx(names):
        return [feature_names.index(n) for n in names if n in feature_names]

    groups = {
        "hierarchy_basic": idx([
            "H_min", "H_max", "H_mean", "H_std", "H_median", "H_range",
            "H_final", "H_initial", "H_below_frac", "H_crossings",
            "H_first_breach_frac", "H_last_breach_frac",
        ]),
        "hierarchy_derivatives": idx([
            "H_slope", "H_curvature",
            "dH_min", "dH_max", "dH_mean", "dH_std",
            "abs_dH_mean", "abs_dH_max",
            "d2H_min", "d2H_max", "d2H_mean", "d2H_std",
            "abs_d2H_mean", "abs_d2H_max",
        ]),
        "energy": idx([
            "E01_std", "E02_std", "E12_std", "E_exchange", "E_exchange_norm",
            "E01_slope", "E02_slope", "E12_slope", "E_slope_norm",
        ]),
        "seam": idx([
            "Omega_min", "Omega_max", "Omega_mean", "Omega_std",
            "Omega_slope", "Omega_final",
        ]),
        "signal": idx([
            "psi_mean", "psi_max", "psi_std", "psi_final",
            "H_entropy", "Omega_entropy", "H_autocorr_lag5",
        ]),
    }

    n = len(feature_names)
    groups["full"] = list(range(n))

    groups["hierarchy_basic_plus_derivatives"] = sorted(
        set(groups["hierarchy_basic"] + groups["hierarchy_derivatives"])
    )

    groups["hierarchy_plus_signal"] = sorted(
        set(groups["hierarchy_basic"] + groups["hierarchy_derivatives"] + groups["signal"])
    )

    groups["energy_plus_seam"] = sorted(
        set(groups["energy"] + groups["seam"])
    )

    groups["all_except_energy"] = sorted(
        set(range(n)) - set(groups["energy"])
    )

    groups["all_except_hierarchy"] = sorted(
        set(range(n)) - set(groups["hierarchy_basic"] + groups["hierarchy_derivatives"])
    )

    return groups


def fit_eval_model(X_train, X_val, y_train, y_val, seed=42):
    model = make_fast_classifier(seed)
    model.fit(X_train, y_train)

    pred = model.predict(X_val)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_val)[:, 1]
    else:
        prob = pred.astype(float)

    return classification_metrics(y_val, pred, prob), model


def hmin_threshold_baseline(X_train, X_val, y_train, y_val, feature_names):
    h_idx = feature_names.index("H_min")
    thresholds = np.linspace(1.5, 8.0, 261)

    best = None

    for th in thresholds:
        pred_train = (X_train[:, h_idx] < th).astype(int)
        m = classification_metrics(y_train, pred_train)
        row = {"threshold": float(th), **m}

        if best is None or row["f1_eject"] > best["f1_eject"]:
            best = row

    th = best["threshold"]
    pred_val = (X_val[:, h_idx] < th).astype(int)
    val = classification_metrics(y_val, pred_val)

    return {
        "threshold_fit_on_train": float(th),
        "train_best": best,
        "validation": val,
    }


def load_features(feature_file):
    data = np.load(feature_file, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(int)
    feature_names = [str(x) for x in data["feature_names"]]
    return X, y, feature_names


def load_initial_condition_matrix(records_file, y_reference=None):
    with open(records_file, "r") as f:
        records = json.load(f)

    X_ic = []
    y_ic = []

    for r in records:
        status = r.get("status")
        if status == "numerical_error":
            continue

        ic = r.get("initial_conditions", {})

        row = [
            ic.get("m0", np.nan),
            ic.get("m1", np.nan),
            ic.get("m2", np.nan),
            ic.get("a_inner", np.nan),
            ic.get("e_inner", np.nan),
            ic.get("inc_inner", np.nan),
            ic.get("Omega_inner", np.nan),
            ic.get("omega_inner", np.nan),
            ic.get("M_inner", np.nan),
            ic.get("a_outer", np.nan),
            ic.get("e_outer", np.nan),
            ic.get("inc_outer", np.nan),
            ic.get("Omega_outer", np.nan),
            ic.get("omega_outer", np.nan),
            ic.get("M_outer", np.nan),
        ]

        X_ic.append(row)
        y_ic.append(1 if status == "ejected" else 0)

    X_ic = np.asarray(X_ic, dtype=float)
    y_ic = np.asarray(y_ic, dtype=int)

    m0, m1, m2 = X_ic[:, 0], X_ic[:, 1], X_ic[:, 2]
    a_in, e_in = X_ic[:, 3], X_ic[:, 4]
    a_out, e_out = X_ic[:, 9], X_ic[:, 10]
    inc_out = X_ic[:, 11]

    mass_ratio_outer = m2 / (m0 + m1 + 1e-30)
    sep_ratio = a_out / (a_in + 1e-30)
    peri_apo_ratio = (a_out * (1 - e_out)) / (a_in * (1 + e_in) + 1e-30)

    X_ic_aug = np.column_stack([
        X_ic,
        mass_ratio_outer,
        sep_ratio,
        peri_apo_ratio,
        np.cos(inc_out),
        np.sin(inc_out),
    ])

    if y_reference is not None and len(y_reference) == len(y_ic):
        label_match_fraction = float(np.mean(y_reference == y_ic))
    else:
        label_match_fraction = None

    feature_names_ic = [
        "m0", "m1", "m2",
        "a_inner", "e_inner", "inc_inner", "Omega_inner", "omega_inner", "M_inner",
        "a_outer", "e_outer", "inc_outer", "Omega_outer", "omega_outer", "M_outer",
        "m2/(m0+m1)", "a_outer/a_inner",
        "outer_pericenter/inner_apocenter",
        "cos(inc_outer)", "sin(inc_outer)",
    ]

    return X_ic_aug, y_ic, feature_names_ic, label_match_fraction


def run_repeated_splits(X, y, feature_names, n_splits=10):
    rows = []
    hmin_rows = []

    seeds = list(range(1000, 1000 + n_splits))

    for s in seeds:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=s,
            stratify=y,
        )

        m, _ = fit_eval_model(X_train, X_val, y_train, y_val, seed=s)
        m["split_seed"] = s
        rows.append(m)

        hb = hmin_threshold_baseline(X_train, X_val, y_train, y_val, feature_names)
        hb_row = {
            "split_seed": s,
            "threshold": hb["threshold_fit_on_train"],
            **hb["validation"],
        }
        hmin_rows.append(hb_row)

        print(
            f"seed={s} | "
            f"acc={m['accuracy']:.4f} auc={m.get('auc', np.nan):.4f} "
            f"recall={m['recall_eject']:.4f} f1={m['f1_eject']:.4f} | "
            f"Hmin_acc={hb_row['accuracy']:.4f}"
        )

    return rows, hmin_rows


def run_ablation(X, y, feature_names):
    groups = feature_groups(feature_names)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=999,
        stratify=y,
    )

    rows = []

    for group_name, cols in groups.items():
        if len(cols) == 0:
            continue

        Xtr = X_train[:, cols]
        Xva = X_val[:, cols]

        m, _ = fit_eval_model(Xtr, Xva, y_train, y_val, seed=999)
        row = {
            "feature_group": group_name,
            "n_features": len(cols),
            **m,
        }

        rows.append(row)

        print(
            f"{group_name:32s} | "
            f"n={len(cols):2d} | "
            f"acc={m['accuracy']:.4f} auc={m.get('auc', np.nan):.4f} "
            f"recall={m['recall_eject']:.4f} f1={m['f1_eject']:.4f}"
        )

    return rows


def run_initial_condition_baselines(records_file, y_reference=None):
    X_ic, y_ic, ic_names, label_match_fraction = load_initial_condition_matrix(
        records_file, y_reference=y_reference
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_ic,
        y_ic,
        test_size=0.30,
        random_state=999,
        stratify=y_ic,
    )

    # Logistic baseline
    ic_logit = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=999)),
    ])

    ic_logit.fit(X_train, y_train)
    pred = ic_logit.predict(X_val)
    prob = ic_logit.predict_proba(X_val)[:, 1]
    ic_logit_metrics = classification_metrics(y_val, pred, prob)

    # Tree baseline
    ic_model = make_fast_classifier(999)
    ic_model.fit(X_train, y_train)
    pred2 = ic_model.predict(X_val)
    prob2 = ic_model.predict_proba(X_val)[:, 1]
    ic_tree_metrics = classification_metrics(y_val, pred2, prob2)

    return {
        "n": int(len(y_ic)),
        "label_match_fraction": label_match_fraction,
        "logistic_initial_conditions": ic_logit_metrics,
        "tree_initial_conditions": ic_tree_metrics,
        "features_used": ic_names,
    }


def run_robustness_analysis(
    feature_file,
    records_file=None,
    output_json="results/robustness_results.json",
    ablation_csv="results/ablation_results.csv",
    repeated_csv="results/repeated_split_results.csv",
    hmin_csv="results/hmin_repeated_split_results.csv",
    n_splits=10,
):
    feature_file = Path(feature_file)
    output_json = Path(output_json)
    ablation_csv = Path(ablation_csv)
    repeated_csv = Path(repeated_csv)
    hmin_csv = Path(hmin_csv)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    ablation_csv.parent.mkdir(parents=True, exist_ok=True)
    repeated_csv.parent.mkdir(parents=True, exist_ok=True)
    hmin_csv.parent.mkdir(parents=True, exist_ok=True)

    X, y, feature_names = load_features(feature_file)

    print("Loaded:", feature_file)
    print("X shape:", X.shape)
    print("Ejection fraction:", float(y.mean()))
    print("Feature count:", len(feature_names))

    print("\n================================================")
    print("RUNNING REPEATED SPLIT TESTS")
    print("================================================")
    repeated_rows, hmin_rows = run_repeated_splits(
        X, y, feature_names, n_splits=n_splits
    )

    repeated_summary = summarize_metric_rows(repeated_rows)
    hmin_summary = summarize_metric_rows(hmin_rows)

    pd.DataFrame(repeated_rows).to_csv(repeated_csv, index=False)
    pd.DataFrame(hmin_rows).to_csv(hmin_csv, index=False)

    print("\nRepeated split full-model summary:")
    print(json.dumps(repeated_summary, indent=2))

    print("\nRepeated split H_min-baseline summary:")
    print(json.dumps(hmin_summary, indent=2))

    print("\n================================================")
    print("RUNNING FEATURE ABLATION STUDY")
    print("================================================")
    ablation_rows = run_ablation(X, y, feature_names)
    pd.DataFrame(ablation_rows).to_csv(ablation_csv, index=False)

    initial_condition_results = None

    if records_file is not None and Path(records_file).exists():
        print("\n================================================")
        print("RUNNING INITIAL-CONDITION-ONLY BASELINE")
        print("================================================")
        initial_condition_results = run_initial_condition_baselines(
            records_file, y_reference=y
        )
        print(json.dumps(initial_condition_results, indent=2))
    else:
        print("\nNo simulation records file found. Skipping initial-condition baselines.")

    robustness_results = {
        "dataset": {
            "feature_file": str(feature_file),
            "n": int(len(y)),
            "n_features": int(X.shape[1]),
            "eject_fraction": float(y.mean()),
            "feature_names": feature_names,
        },
        "repeated_splits": {
            "n_splits": int(n_splits),
            "rows": repeated_rows,
            "summary": repeated_summary,
        },
        "hmin_baseline_repeated_splits": {
            "rows": hmin_rows,
            "summary": hmin_summary,
        },
        "ablation": {
            "rows": ablation_rows,
        },
        "initial_condition_baselines": initial_condition_results,
    }

    with output_json.open("w") as f:
        json.dump(robustness_results, f, indent=2)

    print("\n================================================")
    print("SAVED OUTPUTS")
    print("================================================")
    print(output_json)
    print(ablation_csv)
    print(repeated_csv)
    print(hmin_csv)

    return robustness_results
