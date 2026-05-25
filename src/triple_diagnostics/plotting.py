"""
Plotting utilities for hierarchical-triple ejection diagnostics.

This module creates the main validation dashboard from saved feature,
metric, model, and threshold-sweep files.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split


# ============================================================
# STYLE CONSTANTS
# ============================================================

FIG_BG = "#0d1117"
AX_BG = "#161b22"
GRID = "#30363d"
TEXT = "#c9d1d9"

BLUE = "#58a6ff"
GREEN = "#7ee787"
RED = "#f85149"
ORANGE = "#ffa657"
PURPLE = "#bc8cff"


def style_ax(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, alpha=0.22, color=GRID, linestyle="--", linewidth=0.6)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def finite_array(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def safe_hist(ax, data, **kwargs):
    data = finite_array(data)
    if len(data) > 0:
        ax.hist(data, **kwargs)


def percentile_limits(x, lo=1, hi=99, fallback=(0.0, 1.0)):
    x = finite_array(x)
    if len(x) == 0:
        return fallback
    a, b = np.percentile(x, [lo, hi])
    if not np.isfinite(a) or not np.isfinite(b) or a == b:
        return fallback
    return float(a), float(b)


def get_feature_importance(model, feature_names):
    """
    Try XGBoost first, then RandomForest.
    VotingClassifier stores fitted estimators in named_estimators_.
    """
    importances = None
    source = "Model"

    try:
        boost = model.named_estimators_["boost"]
        if hasattr(boost, "feature_importances_"):
            importances = boost.feature_importances_
            source = "XGBoost"
    except Exception:
        pass

    if importances is None:
        try:
            rf = model.named_estimators_["rf"]
            if hasattr(rf, "feature_importances_"):
                importances = rf.feature_importances_
                source = "RandomForest"
        except Exception:
            pass

    if importances is None:
        importances = np.zeros(len(feature_names), dtype=float)
        source = "Unavailable"

    order = np.argsort(importances)[::-1]
    return np.asarray(importances), order, source


def load_threshold_sweep(sweep_file):
    thresholds, f1, acc = [], [], []

    if sweep_file is not None and os.path.exists(sweep_file):
        with open(sweep_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                thresholds.append(float(row["threshold"]))
                f1.append(float(row["f1_eject"]))
                acc.append(float(row["accuracy"]))

    thresholds = np.asarray(thresholds, dtype=float)
    f1 = np.asarray(f1, dtype=float)
    acc = np.asarray(acc, dtype=float)

    if len(thresholds) > 0:
        best_idx = int(np.argmax(f1))
        best_h = float(thresholds[best_idx])
        best_f1 = float(f1[best_idx])
    else:
        best_h = 2.5
        best_f1 = np.nan

    return thresholds, f1, acc, best_h, best_f1


def _load_model_bundle(model_file):
    bundle = joblib.load(model_file)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"]
    return bundle


def make_validation_dashboard(
    feature_file,
    metrics_file,
    model_file,
    sweep_file=None,
    output_fig="figures/Accuracy_10k.png",
    test_size=0.30,
    split_seed=999,
    dpi=250,
):
    """
    Generate the complete validation dashboard.

    Parameters
    ----------
    feature_file:
        NPZ file containing X, y, and feature_names.
    metrics_file:
        JSON file containing train/validation metrics.
    model_file:
        Joblib model bundle containing key 'model', or direct model object.
    sweep_file:
        Optional CSV threshold sweep file.
    output_fig:
        Output PNG filename.
    test_size:
        Validation fraction.
    split_seed:
        Random seed used for train/validation split.
    dpi:
        Figure DPI.
    """

    feature_file = Path(feature_file)
    metrics_file = Path(metrics_file)
    model_file = Path(model_file)
    output_fig = Path(output_fig)

    if not feature_file.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_file}")
    if not metrics_file.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_file}")
    if not model_file.exists():
        raise FileNotFoundError(f"Missing model file: {model_file}")

    data = np.load(feature_file, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(int)
    feature_names = [str(x) for x in data["feature_names"]]

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    model = _load_model_bundle(model_file)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=split_seed,
        stratify=y,
    )

    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_val = model.predict_proba(X_val)[:, 1]
    y_pred_val = model.predict(X_val)

    fpr, tpr, _ = roc_curve(y_val, y_prob_val)
    roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(y_val, y_pred_val, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    n_total = len(y)
    n_train = len(y_train)
    n_val = len(y_val)
    eject_rate = 100.0 * np.mean(y)
    stable_rate = 100.0 * (1.0 - np.mean(y))

    val_metrics = metrics["validation"]
    train_metrics = metrics["train"]

    def col(name):
        return feature_names.index(name)

    def get(name):
        return X[:, col(name)]

    # Feature arrays
    H_min = get("H_min")
    H_below_frac = get("H_below_frac")
    H_crossings = get("H_crossings")
    H_first_breach_frac = get("H_first_breach_frac")
    H_slope = get("H_slope")
    H_curvature = get("H_curvature")

    abs_d2H_max = get("abs_d2H_max")
    psi_mean = get("psi_mean")

    E_exchange_norm = get("E_exchange_norm")
    E_slope_norm = get("E_slope_norm")

    Omega_slope = get("Omega_slope")
    H_entropy = get("H_entropy")

    stable_mask = y == 0
    eject_mask = y == 1

    # Feature importance
    importances, importance_order, importance_source = get_feature_importance(
        model, feature_names
    )
    top_k = 12
    top_idx = importance_order[:top_k]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = importances[top_idx]

    # Threshold sweep
    sweep_thresholds, sweep_f1, sweep_acc, best_H, best_H_f1 = load_threshold_sweep(
        sweep_file
    )

    # Breach / lead proxy
    breached = H_below_frac > 0
    ejected_breached = eject_mask & breached

    lead_fraction_proxy = 1.0 - H_first_breach_frac[ejected_breached]
    lead_fraction_proxy = finite_array(lead_fraction_proxy)
    lead_fraction_proxy = lead_fraction_proxy[
        (lead_fraction_proxy >= 0) & (lead_fraction_proxy <= 1)
    ]

    zombie_ejected_pct = (
        100 * np.mean(H_below_frac[eject_mask] > 0) if np.sum(eject_mask) else 0
    )
    zombie_stable_pct = (
        100 * np.mean(H_below_frac[stable_mask] > 0) if np.sum(stable_mask) else 0
    )

    prob_stable_true = y_prob_val[y_val == 0]
    prob_eject_true = y_prob_val[y_val == 1]

    # Figure
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(24, 16), facecolor=FIG_BG)

    title = (
        "Enhanced Topological Three-Body Framework: Complete Validation Results\n"
        f"{n_total:,} Hierarchical Triples | "
        f"Validation Acc={100 * val_metrics['accuracy']:.2f}% | "
        f"AUC={val_metrics.get('auc', roc_auc):.4f}"
    )

    fig.suptitle(title, fontsize=20, color=TEXT, weight="bold", y=0.975)

    # 1 ROC
    ax1 = plt.subplot(4, 5, 1)
    ax1.plot(fpr, tpr, color=GREEN, lw=2.5, label=f"AUC = {roc_auc:.4f}")
    ax1.plot([0, 1], [0, 1], color=ORANGE, lw=1.2, ls="--", alpha=0.7)
    ax1.fill_between(fpr, 0, tpr, color=GREEN, alpha=0.18)
    ax1.set_title("ROC Curve (Validation)", fontsize=11, weight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    style_ax(ax1)

    # 2 Confusion matrix
    ax2 = plt.subplot(4, 5, 2)
    cm_plot = ax2.imshow(cm, cmap="Blues")
    ax2.set_title("Confusion Matrix", fontsize=11, weight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Stable", "Ejected"], color=TEXT)
    ax2.set_yticklabels(["Stable", "Ejected"], color=TEXT)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "#0d1117"
            ax2.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color=color,
                fontsize=16,
                weight="bold",
            )

    plt.colorbar(cm_plot, ax=ax2, fraction=0.046, pad=0.04)
    style_ax(ax2)

    # 3 Top features
    ax3 = plt.subplot(4, 5, 3)
    ypos = np.arange(len(top_names))[::-1]
    colors = [GREEN if i == 0 else BLUE for i in range(len(top_names))]
    ax3.barh(ypos, top_vals[::-1], color=colors[::-1], alpha=0.85)
    ax3.set_yticks(ypos)
    ax3.set_yticklabels(top_names[::-1], fontsize=8, color=TEXT)
    ax3.set_xlabel("Importance")
    ax3.set_title(f"Top Features ({importance_source})", fontsize=11, weight="bold")
    style_ax(ax3)

    # 4 Train vs validation
    ax4 = plt.subplot(4, 5, 4)
    metric_names = ["Accuracy", "Precision", "Recall", "Specificity", "F1"]
    train_vals = [
        train_metrics["accuracy"],
        train_metrics["precision_eject"],
        train_metrics["recall_eject"],
        train_metrics["specificity_stable"],
        train_metrics["f1_eject"],
    ]
    val_vals = [
        val_metrics["accuracy"],
        val_metrics["precision_eject"],
        val_metrics["recall_eject"],
        val_metrics["specificity_stable"],
        val_metrics["f1_eject"],
    ]
    x = np.arange(len(metric_names))
    w = 0.36
    ax4.bar(x - w / 2, train_vals, width=w, color=BLUE, alpha=0.75, label="Training")
    ax4.bar(x + w / 2, val_vals, width=w, color=GREEN, alpha=0.75, label="Validation")
    ax4.axhline(0.90, color=ORANGE, ls="--", lw=1, alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metric_names, rotation=30, ha="right", fontsize=8)
    ax4.set_ylim(0.80, 1.02)
    ax4.set_ylabel("Score")
    ax4.set_title("Train vs Validation Metrics", fontsize=11, weight="bold")
    ax4.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    style_ax(ax4)

    # 5 Prediction confidence
    ax5 = plt.subplot(4, 5, 5)
    safe_hist(ax5, prob_stable_true, bins=35, density=True, alpha=0.65, color=BLUE, label="True Stable")
    safe_hist(ax5, prob_eject_true, bins=35, density=True, alpha=0.65, color=RED, label="True Ejected")
    ax5.axvline(0.5, color=ORANGE, ls="--", lw=2, label="Threshold")
    ax5.set_title("Prediction Confidence", fontsize=11, weight="bold")
    ax5.set_xlabel("P(Ejection)")
    ax5.set_ylabel("Density")
    ax5.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    style_ax(ax5)

    # 6 H_min
    ax6 = plt.subplot(4, 5, 6)
    h_hi = min(10, np.nanpercentile(H_min, 99))
    safe_hist(ax6, H_min[stable_mask], bins=60, density=True, alpha=0.65, color=BLUE, label="Stable", range=(0, h_hi))
    safe_hist(ax6, H_min[eject_mask], bins=60, density=True, alpha=0.65, color=RED, label="Ejected", range=(0, h_hi))
    ax6.axvline(2.5, color=ORANGE, ls="--", lw=2.0, label="Paper Hcrit=2.5")
    ax6.axvline(best_H, color=GREEN, ls=":", lw=2.0, label=f"Best={best_H:.2f}")
    ax6.set_xlim(0, h_hi)
    ax6.set_title("Observation II: Manifold Convergence", fontsize=11, weight="bold")
    ax6.set_xlabel("H_min")
    ax6.set_ylabel("Density")
    ax6.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=7)
    style_ax(ax6)

    # 7 Energy exchange
    ax7 = plt.subplot(4, 5, 7)
    logE = np.log10(E_exchange_norm + 1e-30)
    lo, hi = percentile_limits(logE)
    safe_hist(ax7, logE[stable_mask], bins=55, density=True, alpha=0.65, color=BLUE, label="Stable", range=(lo, hi))
    safe_hist(ax7, logE[eject_mask], bins=55, density=True, alpha=0.65, color=RED, label="Ejected", range=(lo, hi))
    ax7.set_xlim(lo, hi)
    ax7.set_title("Observation I: Pairwise Energy Exchange", fontsize=11, weight="bold")
    ax7.set_xlabel("log10(E_exchange_norm)")
    ax7.set_ylabel("Density")
    ax7.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    style_ax(ax7)

    # 8 Seam evolution
    ax8 = plt.subplot(4, 5, 8)
    lo, hi = percentile_limits(Omega_slope)
    safe_hist(ax8, Omega_slope[stable_mask], bins=55, density=True, alpha=0.65, color=BLUE, label="Stable", range=(lo, hi))
    safe_hist(ax8, Omega_slope[eject_mask], bins=55, density=True, alpha=0.65, color=RED, label="Ejected", range=(lo, hi))
    ax8.set_xlim(lo, hi)
    ax8.set_title("Topological Seam Evolution", fontsize=11, weight="bold")
    ax8.set_xlabel("Omega_res Slope")
    ax8.set_ylabel("Density")
    ax8.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    style_ax(ax8)

    # 9 D2H
    ax9 = plt.subplot(4, 5, 9)
    lo, hi = percentile_limits(abs_d2H_max, fallback=(0, 1))
    safe_hist(ax9, abs_d2H_max[stable_mask], bins=55, density=True, alpha=0.65, color=BLUE, label="Stable", range=(lo, hi))
    safe_hist(ax9, abs_d2H_max[eject_mask], bins=55, density=True, alpha=0.65, color=RED, label="Ejected", range=(lo, hi))
    ax9.set_xlim(lo, hi)
    ax9.set_title("Hierarchy Acceleration D²H", fontsize=11, weight="bold")
    ax9.set_xlabel("max |d²H/dt²|")
    ax9.set_ylabel("Density")
    ax9.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    style_ax(ax9)

    # 10 Phase space
    ax10 = plt.subplot(4, 5, 10)
    h_plot = np.clip(H_min, 0, np.nanpercentile(H_min, 99))
    d2_plot = np.log10(abs_d2H_max + 1e-30)
    lo_d2, hi_d2 = percentile_limits(d2_plot)
    ax10.scatter(h_plot[stable_mask], d2_plot[stable_mask], s=8, color=BLUE, alpha=0.35, label="Stable")
    ax10.scatter(h_plot[eject_mask], d2_plot[eject_mask], s=8, color=RED, alpha=0.35, label="Ejected")
    ax10.axvline(2.5, color=ORANGE, ls="--", lw=2)
    ax10.set_xlim(0, h_hi)
    ax10.set_ylim(lo_d2, hi_d2)
    ax10.set_title("Topological Phase Space", fontsize=11, weight="bold")
    ax10.set_xlabel("H_min")
    ax10.set_ylabel("log10 max |D²H|")
    ax10.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    style_ax(ax10)

    # 11 Lead proxy
    ax11 = plt.subplot(4, 5, 11)
    if len(lead_fraction_proxy) > 0:
        ax11.hist(
            100 * lead_fraction_proxy,
            bins=45,
            color=PURPLE,
            alpha=0.78,
            edgecolor="#ffffff",
            linewidth=0.3,
        )
        med_lead = np.median(lead_fraction_proxy) * 100
        mean_lead = np.mean(lead_fraction_proxy) * 100
        ax11.axvline(med_lead, color=ORANGE, ls="--", lw=2, label=f"Median={med_lead:.1f}%")
        ax11.axvline(mean_lead, color=GREEN, ls="--", lw=2, label=f"Mean={mean_lead:.1f}%")
    else:
        med_lead = 0.0
        mean_lead = 0.0
        ax11.text(0.5, 0.5, "No breach proxy data", ha="center", va="center", color=TEXT)
    ax11.set_title("Early Warning Lead Proxy", fontsize=11, weight="bold")
    ax11.set_xlabel("Advance Fraction Proxy (%)")
    ax11.set_ylabel("Count")
    ax11.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    style_ax(ax11)

    # 12 Lead vs hierarchy
    ax12 = plt.subplot(4, 5, 12)
    if np.sum(ejected_breached) > 0:
        ax12.scatter(
            H_min[ejected_breached],
            100 * (1.0 - H_first_breach_frac[ejected_breached]),
            s=10,
            color=RED,
            alpha=0.35,
            label="Ejected + Breached",
        )
    ax12.axvline(2.5, color=ORANGE, ls="--", lw=2)
    ax12.set_xlim(0, h_hi)
    ax12.set_ylim(0, 105)
    ax12.set_title("Warning Lead vs Hierarchy", fontsize=11, weight="bold")
    ax12.set_xlabel("H_min")
    ax12.set_ylabel("Advance Proxy (%)")
    ax12.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    style_ax(ax12)

    # 13 Metastable / breach
    ax13 = plt.subplot(4, 5, 13)
    bars = [zombie_stable_pct, zombie_ejected_pct]
    labels = ["Stable\nSystems", "Ejected\nSystems"]
    ax13.bar(labels, bars, color=[BLUE, RED], alpha=0.80, edgecolor="white", linewidth=1.0)
    for i, v in enumerate(bars):
        ax13.text(i, v + 2, f"{v:.1f}%", ha="center", color=TEXT, fontsize=12, weight="bold")
    ax13.set_ylim(0, 105)
    ax13.set_ylabel("Breach Fraction (%)")
    ax13.set_title("Metastable / Breach Detection", fontsize=11, weight="bold")
    style_ax(ax13)

    # 14 Spectral entropy
    ax14 = plt.subplot(4, 5, 14)
    safe_hist(ax14, H_entropy[stable_mask], bins=50, density=True, alpha=0.65, color=BLUE, label="Stable")
    safe_hist(ax14, H_entropy[eject_mask], bins=50, density=True, alpha=0.65, color=RED, label="Ejected")
    ax14.set_title("Chaos Measure: H Spectral Entropy", fontsize=11, weight="bold")
    ax14.set_xlabel("Spectral Entropy H(t)")
    ax14.set_ylabel("Density")
    ax14.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    style_ax(ax14)

    # 15 Threshold sweep
    ax15 = plt.subplot(4, 5, 15)
    if len(sweep_thresholds) > 0:
        ax15.plot(sweep_thresholds, sweep_f1, color=GREEN, lw=2.2, label="F1")
        ax15.plot(sweep_thresholds, sweep_acc, color=BLUE, lw=1.6, label="Accuracy")
        ax15.axvline(2.5, color=ORANGE, ls="--", lw=2, label="Hcrit=2.5")
        ax15.axvline(best_H, color=RED, ls=":", lw=2, label=f"Best={best_H:.2f}")
    else:
        ax15.text(0.5, 0.5, "No sweep file", ha="center", va="center", color=TEXT)
    ax15.set_title("Empirical Hcrit Sweep", fontsize=11, weight="bold")
    ax15.set_xlabel("H_min Threshold")
    ax15.set_ylabel("Score")
    ax15.set_ylim(0, 1.05)
    ax15.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=7)
    style_ax(ax15)

    # 16 Psi
    ax16 = plt.subplot(4, 5, 16)
    safe_hist(ax16, psi_mean[stable_mask], bins=50, density=True, alpha=0.65, color=BLUE, label="Stable")
    safe_hist(ax16, psi_mean[eject_mask], bins=50, density=True, alpha=0.65, color=RED, label="Ejected")
    ax16.axvline(0.5, color=ORANGE, ls="--", lw=2)
    ax16.set_title("Tanh-Sigmoid Filter Ψ(|dH|)", fontsize=11, weight="bold")
    ax16.set_xlabel("Mean Ψ")
    ax16.set_ylabel("Density")
    ax16.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    style_ax(ax16)

    # 17 Crossings
    ax17 = plt.subplot(4, 5, 17)
    max_cross = min(25, int(np.nanpercentile(H_crossings, 99)) + 1)
    bins = np.arange(0, max_cross + 2) - 0.5
    ax17.hist(H_crossings[stable_mask], bins=bins, alpha=0.65, color=BLUE, label="Stable")
    ax17.hist(H_crossings[eject_mask], bins=bins, alpha=0.65, color=RED, label="Ejected")
    ax17.set_xlim(-0.5, max_cross + 0.5)
    ax17.set_title("Critical Boundary Crossings", fontsize=11, weight="bold")
    ax17.set_xlabel("Count H(t)<Hcrit Entries")
    ax17.set_ylabel("Systems")
    ax17.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    style_ax(ax17)

    # 18 Energy slope
    ax18 = plt.subplot(4, 5, 18)
    log_slope = np.log10(E_slope_norm + 1e-30)
    lo, hi = percentile_limits(log_slope)
    safe_hist(ax18, log_slope[stable_mask], bins=50, density=True, alpha=0.65, color=BLUE, label="Stable", range=(lo, hi))
    safe_hist(ax18, log_slope[eject_mask], bins=50, density=True, alpha=0.65, color=RED, label="Ejected", range=(lo, hi))
    ax18.set_xlim(lo, hi)
    ax18.set_title("Pair-Energy Transfer Rate", fontsize=11, weight="bold")
    ax18.set_xlabel("log10(E_slope_norm)")
    ax18.set_ylabel("Density")
    ax18.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    style_ax(ax18)

    # 19 Curvature vs seam
    ax19 = plt.subplot(4, 5, 19)
    curv_plot = np.log10(np.abs(H_curvature) + 1e-30)
    om_plot = np.log10(np.abs(Omega_slope) + 1e-30)
    xlo, xhi = percentile_limits(curv_plot)
    ylo, yhi = percentile_limits(om_plot)
    ax19.scatter(curv_plot[stable_mask], om_plot[stable_mask], s=8, color=BLUE, alpha=0.35, label="Stable")
    ax19.scatter(curv_plot[eject_mask], om_plot[eject_mask], s=8, color=RED, alpha=0.35, label="Ejected")
    ax19.set_xlim(xlo, xhi)
    ax19.set_ylim(ylo, yhi)
    ax19.set_title("Curvature vs Seam Drift", fontsize=11, weight="bold")
    ax19.set_xlabel("log10 |H curvature|")
    ax19.set_ylabel("log10 |Omega slope|")
    ax19.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    style_ax(ax19)

    # 20 Summary
    ax20 = plt.subplot(4, 5, 20)
    ax20.axis("off")
    ax20.set_facecolor(AX_BG)

    gap = train_metrics["accuracy"] - val_metrics["accuracy"]

    summary = f"""
TOPOLOGICAL THREE-BODY FRAMEWORK
Large-Scale Validation Summary
══════════════════════════════════════

DATASET
  Total systems:       {n_total:,}
  Training:            {n_train:,}
  Validation:          {n_val:,}

  Ejection rate:       {eject_rate:6.2f}%
  Stability rate:      {stable_rate:6.2f}%

VALIDATION PERFORMANCE
  Accuracy:            {100*val_metrics['accuracy']:6.2f}%
  AUC-ROC:             {val_metrics.get('auc', roc_auc):6.4f}
  F1 Score:            {val_metrics['f1_eject']:6.4f}

  Precision:           {100*val_metrics['precision_eject']:6.2f}%
  Recall:              {100*val_metrics['recall_eject']:6.2f}%
  Specificity:         {100*val_metrics['specificity_stable']:6.2f}%

CONFUSION MATRIX
  TN Stable:           {tn:,}
  FP False Alarm:      {fp:,}
  FN Missed Ejection:  {fn:,}
  TP Ejection:         {tp:,}

GENERALIZATION
  Train Acc:           {100*train_metrics['accuracy']:6.2f}%
  Val Acc:             {100*val_metrics['accuracy']:6.2f}%
  Gap:                 {100*gap:+6.2f}%

TOPOLOGICAL DIAGNOSTICS
  Paper Hcrit:         2.500
  Best Hcrit Sweep:    {best_H:6.3f}
  Best Hcrit F1:       {best_H_f1:6.3f}

  Ejected breached:    {zombie_ejected_pct:6.2f}%
  Stable breached:     {zombie_stable_pct:6.2f}%

EARLY WARNING PROXY
  Warned ejections:    {len(lead_fraction_proxy):,}
  Median advance:      {med_lead:6.2f}%
  Mean advance:        {mean_lead:6.2f}%

TOP FEATURE
  {top_names[0] if len(top_names) else 'N/A'}
  importance={top_vals[0] if len(top_vals) else 0:.4f}
"""

    ax20.text(
        0.03,
        0.98,
        summary,
        color=TEXT,
        family="monospace",
        va="top",
        fontsize=9.2,
        bbox=dict(
            boxstyle="round",
            facecolor="#0d1117",
            edgecolor=ORANGE,
            lw=2.0,
            pad=1.0,
        ),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_fig, dpi=dpi, facecolor=FIG_BG, bbox_inches="tight")
    plt.close(fig)

    return {
        "output_fig": str(output_fig),
        "n_total": int(n_total),
        "n_train": int(n_train),
        "n_validation": int(n_val),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "best_H": float(best_H),
        "best_H_f1": float(best_H_f1) if np.isfinite(best_H_f1) else None,
    }
