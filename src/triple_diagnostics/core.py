"""
Core simulation, diagnostic, feature extraction, and model utilities for
hierarchical-triple ejection classification.

This module is the cleaned GitHub-ready version of the original Colab
simulation/training code. It contains no notebook state and no inline pip
installation commands.

Main public functions
---------------------
- generate_dataset(cfg)
- sweep_hmin_threshold(X, y, feature_names, output_csv)
- train_evaluate(X, y, cfg)
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import rebound
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ============================================================
# CONFIGURATION
# ============================================================

TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class SimulationConfig:
    """
    Configuration for the fiducial hierarchical-triple simulation suite.
    """

    # Dataset
    n_systems: int = 10000
    seed: int = 42

    # Sampling
    mass_range: tuple[float, float] = (0.5, 2.0)
    a_in_range: tuple[float, float] = (0.8, 1.2)
    a_out_range: tuple[float, float] = (3.0, 12.0)
    e_in_range: tuple[float, float] = (0.0, 0.4)
    e_out_range: tuple[float, float] = (0.0, 0.7)
    inc_range_deg: tuple[float, float] = (0.0, 60.0)

    # Integration
    n_samples: int = 160
    t_max_outer_periods: float = 100.0
    ejection_radius_factor: float = 5.0
    collision_radius_factor: float = 0.01
    max_relative_energy_error: float = 1.0e-8

    # Diagnostics
    H_critical: float = 2.5
    sigmoid_kappa: float = 4.0

    # ML
    test_size: float = 0.30
    split_seed: int = 999

    # Output files
    feature_file: str = "results/hts10k_features.npz"
    records_file: str = "results/simulation_records.json"
    checkpoint_records_file: str = "results/simulation_records_checkpoint.json"
    metrics_file: str = "results/metrics_10k.json"
    model_file: str = "results/model_10k.joblib"
    sweep_file: str = "results/Hcrit_sweep_10k.csv"

    # Runtime
    checkpoint_every: int = 500


@dataclass(frozen=True)
class TripleIC:
    """
    Jacobi-style hierarchical triple initial condition.

    Bodies 0 and 1 form the inner binary.
    Body 2 is the tertiary on the outer orbit.

    Units:
    - masses: Msun
    - semimajor axes: AU
    - angles: radians
    """

    m0: float
    m1: float
    m2: float

    a_inner: float
    e_inner: float
    inc_inner: float
    Omega_inner: float
    omega_inner: float
    M_inner: float

    a_outer: float
    e_outer: float
    inc_outer: float
    Omega_outer: float
    omega_outer: float
    M_outer: float


# ============================================================
# INITIAL CONDITIONS
# ============================================================


def sample_triple_ic(system_id: int, cfg: SimulationConfig) -> TripleIC:
    """
    Deterministic independent RNG per system_id.

    This makes the run reproducible and independent of parallelization order.
    """
    ss = np.random.SeedSequence([cfg.seed, int(system_id)])
    rng = np.random.default_rng(ss)

    return TripleIC(
        m0=float(rng.uniform(*cfg.mass_range)),
        m1=float(rng.uniform(*cfg.mass_range)),
        m2=float(rng.uniform(*cfg.mass_range)),
        a_inner=float(rng.uniform(*cfg.a_in_range)),
        e_inner=float(rng.uniform(*cfg.e_in_range)),
        inc_inner=0.0,
        Omega_inner=0.0,
        omega_inner=float(rng.uniform(0.0, TWOPI)),
        M_inner=float(rng.uniform(0.0, TWOPI)),
        a_outer=float(rng.uniform(*cfg.a_out_range)),
        e_outer=float(rng.uniform(*cfg.e_out_range)),
        inc_outer=float(np.deg2rad(rng.uniform(*cfg.inc_range_deg))),
        Omega_outer=float(rng.uniform(0.0, TWOPI)),
        omega_outer=float(rng.uniform(0.0, TWOPI)),
        M_outer=float(rng.uniform(0.0, TWOPI)),
    )


def outer_period_years(ic: TripleIC) -> float:
    """
    Keplerian outer-period estimate in years using AU/Msun units.

    REBOUND with units ('yr', 'AU', 'Msun') uses G=4*pi^2 internally.
    Period in years is sqrt(a^3/Mtot).
    """
    return float(np.sqrt(ic.a_outer**3 / (ic.m0 + ic.m1 + ic.m2)))


def make_sim(ic: TripleIC) -> rebound.Simulation:
    """
    Create a REBOUND IAS15 simulation with Jacobi-style setup.

    Body 1 orbits body 0.
    Body 2 orbits the center of mass of the inner pair.
    """
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.integrator = "ias15"

    # Inner binary
    sim.add(m=ic.m0)
    sim.add(
        m=ic.m1,
        a=ic.a_inner,
        e=ic.e_inner,
        inc=ic.inc_inner,
        Omega=ic.Omega_inner,
        omega=ic.omega_inner,
        M=ic.M_inner,
        primary=sim.particles[0],
    )
    sim.move_to_com()

    # Outer body around inner COM
    inner_com = sim.com(first=0, last=2)
    sim.add(
        m=ic.m2,
        a=ic.a_outer,
        e=ic.e_outer,
        inc=ic.inc_outer,
        Omega=ic.Omega_outer,
        omega=ic.omega_outer,
        M=ic.M_outer,
        primary=inner_com,
    )

    sim.move_to_com()
    return sim


# ============================================================
# DIAGNOSTICS
# ============================================================


def particle_arrays(sim: rebound.Simulation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return masses, positions, and velocities for the three particles.
    """
    p = sim.particles
    masses = np.array([p[0].m, p[1].m, p[2].m], dtype=float)
    pos = np.array([[p[k].x, p[k].y, p[k].z] for k in range(3)], dtype=float)
    vel = np.array([[p[k].vx, p[k].vy, p[k].vz] for k in range(3)], dtype=float)
    return masses, pos, vel


def inner_com_position(masses: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Mass-weighted center of mass of the inner binary.
    """
    return (masses[0] * pos[0] + masses[1] * pos[1]) / (masses[0] + masses[1])


def hierarchy_metric(masses: np.ndarray, pos: np.ndarray) -> float:
    """
    H(t) = distance(outer, inner-COM) / inner binary separation.
    """
    r_in = np.linalg.norm(pos[1] - pos[0])
    if r_in <= 1.0e-14:
        return 1.0e30

    com_in = inner_com_position(masses, pos)
    r_out = np.linalg.norm(pos[2] - com_in)
    return float(r_out / r_in)


def outer_distance_from_inner_com(masses: np.ndarray, pos: np.ndarray) -> float:
    """
    Distance from tertiary to inner-binary center of mass.
    """
    com_in = inner_com_position(masses, pos)
    return float(np.linalg.norm(pos[2] - com_in))


def min_pairwise_distance(pos: np.ndarray) -> float:
    """
    Minimum of the three pairwise separations.
    """
    d01 = np.linalg.norm(pos[0] - pos[1])
    d02 = np.linalg.norm(pos[0] - pos[2])
    d12 = np.linalg.norm(pos[1] - pos[2])
    return float(min(d01, d02, d12))


def pairwise_energies_and_angular_momenta(
    sim: rebound.Simulation,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pairwise diagnostic energies and angular momenta.

    E_ij are diagnostic two-body energies, not separately conserved
    Hamiltonians of the full three-body system.
    """
    G = sim.G
    masses, pos, vel = particle_arrays(sim)

    pairs = [(0, 1), (0, 2), (1, 2)]

    E_list: list[float] = []
    L_list: list[np.ndarray] = []

    for i, j in pairs:
        r_vec = pos[j] - pos[i]
        v_vec = vel[j] - vel[i]

        r = np.linalg.norm(r_vec)
        v2 = np.dot(v_vec, v_vec)

        mu = masses[i] * masses[j] / (masses[i] + masses[j])

        E = 0.5 * mu * v2 - G * masses[i] * masses[j] / r
        L = mu * np.cross(r_vec, v_vec)

        E_list.append(float(E))
        L_list.append(L)

    return np.array(E_list, dtype=float), np.array(L_list, dtype=float)


def normalized_omega_res(L: np.ndarray) -> float:
    """
    Dimensionless angular-momentum seam tension:

    Omega = (|L01-L02| + |L02-L12| + |L12-L01|)
            / (|L01| + |L02| + |L12|)
    """
    L01, L02, L12 = L[0], L[1], L[2]

    raw = (
        np.linalg.norm(L01 - L02)
        + np.linalg.norm(L02 - L12)
        + np.linalg.norm(L12 - L01)
    )

    denom = np.linalg.norm(L01) + np.linalg.norm(L02) + np.linalg.norm(L12) + 1e-30
    return float(raw / denom)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

FEATURE_NAMES = [
    "H_min",
    "H_max",
    "H_mean",
    "H_std",
    "H_median",
    "H_range",
    "H_final",
    "H_initial",
    "H_slope",
    "H_curvature",
    "H_below_frac",
    "H_crossings",
    "H_first_breach_frac",
    "H_last_breach_frac",
    "dH_min",
    "dH_max",
    "dH_mean",
    "dH_std",
    "abs_dH_mean",
    "abs_dH_max",
    "d2H_min",
    "d2H_max",
    "d2H_mean",
    "d2H_std",
    "abs_d2H_mean",
    "abs_d2H_max",
    "psi_mean",
    "psi_max",
    "psi_std",
    "psi_final",
    "E01_std",
    "E02_std",
    "E12_std",
    "E_exchange",
    "E_exchange_norm",
    "E01_slope",
    "E02_slope",
    "E12_slope",
    "E_slope_norm",
    "Omega_min",
    "Omega_max",
    "Omega_mean",
    "Omega_std",
    "Omega_slope",
    "Omega_final",
    "H_entropy",
    "Omega_entropy",
    "H_autocorr_lag5",
]


def safe_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=1.0e30, neginf=-1.0e30)


def safe_slope(y: Any, t: Any | None = None) -> float:
    y = safe_array(y)

    if y.size < 2:
        return 0.0

    if t is None:
        t = np.arange(y.size, dtype=float)
    else:
        t = safe_array(t)

    if np.allclose(t, t[0]):
        return 0.0

    try:
        return float(np.polyfit(t, y, 1)[0])
    except Exception:
        return 0.0


def safe_curvature(y: Any, t: Any | None = None) -> float:
    y = safe_array(y)

    if y.size < 3:
        return 0.0

    if t is None:
        t = np.arange(y.size, dtype=float)
    else:
        t = safe_array(t)

    try:
        coeff = np.polyfit(t, y, 2)
        return float(2.0 * coeff[0])
    except Exception:
        return 0.0


def derivatives(y: Any, t: Any) -> tuple[np.ndarray, np.ndarray]:
    y = safe_array(y)
    t = safe_array(t)

    if y.size < 2:
        return np.zeros_like(y), np.zeros_like(y)

    if y.size < 3 or np.any(np.diff(t) <= 0):
        t = np.arange(y.size, dtype=float)

    d1 = np.gradient(y, t, edge_order=1)

    if y.size < 3:
        d2 = np.zeros_like(y)
    else:
        d2 = np.gradient(d1, t, edge_order=1)

    return safe_array(d1), safe_array(d2)


def spectral_entropy(y: Any) -> float:
    y = safe_array(y)

    if y.size < 8 or np.allclose(y, y[0]):
        return 0.0

    yc = y - np.mean(y)
    psd = np.abs(np.fft.rfft(yc)) ** 2

    if psd.size <= 1:
        return 0.0

    psd = psd[1:]  # remove DC
    total = np.sum(psd)

    if total <= 0:
        return 0.0

    p = psd / total
    return float(-np.sum(p * np.log(p + 1.0e-30)) / np.log(len(p)))


def autocorr_lag(y: Any, lag: int = 5) -> float:
    y = safe_array(y)

    if y.size <= lag or np.allclose(y, y[0]):
        return 0.0

    yc = y - np.mean(y)
    denom = np.dot(yc, yc)

    if denom <= 0:
        return 0.0

    return float(np.dot(yc[:-lag], yc[lag:]) / denom)


def sigmoid_filter_abs_dH(
    dH: np.ndarray,
    H0: float,
    duration: float,
    kappa: float = 4.0,
) -> np.ndarray:
    """
    Tanh-sigmoid contraction filter.

    Psi = 0.5 * [1 + tanh(kappa * (|dH|/Hdot_c - 1))]
    """
    duration = max(float(duration), 1.0e-12)
    Hdot_c = max(abs(float(H0)) / duration, 1.0e-12)

    z = kappa * (np.abs(dH) / Hdot_c - 1.0)
    return 0.5 * (1.0 + np.tanh(z))


def extract_features(
    times: list[float] | np.ndarray,
    H_series: list[float] | np.ndarray,
    pair_E_series: list[list[float]] | np.ndarray,
    omega_series: list[float] | np.ndarray,
    Hcrit: float = 2.5,
    kappa: float = 4.0,
) -> np.ndarray:
    """
    Convert one integrated trajectory into a fixed-length feature vector.
    """
    t = safe_array(times)
    H = safe_array(H_series)
    pair_E = safe_array(pair_E_series)
    Omega = safe_array(omega_series)

    n = min(len(t), len(H), len(Omega), len(pair_E))

    if n == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=float)

    t = t[:n]
    H = H[:n]
    Omega = Omega[:n]
    pair_E = pair_E[:n]

    if pair_E.ndim != 2 or pair_E.shape[1] != 3:
        tmp = np.zeros((len(H), 3), dtype=float)
        if pair_E.ndim == 2 and len(pair_E) > 0:
            rows = min(tmp.shape[0], pair_E.shape[0])
            cols = min(3, pair_E.shape[1])
            tmp[:rows, :cols] = pair_E[:rows, :cols]
        pair_E = tmp

    dH, d2H = derivatives(H, t)

    duration = t[-1] - t[0] if len(t) > 1 else 1.0
    psi = sigmoid_filter_abs_dH(dH, H0=H[0], duration=duration, kappa=kappa)

    below = H < Hcrit

    if H.size > 1:
        H_crossings = int(np.sum((H[1:] < Hcrit) & (H[:-1] >= Hcrit)))
    else:
        H_crossings = int(below[0])

    if np.any(below):
        idx = np.where(below)[0]
        first_breach_frac = float(idx[0] / max(1, len(H) - 1))
        last_breach_frac = float(idx[-1] / max(1, len(H) - 1))
    else:
        first_breach_frac = 1.0
        last_breach_frac = 1.0

    E_std = np.std(pair_E, axis=0)
    E_exchange = float(np.sum(E_std))
    E_scale = float(np.mean(np.abs(pair_E)) + 1.0e-30)
    E_exchange_norm = E_exchange / E_scale

    E_slopes = np.array(
        [safe_slope(pair_E[:, k], t) for k in range(3)],
        dtype=float,
    )

    E_slope_norm = float(np.linalg.norm(E_slopes) / E_scale)

    values = [
        np.min(H),
        np.max(H),
        np.mean(H),
        np.std(H),
        np.median(H),
        np.max(H) - np.min(H),
        H[-1],
        H[0],
        safe_slope(H, t),
        safe_curvature(H, t),
        np.mean(below),
        H_crossings,
        first_breach_frac,
        last_breach_frac,
        np.min(dH),
        np.max(dH),
        np.mean(dH),
        np.std(dH),
        np.mean(np.abs(dH)),
        np.max(np.abs(dH)),
        np.min(d2H),
        np.max(d2H),
        np.mean(d2H),
        np.std(d2H),
        np.mean(np.abs(d2H)),
        np.max(np.abs(d2H)),
        np.mean(psi),
        np.max(psi),
        np.std(psi),
        psi[-1],
        E_std[0],
        E_std[1],
        E_std[2],
        E_exchange,
        E_exchange_norm,
        E_slopes[0],
        E_slopes[1],
        E_slopes[2],
        E_slope_norm,
        np.min(Omega),
        np.max(Omega),
        np.mean(Omega),
        np.std(Omega),
        safe_slope(Omega, t),
        Omega[-1],
        spectral_entropy(H),
        spectral_entropy(Omega),
        autocorr_lag(H, lag=5),
    ]

    return safe_array(values).astype(float)


# ============================================================
# SIMULATION PIPELINE
# ============================================================


def simulate_system(system_id: int, cfg: SimulationConfig) -> tuple[np.ndarray, int, dict[str, Any]]:
    """
    Simulate one hierarchical triple and extract its feature vector.

    Returns
    -------
    features:
        48-dimensional feature vector.
    label:
        1 if ejected, 0 otherwise.
    record:
        Metadata dictionary for reproducibility.
    """
    ic = sample_triple_ic(system_id, cfg)
    sim = make_sim(ic)

    E0 = float(sim.energy())

    Pout = outer_period_years(ic)
    t_max = cfg.t_max_outer_periods * Pout

    sample_times = np.linspace(0.0, t_max, cfg.n_samples)

    r_eject = cfg.ejection_radius_factor * ic.a_outer
    r_collide = cfg.collision_radius_factor * ic.a_inner

    times: list[float] = []
    H_series: list[float] = []
    pair_E_series: list[list[float]] = []
    omega_series: list[float] = []

    status = "stable"
    ejection_time = None
    collision_time = None

    max_energy_error = 0.0

    for t in sample_times:
        sim.integrate(float(t), exact_finish_time=0)

        E = float(sim.energy())
        rel_err = abs((E - E0) / E0) if E0 != 0 else abs(E - E0)
        max_energy_error = max(max_energy_error, rel_err)

        masses, pos, _vel = particle_arrays(sim)

        H_t = hierarchy_metric(masses, pos)
        pair_E, L = pairwise_energies_and_angular_momenta(sim)
        Omega_t = normalized_omega_res(L)

        times.append(float(sim.t))
        H_series.append(float(H_t))
        pair_E_series.append(pair_E.tolist())
        omega_series.append(float(Omega_t))

        if rel_err > cfg.max_relative_energy_error:
            status = "numerical_error"
            break

        if min_pairwise_distance(pos) < r_collide:
            status = "collision"
            collision_time = float(sim.t)
            break

        if outer_distance_from_inner_com(masses, pos) > r_eject:
            status = "ejected"
            ejection_time = float(sim.t)
            break

    E_final = float(sim.energy())
    energy_error_final = abs((E_final - E0) / E0) if E0 != 0 else abs(E_final - E0)

    features = extract_features(
        times,
        H_series,
        pair_E_series,
        omega_series,
        Hcrit=cfg.H_critical,
        kappa=cfg.sigmoid_kappa,
    )

    label = 1 if status == "ejected" else 0

    record = {
        "system_id": int(system_id),
        "status": status,
        "label": int(label),
        "ejection_time": ejection_time,
        "collision_time": collision_time,
        "energy_error_final": float(energy_error_final),
        "max_energy_error": float(max_energy_error),
        "H_min": float(np.min(H_series)) if len(H_series) else None,
        "H_final": float(H_series[-1]) if len(H_series) else None,
        "n_samples_actual": len(times),
        "initial_conditions": asdict(ic),
    }

    return features, label, record


def generate_dataset(cfg: SimulationConfig) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, int]]:
    """
    Generate a full dataset of hierarchical triples.

    Numerical-error systems are stored in records but excluded from the
    ML feature matrix.
    """
    X: list[np.ndarray] = []
    y: list[int] = []
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    feature_path = Path(cfg.feature_file)
    record_path = Path(cfg.records_file)
    checkpoint_path = Path(cfg.checkpoint_records_file)

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()

    for system_id in tqdm(range(cfg.n_systems), desc="IAS15 triples"):
        features, label, record = simulate_system(system_id, cfg)

        status = record["status"]
        counts[status] = counts.get(status, 0) + 1

        # Exclude numerical failures from ML training.
        # Collisions are treated as non-ejection unless modeled separately.
        if status != "numerical_error":
            X.append(features)
            y.append(label)

        records.append(record)

        if cfg.checkpoint_every and (system_id + 1) % cfg.checkpoint_every == 0:
            np.savez_compressed(
                feature_path,
                X=np.asarray(X, dtype=float),
                y=np.asarray(y, dtype=int),
                feature_names=np.asarray(FEATURE_NAMES),
            )

            with checkpoint_path.open("w") as f:
                json.dump(records, f, indent=2)

            elapsed = time.time() - start
            print(
                f"\nCheckpoint {system_id + 1}/{cfg.n_systems} | "
                f"elapsed={elapsed/60:.1f} min | counts={counts}"
            )

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int)

    np.savez_compressed(
        feature_path,
        X=X_arr,
        y=y_arr,
        feature_names=np.asarray(FEATURE_NAMES),
    )

    with record_path.open("w") as f:
        json.dump(records, f, indent=2)

    print("\nSaved:", feature_path)
    print("X shape:", X_arr.shape)
    print("Ejection fraction:", float(y_arr.mean()) if len(y_arr) else 0.0)
    print("Outcome counts:", counts)

    return X_arr, y_arr, records, counts


# ============================================================
# METRICS AND MODELS
# ============================================================


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_eject": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_eject": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_eject": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity_stable": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if y_score is not None and len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, y_score))

    return out


def make_classifier(random_state: int = 42):
    """
    Full soft-voting classifier used for the headline result.
    """
    logit = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=0.5,
                    random_state=random_state,
                ),
            ),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )

    if HAS_XGB:
        boost = XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            reg_alpha=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        boost = HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=random_state,
        )

    clf = VotingClassifier(
        estimators=[
            ("logit", logit),
            ("boost", boost),
            ("rf", rf),
        ],
        voting="soft",
        weights=[1, 3, 2],
        n_jobs=-1,
    )

    return clf


def make_fast_classifier(random_state: int = 42):
    """
    Faster classifier used for robustness and ablation experiments.
    """
    if HAS_XGB:
        return XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            reg_alpha=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

    return HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        random_state=random_state,
    )


def sweep_hmin_threshold(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    output_csv: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Sweep H_min threshold and save the full sweep as CSV.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    h_idx = feature_names.index("H_min")
    Hmin = X[:, h_idx]

    thresholds = np.linspace(1.5, 8.0, 261)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for th in thresholds:
        pred = (Hmin < th).astype(int)

        m = classification_metrics(y, pred)
        row = {"threshold": float(th), **m}
        rows.append(row)

        if best is None or row["f1_eject"] > best["f1_eject"]:
            best = row

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nBest H_min threshold on full dataset:")
    print(best)
    print("Saved threshold sweep:", output_csv)

    assert best is not None
    return best, rows


# Backwards-compatible alias matching original Colab name.
sweep_Hmin_threshold = sweep_hmin_threshold


def train_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    cfg: SimulationConfig,
) -> tuple[Any, dict[str, Any]]:
    """
    Train the full classifier and evaluate on a stratified validation split.
    """
    if len(np.unique(y)) < 2:
        raise RuntimeError("Only one class found. Increase n_systems or broaden sampling.")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.split_seed,
        stratify=y,
    )

    clf = make_classifier(random_state=42)
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    val_pred = clf.predict(X_val)

    train_prob = clf.predict_proba(X_train)[:, 1]
    val_prob = clf.predict_proba(X_val)[:, 1]

    train_metrics = classification_metrics(y_train, train_pred, train_prob)
    val_metrics = classification_metrics(y_val, val_pred, val_prob)

    # Baseline threshold fitted on training only.
    h_idx = FEATURE_NAMES.index("H_min")
    thresholds = np.linspace(1.5, 8.0, 261)

    best_train: dict[str, Any] | None = None

    for th in thresholds:
        pred_train = (X_train[:, h_idx] < th).astype(int)
        m = classification_metrics(y_train, pred_train)
        row = {"threshold": float(th), **m}

        if best_train is None or row["f1_eject"] > best_train["f1_eject"]:
            best_train = row

    assert best_train is not None
    best_th = best_train["threshold"]

    pred_val_baseline = (X_val[:, h_idx] < best_th).astype(int)
    baseline_val_metrics = classification_metrics(y_val, pred_val_baseline)

    metrics = {
        "n_total_used": int(len(y)),
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_val)),
        "eject_fraction_total": float(y.mean()),
        "train": train_metrics,
        "validation": val_metrics,
        "baseline_H_min": {
            "threshold_fit_on_train": float(best_th),
            "validation": baseline_val_metrics,
        },
    }

    model_path = Path(cfg.model_file)
    metrics_path = Path(cfg.metrics_file)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": clf,
            "feature_names": FEATURE_NAMES,
            "metrics": metrics,
            "config": asdict(cfg),
        },
        model_path,
    )

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\n================ FINAL METRICS ================")
    print(json.dumps(metrics, indent=2))
    print("================================================")
    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)

    return clf, metrics
