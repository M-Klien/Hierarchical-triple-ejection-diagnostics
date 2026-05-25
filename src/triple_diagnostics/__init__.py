"""
Triple-system ejection diagnostics.

This package contains simulation, feature extraction, classification,
plotting, and robustness utilities for hierarchical triple ejection
classification.
"""

from .core import (
    FEATURE_NAMES,
    SimulationConfig,
    TripleIC,
    generate_dataset,
    train_evaluate,
    sweep_hmin_threshold,
)

__all__ = [
    "FEATURE_NAMES",
    "SimulationConfig",
    "TripleIC",
    "generate_dataset",
    "train_evaluate",
    "sweep_hmin_threshold",
]
