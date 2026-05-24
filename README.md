# Hierarchical Triple Ejection Diagnostics

Reproducible code for the manuscript:

**Time-dependent topology-inspired diagnostics for ejection classification in hierarchical triple systems**

This repository generates hierarchical triple systems, integrates them with REBOUND IAS15, extracts time-dependent diagnostic features, trains baseline and full classifiers, performs robustness tests, and reproduces the main validation figures.

## Summary

The fiducial experiment generates 10,000 hierarchical triples and integrates each system for 100 outer orbital periods or until termination.

Main reported result:

- 10,000 generated systems
- 9,999 usable systems after one numerical-error rejection
- 6,999 / 3,000 stratified train-validation split
- Full diagnostic model validation accuracy: 99.43%
- AUC-ROC: 0.99974
- Ejection recall: 99.74%
- Stable specificity: 99.33%
- H_min threshold baseline repeated-split accuracy: 85.30 ± 0.60%
- Initial-condition-only baselines: approximately 94.6–94.8%
- Repeated-split full-feature accuracy: 99.09 ± 0.18%

The method is a finite-time outcome-classification framework. It does not solve the three-body problem and is not, by itself, a causal early-warning model.

## Repository structure

```text
hierarchical-triple-ejection-diagnostics/
│
├── src/
│   └── triple_diagnostics/
│       ├── __init__.py
│       ├── core.py
│       ├── plotting.py
│       └── robustness.py
│
├── scripts/
│   ├── run_10k.py
│   ├── make_dashboard.py
│   └── run_robustness.py
│
├── results/
│   ├── metrics_10k.json
│   ├── summary_10k.json
│   ├── robustness_results.json
│   ├── ablation_results.csv
│   └── Hcrit_sweep_10k.csv
│
├── figures/
│   ├── Accuracy_10k.png
│   ├── TOPOLOGICAL_FRAMEWORK_VALIDATION_500kyr.png
│   ├── Plot_1_Chaos_vs_Determinism.png
│   ├── Plot_2_Energy_Transfer.png
│   ├── phase_space_decomposition_marginal.png
│   ├── kam_tori_breakdown.png
│   └── forced_ejection_Hierarchical_Massive_Primary.png
│
└── manuscript/
    └── main.tex
