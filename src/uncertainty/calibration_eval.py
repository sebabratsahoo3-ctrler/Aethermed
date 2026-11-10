"""
Phase 7 — Calibration validation: does a claimed 90% conformal interval
actually cover ~90% of held-out outcomes? This is the mandatory check per
project rules — a conformal method that isn't empirically validated on
held-out data is just an unverified claim wearing a rigorous-sounding name.

Usage pattern:
    coverage = empirical_coverage(mapie_regressor, X_holdout, y_holdout, alpha=0.1)
    # coverage['empirical_coverage_rate'] should be close to 0.90
"""
from __future__ import annotations

import numpy as np

from src.uncertainty.conformal import predict_with_interval


def empirical_coverage(mapie_regressor, X_holdout: np.ndarray, y_holdout: np.ndarray, alpha: float = 0.1) -> dict:
    _, (lower, upper) = predict_with_interval(mapie_regressor, X_holdout, alpha=alpha)
    covered = (y_holdout >= lower) & (y_holdout <= upper)
    empirical_rate = float(np.mean(covered))
    nominal_rate = 1 - alpha

    # Rough binomial standard error for the coverage estimate — gives a
    # sense of whether a deviation from nominal is meaningful given n.
    n = len(y_holdout)
    se = float(np.sqrt(nominal_rate * (1 - nominal_rate) / n)) if n > 0 else float("nan")

    return {
        "nominal_coverage": nominal_rate,
        "empirical_coverage_rate": empirical_rate,
        "n_holdout": n,
        "standard_error": se,
        "within_1_se_of_nominal": abs(empirical_rate - nominal_rate) <= se,
        "mean_interval_width": float(np.mean(upper - lower)),
    }


def calibration_report(mapie_regressor, X_holdout: np.ndarray, y_holdout: np.ndarray, alphas: list[float] = None) -> list[dict]:
    """Check calibration across several confidence levels (e.g. 80/90/95%)
    rather than just one — a method can look calibrated at one alpha and
    not another, which is worth catching before trusting it broadly."""
    alphas = alphas or [0.2, 0.1, 0.05]
    return [empirical_coverage(mapie_regressor, X_holdout, y_holdout, alpha=a) for a in alphas]
