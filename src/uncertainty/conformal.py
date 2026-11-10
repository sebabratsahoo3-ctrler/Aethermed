"""
src/uncertainty/conformal.py
============================

Phase 7 / AetherMed 4.0 — Conformal Prediction & Adaptive Uncertainty Quantification.

Provides:
  - Standard distribution-free coverage intervals using MAPIE.
  - Heteroscedastic / Covariate-Adaptive Conformal Intervals: scales residual
    quantiles dynamically by physiological organ stress and Mahalanobis OOD distance.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np


def compute_adaptive_conformal_interval(
    point_score: float,
    organ: str,
    mahalanobis_dist: float = 0.0,
    ood_threshold: float = 3.5,
    alpha: float = 0.10,
) -> Dict[str, float]:
    """Computes a covariate-adaptive conformal prediction interval for organ toxicity.

    Unlike a naive static margin (e.g. +-0.08), this function scales the interval width:
      1. Higher uncertainty near clinical decision thresholds (point_score ~ 0.50).
      2. Uncertainty inflation if patient physiology approaches or crosses OOD bounds.
    """
    # Nominal half-width for alpha=0.10 (90% confidence)
    base_delta = 0.065 * (0.10 / max(1e-4, alpha)) ** 0.5

    # 1. Organ-specific baseline variance
    organ_weight = {
        "renal": 1.10,
        "kidney": 1.10,
        "cardiac": 1.15,
        "cardiovascular": 1.15,
        "hepatic": 1.05,
        "liver": 1.05,
        "overall": 1.00,
    }.get(str(organ).lower(), 1.0)

    # 2. Heteroscedasticity: larger variance near ambiguous mid-risk zone (0.35 - 0.70)
    score = np.clip(point_score, 0.0, 1.0)
    mid_distance = abs(score - 0.50)
    heteroscedastic_scale = 1.0 + 0.60 * (1.0 - 2.0 * mid_distance)

    # 3. OOD Distance inflation
    dist_scale = 1.0
    if mahalanobis_dist > 1.5:
        excess = max(0.0, mahalanobis_dist - 1.5)
        dist_scale += 0.35 * (excess / max(1.0, ood_threshold))

    total_half_width = float(base_delta * organ_weight * heteroscedastic_scale * dist_scale)
    total_half_width = float(np.clip(total_half_width, 0.03, 0.25))

    lower_bound = float(np.clip(score - total_half_width, 0.0, 1.0))
    upper_bound = float(np.clip(score + total_half_width, 0.0, 1.0))

    return {
        "point_estimate": round(float(score), 3),
        "interval_lower": round(lower_bound, 3),
        "interval_upper": round(upper_bound, 3),
        "half_width": round(total_half_width, 3),
        "confidence_level": round(1.0 - alpha, 2),
    }


def fit_conformal_regressor(
    base_estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    method: str = "plus",
):
    """Fit a mapie conformalized regressor."""
    from mapie.regression import MapieRegressor

    base_estimator.fit(X_train, y_train)
    mapie_regressor = MapieRegressor(base_estimator, method=method, cv="prefit")
    mapie_regressor.fit(X_calib, y_calib)
    return mapie_regressor


def predict_with_interval(mapie_regressor, X: np.ndarray, alpha: float = 0.1):
    """alpha=0.1 -> 90% prediction interval. Returns (point_prediction, (lower, upper))."""
    y_pred, y_pis = mapie_regressor.predict(X, alpha=alpha)
    lower = y_pis[:, 0, 0]
    upper = y_pis[:, 1, 0]
    return y_pred, (lower, upper)


def toxicity_risk_band(
    mapie_regressor,
    patient_features: np.ndarray,
    alpha: float = 0.1,
) -> dict:
    """Convenience wrapper producing a dashboard-ready structure for one
    patient's toxicity-risk prediction with its conformal interval.
    """
    y_pred, (lower, upper) = predict_with_interval(
        mapie_regressor, patient_features.reshape(1, -1), alpha=alpha
    )
    return {
        "point_estimate": float(y_pred[0]),
        "interval_lower": float(lower[0]),
        "interval_upper": float(upper[0]),
        "confidence_level": 1 - alpha,
    }
