"""
src/rl_policy/actions.py
========================

Action normalization and denormalization utilities for continuous dosing
policies. Supports both generic bounding [low, high] and drug-aware therapeutic
dosing ranges from `src.pk_model.drug_library`.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

DOSE_MIN_MG: float = 0.0
DOSE_MAX_MG: float = 200.0
DOSE_CENTER_MG: float = (DOSE_MIN_MG + DOSE_MAX_MG) / 2.0
DOSE_SCALE_MG: float = (DOSE_MAX_MG - DOSE_MIN_MG) / 2.0

DRUG_DOSE_RANGES: dict[str, tuple[float, float]] = {
    "warfarin": (0.5, 15.0),
    "vancomycin": (250.0, 2000.0),
    "digoxin": (0.0625, 0.50),
    "generic": (0.0, 200.0),
}


def _get_bounds(low: float = 0.0, high: float = 200.0, drug: Optional[str] = None) -> tuple[float, float]:
    if drug is not None:
        key = str(drug).strip().lower()
        if key in DRUG_DOSE_RANGES:
            return DRUG_DOSE_RANGES[key]
        for k, bounds in DRUG_DOSE_RANGES.items():
            if k in key or key in k:
                return bounds
    return float(low), float(high)


def normalize_action(
    action_mg: float | np.ndarray,
    low: float = 0.0,
    high: float = 200.0,
    drug: Optional[str] = None,
) -> float | np.ndarray:
    """Map raw dose(s) in mg to normalized continuous action(s) in [-1, 1]."""
    d_min, d_max = _get_bounds(low, high, drug)
    center = (d_min + d_max) / 2.0
    scale = (d_max - d_min) / 2.0
    if scale <= 1e-8:
        scale = 1.0

    is_scalar = np.isscalar(action_mg)
    arr = np.asarray(action_mg, dtype=np.float32)
    a_norm = np.clip((arr - center) / scale, -1.0, 1.0)

    if is_scalar:
        return float(a_norm)
    return a_norm


def denormalize_action(
    action_norm: float | np.ndarray,
    low: float = 0.0,
    high: float = 200.0,
    drug: Optional[str] = None,
) -> float | np.ndarray:
    """Map normalized continuous action(s) in [-1, 1] back to raw dose(s) in mg."""
    d_min, d_max = _get_bounds(low, high, drug)
    center = (d_min + d_max) / 2.0
    scale = (d_max - d_min) / 2.0

    is_scalar = np.isscalar(action_norm)
    arr = np.asarray(action_norm, dtype=np.float32)
    dose = np.clip(arr * scale + center, d_min, d_max)

    if is_scalar:
        return float(dose)
    return dose
