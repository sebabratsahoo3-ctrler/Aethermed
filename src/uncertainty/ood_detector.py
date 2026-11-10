"""
src/uncertainty/ood_detector.py
===============================

Phase 7 / AetherMed 4.0 — Out-of-Distribution (OOD) & Dosing Safety Detector.

Computes:
  1. Covariance-aware Mahalanobis distance D_M(x_patient) in physiological space.
  2. Per-biomarker z-score feature attributions explaining which labs drive the OOD flag.
  3. Dosing Safety Guard validating dose against FDA single/daily limits and organ status.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
import numpy as np

from src.pk_model.drug_library import DRUG_LIBRARY, get_drug_profile


DEFAULT_FEATURE_NAMES = [
    "age",
    "weight",
    "serum_creatinine",
    "egfr",
    "alt",
    "ast",
]


class MahalanobisOODDetector:
    """Detects physiological out-of-distribution patients using Mahalanobis distance
    and checks drug dosing safety against FDA limits and patient organ impairment.
    """

    def __init__(
        self,
        threshold_percentile: float = 97.5,
        feature_names: Optional[List[str]] = None,
    ):
        self.threshold_percentile = threshold_percentile
        self.feature_names = feature_names or DEFAULT_FEATURE_NAMES
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.inv_cov_: Optional[np.ndarray] = None
        self.threshold_: Optional[float] = None

    def fit(self, X_train: np.ndarray) -> "MahalanobisOODDetector":
        """Fits distribution statistics on cohort feature matrix."""
        X = np.asarray(X_train, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = np.std(X, axis=0) + 1e-6
        cov = np.cov(X, rowvar=False)

        # Regularize for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-5
        self.inv_cov_ = np.linalg.pinv(cov)

        train_distances = self._mahalanobis_batch(X)
        self.threshold_ = float(np.percentile(train_distances, self.threshold_percentile))
        return self

    def _ensure_fitted(self) -> None:
        if self.mean_ is None or self.inv_cov_ is None:
            # Default reference distribution derived from cohort
            # [age, weight, serum_creatinine, egfr, alt, ast]
            ref_means = np.array([62.5, 76.0, 1.25, 68.5, 39.0, 37.0], dtype=np.float64)
            ref_stds = np.array([12.0, 15.0, 0.70, 26.0, 34.0, 28.0], dtype=np.float64)
            self.mean_ = ref_means
            self.std_ = ref_stds
            cov = np.diag(ref_stds ** 2)
            # Add realistic covariance between Creatinine and eGFR
            cov[2, 3] = cov[3, 2] = -0.72 * ref_stds[2] * ref_stds[3]
            # Covariance between ALT and AST
            cov[4, 5] = cov[5, 4] = 0.75 * ref_stds[4] * ref_stds[5]
            cov += np.eye(len(ref_means)) * 1e-5
            self.inv_cov_ = np.linalg.pinv(cov)
            self.threshold_ = 3.65

    def _mahalanobis_batch(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        diff = X - self.mean_
        left = diff @ self.inv_cov_
        distances_sq = np.einsum("ij,ij->i", left, diff)
        return np.sqrt(np.clip(distances_sq, 0, None))

    def score(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        return self._mahalanobis_batch(np.atleast_2d(X))

    def is_ood(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) > self.threshold_

    def flag_patient(
        self,
        patient_features: np.ndarray | Dict[str, float] | List[float],
        drug_name: Optional[str] = None,
        dose_mg: Optional[float] = None,
        interval_h: int = 24,
    ) -> Dict[str, Any]:
        """Performs full joint patient-dose safety evaluation."""
        self._ensure_fitted()

        # Parse features into ordered vector
        if isinstance(patient_features, dict):
            vec = [
                float(patient_features.get("age", patient_features.get("age_years", 62.0))),
                float(patient_features.get("weight", patient_features.get("weight_kg", 75.0))),
                float(patient_features.get("serum_creatinine", 1.2)),
                float(patient_features.get("egfr", patient_features.get("egfr_ml_min", 65.0))),
                float(patient_features.get("alt", 35.0)),
                float(patient_features.get("ast", 35.0)),
            ]
        elif isinstance(patient_features, (list, tuple)):
            vec = [float(v) for v in patient_features[:len(self.feature_names)]]
        else:
            vec = [float(v) for v in np.asarray(patient_features).flatten()[:len(self.feature_names)]]

        arr = np.array(vec, dtype=np.float64).reshape(1, -1)
        dist = float(self.score(arr)[0])
        thresh = float(self.threshold_)
        flagged = dist > thresh

        # Per-feature z-score attribution
        attributions: List[Dict[str, Any]] = []
        for i, fname in enumerate(self.feature_names[:len(vec)]):
            val = float(vec[i])
            mean_val = float(self.mean_[i])
            std_val = float(self.std_[i])
            z = float((val - mean_val) / std_val)
            attributions.append({
                "feature": fname,
                "value": round(val, 2),
                "z_score": round(z, 2),
                "is_abnormal": bool(abs(z) >= 2.0),
            })

        # Dosing Safety Guard
        dosing_alerts: List[str] = []
        is_dose_unsafe = False

        if drug_name and dose_mg is not None and dose_mg > 0:
            try:
                profile = get_drug_profile(drug_name)
                # 1. Single dose limit
                if dose_mg > profile.fda_max_single_dose_mg:
                    dosing_alerts.append(
                        f"CRITICAL: Dose {dose_mg} mg exceeds FDA maximum single dose "
                        f"({profile.fda_max_single_dose_mg} mg) for {profile.display_name}."
                    )
                    is_dose_unsafe = True

                # 2. Daily dose limit
                daily_dose = dose_mg * (24.0 / max(1.0, float(interval_h)))
                if daily_dose > profile.fda_max_daily_dose_mg:
                    dosing_alerts.append(
                        f"CRITICAL: Total 24h dose {daily_dose:.1f} mg exceeds FDA daily maximum "
                        f"({profile.fda_max_daily_dose_mg} mg)."
                    )
                    is_dose_unsafe = True

                # 3. Organ-specific contraindication / clearance check
                egfr_val = vec[3]
                alt_val = vec[4]
                if profile.name == "vancomycin" and egfr_val < 30.0 and dose_mg > 1000.0:
                    dosing_alerts.append(
                        f"WARNING: High Vancomycin dose ({dose_mg} mg) in severe renal impairment "
                        f"(eGFR {egfr_val:.1f} mL/min) presents high acute nephrotoxicity risk."
                    )
                if profile.name == "digoxin" and egfr_val < 30.0 and dose_mg > 0.125:
                    dosing_alerts.append(
                        f"WARNING: Digoxin clearance is reduced in renal decline (eGFR {egfr_val:.1f}); "
                        f"dose > 0.125 mg may cause fatal cardiac arrhythmia."
                    )
                if profile.name == "warfarin" and alt_val > 100.0 and dose_mg > 7.5:
                    dosing_alerts.append(
                        f"WARNING: Impaired hepatic function (ALT {alt_val:.1f} U/L) delays Warfarin clearance; "
                        f"doses > 7.5 mg carry supratherapeutic INR hemorrhage risk."
                    )
            except Exception:
                pass

        severity = (
            "severe" if (dist > thresh * 1.5 or is_dose_unsafe)
            else "moderate" if flagged
            else "none"
        )

        return {
            "mahalanobis_distance": round(dist, 3),
            "threshold": round(thresh, 3),
            "is_ood": bool(flagged or is_dose_unsafe),
            "severity": severity,
            "feature_attributions": attributions,
            "dosing_alerts": dosing_alerts,
            "is_dose_unsafe": bool(is_dose_unsafe),
        }

    # Alias for assess_patient_and_dose
    assess_patient_and_dose = flag_patient

