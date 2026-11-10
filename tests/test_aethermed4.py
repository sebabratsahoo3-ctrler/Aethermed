"""
tests/test_aethermed4.py
========================

Comprehensive test suite for AetherMed 4.0:
  1. 2021 CKD-EPI Creatinine Equation physiological coupling.
  2. PopPK reference parameters & multi-dose PK mass conservation.
  3. Bayesian TDM (MAP) parameter personalization.
  4. Drug-aware continuous CQL policy & toxicity dampening.
  5. Covariate-adaptive conformal intervals & Mahalanobis OOD guards.
  6. End-to-end FastAPI endpoints (/api/health, /api/cohort, /api/simulate, /api/trial/simulate).
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.data_pipeline.synthetic_generator import compute_ckd_epi_2021
from src.pk_model.drug_library import (
    DRUG_LIBRARY,
    get_drug_profile,
    compute_individualized_pk_parameters,
)
from src.pk_model.simulation_engine import SimulationEngine
from src.pk_model.bayesian_tdm import fit_bayesian_tdm
from src.rl_policy.train_cql import CQLAgent
from src.rl_policy.actions import DRUG_DOSE_RANGES, denormalize_action, normalize_action
from src.uncertainty.conformal import compute_adaptive_conformal_interval
from src.uncertainty.ood_detector import MahalanobisOODDetector


# --------------------------------------------------------------------------- #
# 1. 2021 CKD-EPI Creatinine Equation Tests
# --------------------------------------------------------------------------- #
def test_ckd_epi_2021_deterministic():
    # Female, 65yo, Cr 0.8 -> Normal renal function (~85-95 mL/min)
    gfr_f_normal = compute_ckd_epi_2021(serum_creatinine=0.8, age=65.0, gender="female", add_noise=False)
    assert 80.0 <= gfr_f_normal <= 105.0

    # Male, 70yo, Cr 2.5 -> Severe renal impairment (< 35 mL/min)
    gfr_m_impaired = compute_ckd_epi_2021(serum_creatinine=2.5, age=70.0, gender="male", add_noise=False)
    assert 15.0 <= gfr_m_impaired <= 35.0

    # Extreme bounds check: clipped into [5.0, 140.0]
    gfr_extreme = compute_ckd_epi_2021(serum_creatinine=12.0, age=90.0, gender="male", add_noise=False)
    assert gfr_extreme >= 5.0


# --------------------------------------------------------------------------- #
# 2. PopPK Drug Library & Reference Profiles
# --------------------------------------------------------------------------- #
def test_drug_library_profiles():
    for drug_name in ["Warfarin", "Vancomycin", "Digoxin"]:
        profile = get_drug_profile(drug_name)
        assert profile.cl_ref > 0.0
        assert profile.v1_ref > 0.0
        assert profile.dose_min_mg < profile.dose_max_mg
        assert profile.c_min_target < profile.c_max_target
        assert profile.fda_max_single_dose_mg >= profile.dose_max_mg


# --------------------------------------------------------------------------- #
# 3. Multi-Dose PK Simulation & Mass Conservation
# --------------------------------------------------------------------------- #
def test_simulation_engine_multi_dose_and_mass_conservation():
    pk_patient = compute_individualized_pk_parameters(
        drug_name="Vancomycin",
        weight_kg=75.0,
        age_years=60.0,
        serum_creatinine=1.1,
        egfr_ml_min=75.0,
    )
    result = SimulationEngine.simulate_regimen(
        drug_name="Vancomycin",
        patient_state=pk_patient,
        dose_mg=1000.0,
        interval_h=12,
        total_duration_h=48,
    )

    assert result.cmax > 0.0
    assert result.cmin_trough > 0.0
    assert result.auc_24h > 0.0
    assert 0.0 <= result.time_in_therapeutic_range_pct <= 100.0
    assert len(result.time_hours) == 1000
    assert result.mass_conservation_error < 1e-4


# --------------------------------------------------------------------------- #
# 4. Bayesian TDM (MAP) Optimization
# --------------------------------------------------------------------------- #
def test_bayesian_tdm_map_estimation():
    # Observed level is higher than expected -> MAP should adjust eta_cl downward
    tdm_res = fit_bayesian_tdm(
        drug_name="Warfarin",
        weight_kg=70.0,
        age_years=65.0,
        serum_creatinine=1.2,
        egfr_ml_min=65.0,
        dose_mg=5.0,
        observations=[(24.0, 1.8)],  # (t_obs, c_obs)
    )
    assert tdm_res.cl_posterior > 0.0
    assert tdm_res.v1_posterior > 0.0
    assert -2.0 <= tdm_res.eta_cl <= 2.0


# --------------------------------------------------------------------------- #
# 5. Drug-Aware Offline RL CQL Dosing & Actions
# --------------------------------------------------------------------------- #
def test_drug_aware_cql_dosing():
    # Verify action normalization & denormalization for each drug
    for drug, (d_min, d_max) in DRUG_DOSE_RANGES.items():
        if drug == "generic":
            continue
        mid_dose = (d_min + d_max) / 2.0
        a_norm = normalize_action(mid_dose, drug=drug)
        assert abs(a_norm) < 0.05
        recovered = denormalize_action(a_norm, drug=drug)
        assert abs(recovered - mid_dose) < 0.1

    # Load agent and check recommendation stays within clinical range
    agent = CQLAgent.load("models/cql_policy.pt", device="cpu")
    obs = np.array([65.0, 72.0, 1.2, 65.0, 5.0], dtype=np.float32)

    warfarin_rec = agent.get_dose_recommendation(obs, drug="Warfarin")
    assert 0.5 <= warfarin_rec <= 15.0

    vanco_rec = agent.get_dose_recommendation(obs, drug="Vancomycin")
    assert 250.0 <= vanco_rec <= 2000.0

    # Critical toxicity hold clamping
    toxic_rec = agent.get_dose_recommendation(obs, drug="Warfarin", organ_risk=0.88)
    assert toxic_rec < 3.0  # Clamped to bottom safe range


# --------------------------------------------------------------------------- #
# 6. Adaptive Conformal Intervals & OOD Detector
# --------------------------------------------------------------------------- #
def test_adaptive_conformal_and_ood():
    # Baseline in-bounds interval
    in_bounds = compute_adaptive_conformal_interval(0.40, organ="renal", mahalanobis_dist=0.8)
    # Borderline / OOD interval
    ood_interval = compute_adaptive_conformal_interval(0.40, organ="renal", mahalanobis_dist=4.5)

    assert ood_interval["half_width"] > in_bounds["half_width"]
    assert 0.0 <= in_bounds["interval_lower"] <= in_bounds["interval_upper"] <= 1.0

    detector = MahalanobisOODDetector()
    detector._ensure_fitted()

    # Normal patient check
    norm_eval = detector.assess_patient_and_dose(
        {"age": 62, "weight": 75, "serum_creatinine": 1.1, "egfr": 75, "alt": 30, "ast": 28},
        drug_name="Warfarin",
        dose_mg=5.0,
    )
    assert norm_eval["is_ood"] is False
    assert norm_eval["is_dose_unsafe"] is False

    # Outrageous dose check (100mg Warfarin)
    overdose_eval = detector.assess_patient_and_dose(
        {"age": 62, "weight": 75, "serum_creatinine": 1.1, "egfr": 75, "alt": 30, "ast": 28},
        drug_name="Warfarin",
        dose_mg=100.0,
    )
    assert overdose_eval["is_dose_unsafe"] is True
    assert len(overdose_eval["dosing_alerts"]) > 0


# --------------------------------------------------------------------------- #
# 7. End-to-End FastAPI Backend Tests
# --------------------------------------------------------------------------- #
def test_fastapi_endpoints():
    with TestClient(app) as client:
        # /api/health
        h_res = client.get("/api/health")
        assert h_res.status_code == 200
        assert h_res.json()["status"] == "healthy"

        # /api/cohort
        c_res = client.get("/api/cohort?limit=10")
        assert c_res.status_code == 200
        c_data = c_res.json()
        assert c_data["total_twins"] >= 100
        assert len(c_data["sample_patients"]) <= 10

        # /api/simulate
        sim_res = client.post("/api/simulate", json={
            "patient_id": "TWIN-TEST-4",
            "age": 67.0,
            "gender": "female",
            "weight": 68.0,
            "serum_creatinine": 1.3,
            "drug": "Warfarin",
            "dose_mg": 5.0,
            "interval_h": 24,
            "duration_h": 72,
        })
        assert sim_res.status_code == 200
        s_data = sim_res.json()
        assert s_data["pk_results"]["cmax"] > 0.0
        assert len(s_data["organ_risks"]) == 3
        assert s_data["rl_policy"]["cql_recommended_dose_mg"] > 0.0
        assert len(s_data["rag_report"]["structured_citations"]) > 0

        # /api/trial/simulate
        trial_res = client.post("/api/trial/simulate", json={
            "drug": "Vancomycin",
            "standard_dose_mg": 1000.0,
            "interval_h": 12,
            "min_age": 55.0,
            "max_age": 85.0,
            "sample_size": 15,
        })
        assert trial_res.status_code == 200
        t_data = trial_res.json()
        assert t_data["sample_size"] > 0
        assert "standard_mean_ttr_pct" in t_data
        assert "cql_mean_ttr_pct" in t_data
