"""
src/pk_model/drug_library.py
============================

Literature-grounded Population Pharmacokinetic (PopPK) drug library for
Narrow Therapeutic Index (NTI) compounds: Warfarin, Vancomycin, and Digoxin.

Provides reference clearance, distribution volumes, allometric scaling exponents,
pharmacogenomic sensitivities, therapeutic target bands, and individualized
parameter computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from src.pk_model.compartmental_ode import PatientState


@dataclass(frozen=True)
class DrugProfile:
    name: str
    display_name: str
    cl_ref: float                  # L/h reference clearance for standard 70kg, eGFR 90 patient
    v1_ref: float                  # L central distribution volume
    v2_ref: float                  # L peripheral distribution volume
    q_ref: float                   # L/h intercompartmental clearance
    theta_renal: float             # Exponent for renal clearance scaling: (eGFR / 90.0) ** theta_renal
    theta_weight_cl: float = 0.75  # Allometric exponent for clearance
    theta_weight_v: float = 1.00   # Allometric exponent for volume
    dose_min_mg: float = 1.0
    dose_max_mg: float = 100.0
    dose_default_mg: float = 10.0
    dose_step_mg: float = 1.0
    standard_interval_h: int = 24
    allowed_intervals_h: List[int] = field(default_factory=lambda: [24])
    c_min_target: float = 1.0      # In display_unit
    c_max_target: float = 4.0      # In display_unit
    target_auc24: float = 40.0     # In display_unit * h
    display_unit: str = "mcg/mL"
    unit_multiplier_from_mcg: float = 1.0  # Multiply mcg/mL by this to get display_unit
    elimination_type: str = "hepatic"      # hepatic, renal, mixed
    fda_max_single_dose_mg: float = 20.0
    fda_max_daily_dose_mg: float = 20.0
    cyp2c9_sensitive: bool = False
    vkorc1_sensitive: bool = False
    loading_dose_eligible: bool = False


DRUG_LIBRARY: Dict[str, DrugProfile] = {
    "warfarin": DrugProfile(
        name="warfarin",
        display_name="Warfarin",
        cl_ref=0.16,                # L/h (~0.0023 L/h/kg)
        v1_ref=8.5,                 # L
        v2_ref=5.5,                 # L
        q_ref=0.45,                 # L/h
        theta_renal=0.08,           # Predominantly hepatic elimination (CYP2C9)
        theta_weight_cl=0.75,
        theta_weight_v=1.0,
        dose_min_mg=0.5,
        dose_max_mg=15.0,
        dose_default_mg=5.0,
        dose_step_mg=0.5,
        standard_interval_h=24,
        allowed_intervals_h=[24],
        c_min_target=1.0,           # mcg/mL (approximates therapeutic anticoagulation)
        c_max_target=3.0,           # mcg/mL
        target_auc24=45.0,          # mcg*h/mL
        display_unit="mcg/mL",
        unit_multiplier_from_mcg=1.0,
        elimination_type="hepatic",
        fda_max_single_dose_mg=15.0,
        fda_max_daily_dose_mg=15.0,
        cyp2c9_sensitive=True,
        vkorc1_sensitive=True,
        loading_dose_eligible=False,
    ),
    "vancomycin": DrugProfile(
        name="vancomycin",
        display_name="Vancomycin",
        cl_ref=4.20,                # L/h (strongly renal: ~0.06 L/h/kg in normal renal function)
        v1_ref=35.0,                # L (~0.5 L/kg)
        v2_ref=25.0,                # L (~0.35 L/kg)
        q_ref=7.50,                 # L/h
        theta_renal=0.85,           # Strong dependence on glomerular filtration rate
        theta_weight_cl=0.75,
        theta_weight_v=1.0,
        dose_min_mg=250.0,
        dose_max_mg=2000.0,
        dose_default_mg=1000.0,
        dose_step_mg=250.0,
        standard_interval_h=12,
        allowed_intervals_h=[8, 12, 24],
        c_min_target=10.0,          # mcg/mL trough
        c_max_target=20.0,          # mcg/mL peak/trough safety limit
        target_auc24=500.0,         # mg*h/L (target AUC24/MIC 400-600)
        display_unit="mcg/mL",
        unit_multiplier_from_mcg=1.0,
        elimination_type="renal",
        fda_max_single_dose_mg=2000.0,
        fda_max_daily_dose_mg=4000.0,
        cyp2c9_sensitive=False,
        vkorc1_sensitive=False,
        loading_dose_eligible=True,
    ),
    "digoxin": DrugProfile(
        name="digoxin",
        display_name="Digoxin",
        cl_ref=6.80,                # L/h (~113 mL/min: ~70% renal filtration + 30% metabolic)
        v1_ref=260.0,               # L (~3.7 L/kg extensive myocardial/tissue binding)
        v2_ref=420.0,               # L
        q_ref=22.0,                 # L/h
        theta_renal=0.68,           # Moderate-strong renal dependence
        theta_weight_cl=0.75,
        theta_weight_v=1.0,
        dose_min_mg=0.0625,
        dose_max_mg=0.50,
        dose_default_mg=0.125,
        dose_step_mg=0.0625,
        standard_interval_h=24,
        allowed_intervals_h=[24],
        c_min_target=0.5,           # ng/mL (narrow therapeutic index: 0.5 - 2.0 ng/mL)
        c_max_target=2.0,           # ng/mL
        target_auc24=25.0,          # ng*h/mL
        display_unit="ng/mL",
        unit_multiplier_from_mcg=1000.0,  # 1 mcg/mL = 1000 ng/mL
        elimination_type="mixed",
        fda_max_single_dose_mg=0.50,
        fda_max_daily_dose_mg=0.50,
        cyp2c9_sensitive=False,
        vkorc1_sensitive=False,
        loading_dose_eligible=True,
    ),
}


def get_drug_profile(drug_name: str) -> DrugProfile:
    """Retrieve drug profile by name (case-insensitive)."""
    key = str(drug_name).strip().lower()
    if key not in DRUG_LIBRARY:
        # Fallback to closest match or default warfarin
        for k in DRUG_LIBRARY:
            if k in key or key in k:
                return DRUG_LIBRARY[k]
        raise ValueError(
            f"Unknown drug '{drug_name}'. Supported drugs: {list(DRUG_LIBRARY.keys())}"
        )
    return DRUG_LIBRARY[key]


def compute_individualized_pk_parameters(
    drug_name: str,
    weight_kg: float,
    age_years: float,
    serum_creatinine: float,
    egfr_ml_min: Optional[float] = None,
    alt: Optional[float] = None,
    ast: Optional[float] = None,
    cyp2c9_status: Optional[str] = None,
    vkorc1_status: Optional[str] = None,
    eta_cl: float = 0.0,
    eta_v1: float = 0.0,
) -> PatientState:
    """Computes individualized pharmacokinetic parameters (CL, V1, V2, Q)
    grounded in PopPK scaling laws and pharmacogenomics.

    Parameters:
      eta_cl: Log-clearance random effect (from Bayesian TDM MAP estimation)
      eta_v1: Log-volume random effect (from Bayesian TDM MAP estimation)
    """
    profile = get_drug_profile(drug_name)

    # 1. Renal scaling factor
    effective_egfr = egfr_ml_min if egfr_ml_min is not None and egfr_ml_min > 0 else 70.0
    renal_ratio = np.clip(effective_egfr / 90.0, 0.1, 2.0)
    renal_factor = renal_ratio ** profile.theta_renal

    # 2. Allometric weight scaling
    ref_weight = 70.0
    weight_cl_factor = (weight_kg / ref_weight) ** profile.theta_weight_cl
    weight_v_factor = (weight_kg / ref_weight) ** profile.theta_weight_v

    # 3. Hepatic scaling factor (for hepatic-cleared drugs like warfarin)
    hepatic_factor = 1.0
    if profile.elimination_type in ("hepatic", "mixed") and alt is not None:
        if alt > 120.0:
            hepatic_factor = 0.70  # Hepatic impairment reduces clearance
        elif alt > 70.0:
            hepatic_factor = 0.85

    # 4. Pharmacogenomics modifier for Warfarin
    pgx_cl_factor = 1.0
    if profile.cyp2c9_sensitive and cyp2c9_status:
        cyp_map = {
            "*1/*1": 1.00,
            "*1/*2": 0.82,
            "*1/*3": 0.65,
            "*2/*2": 0.70,
            "*2/*3": 0.50,
            "*3/*3": 0.30,
        }
        pgx_cl_factor = cyp_map.get(str(cyp2c9_status).strip(), 1.0)

    # Calculate individualized CL and Volumes with Bayesian TDM multipliers
    cl_ind = (
        profile.cl_ref
        * renal_factor
        * weight_cl_factor
        * hepatic_factor
        * pgx_cl_factor
        * np.exp(eta_cl)
    )
    v1_ind = profile.v1_ref * weight_v_factor * np.exp(eta_v1)
    v2_ind = profile.v2_ref * weight_v_factor
    q_ind = profile.q_ref * weight_cl_factor

    # Bound for physical realism
    cl_ind = float(np.clip(cl_ind, 0.01, 30.0))
    v1_ind = float(np.clip(v1_ind, 1.0, 1000.0))
    v2_ind = float(np.clip(v2_ind, 1.0, 1500.0))
    q_ind = float(np.clip(q_ind, 0.05, 50.0))

    return PatientState(
        patient_id=f"TWIN-{int(weight_kg)}_{int(age_years)}",
        weight_kg=float(weight_kg),
        age_years=float(age_years),
        serum_creatinine=float(serum_creatinine),
        clearance=cl_ind,
        v_central=v1_ind,
        v_peripheral=v2_ind,
        intercompartmental_clearance=q_ind,
    )
