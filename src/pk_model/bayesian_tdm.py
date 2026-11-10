"""
src/pk_model/bayesian_tdm.py
============================

Bayesian Therapeutic Drug Monitoring (TDM) using Maximum A Posteriori (MAP)
estimation for digital twin individualization.

Given population pharmacokinetic (PopPK) priors and 1 or 2 observed patient
serum concentrations (t_obs, C_obs), this module optimizes individual random
effects (eta_cl, eta_v1) to forecast personalized concentration trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import minimize

from src.pk_model.drug_library import (
    DrugProfile,
    get_drug_profile,
    compute_individualized_pk_parameters,
)
from src.pk_model.compartmental_ode import DoseSchedule, DoseEvent


@dataclass
class ObservedConcentration:
    """A measured laboratory drug concentration at a specific time."""
    time_h: float
    concentration: float  # In the drug's display_unit (mcg/mL or ng/mL)


@dataclass
class BayesianTDMResult:
    """Output of MAP Bayesian parameter estimation."""
    eta_cl: float
    eta_v1: float
    cl_posterior: float
    v1_posterior: float
    cl_prior: float
    v1_prior: float
    rmse_observed: float
    log_posterior_density: float
    calibrated_patient_state: object


def _analytical_2cmt_concentration(
    t: float,
    dose_mg: float,
    cl: float,
    v1: float,
    v2: float,
    q: float,
) -> float:
    """Computes analytical central concentration C1(t) for an IV bolus
    in a 2-compartment linear model.
    Units: dose in mg, V in L -> mg/L = mcg/mL.
    """
    k10 = cl / v1
    k12 = q / v1
    k21 = q / v2

    b = k10 + k12 + k21
    c = k10 * k21
    disc = np.sqrt(max(0.0, b * b - 4.0 * c))
    alpha = (b + disc) / 2.0
    beta = (b - disc) / 2.0

    if abs(alpha - beta) < 1e-8:
        return float((dose_mg / v1) * np.exp(-alpha * t))

    a_coef = (alpha - k21) / (alpha - beta)
    b_coef = (k21 - beta) / (alpha - beta)

    conc_mcg_ml = (dose_mg / v1) * (a_coef * np.exp(-alpha * t) + b_coef * np.exp(-beta * t))
    return float(max(0.0, conc_mcg_ml))


def fit_bayesian_tdm(
    drug_name: str,
    weight_kg: float,
    age_years: float,
    serum_creatinine: float,
    egfr_ml_min: float,
    dose_mg: float,
    observations: List[ObservedConcentration],
    alt: Optional[float] = None,
    ast: Optional[float] = None,
    cyp2c9_status: Optional[str] = None,
    omega_cl: float = 0.35,  # PopPK inter-individual variability CV
    omega_v1: float = 0.25,
    sigma_prop: float = 0.15,  # Proportional residual error (15%)
    sigma_add: float = 0.10,   # Additive residual error
) -> BayesianTDMResult:
    """Performs Maximum A Posteriori (MAP) Bayesian forecasting.

    Objective:
      argmin_{eta} 0.5 * (eta_cl / omega_cl)^2 + 0.5 * (eta_v1 / omega_v1)^2
                 + sum_i 0.5 * ((C_obs,i - C_pred,i) / sigma_i)^2
    """
    profile = get_drug_profile(drug_name)

    # Base individualized parameters (prior)
    prior_state = compute_individualized_pk_parameters(
        drug_name=drug_name,
        weight_kg=weight_kg,
        age_years=age_years,
        serum_creatinine=serum_creatinine,
        egfr_ml_min=egfr_ml_min,
        alt=alt,
        ast=ast,
        cyp2c9_status=cyp2c9_status,
        eta_cl=0.0,
        eta_v1=0.0,
    )

    if not observations:
        return BayesianTDMResult(
            eta_cl=0.0,
            eta_v1=0.0,
            cl_posterior=prior_state.clearance,
            v1_posterior=prior_state.v_central,
            cl_prior=prior_state.clearance,
            v1_prior=prior_state.v_central,
            rmse_observed=0.0,
            log_posterior_density=0.0,
            calibrated_patient_state=prior_state,
        )

    mult = profile.unit_multiplier_from_mcg

    def objective(eta: np.ndarray) -> float:
        eta_c, eta_v = eta[0], eta[1]
        # Prior penalty (log-normal assumption)
        prior_term = 0.5 * (eta_c / omega_cl) ** 2 + 0.5 * (eta_v / omega_v1) ** 2

        # Candidate parameters
        cl_cand = prior_state.clearance * np.exp(eta_c)
        v1_cand = prior_state.v_central * np.exp(eta_v)
        v2_cand = prior_state.v_peripheral
        q_cand = prior_state.intercompartmental_clearance

        # Likelihood penalty over observations
        likelihood_term = 0.0
        for obs in observations:
            if hasattr(obs, "time_h") and hasattr(obs, "concentration"):
                t_obs = float(obs.time_h)
                c_obs = float(obs.concentration)
            elif isinstance(obs, dict):
                t_obs = float(obs.get("time_h", obs.get("time", 0.0)))
                c_obs = float(obs.get("concentration", obs.get("conc", 0.0)))
            elif isinstance(obs, (list, tuple)) and len(obs) >= 2:
                t_obs = float(obs[0])
                c_obs = float(obs[1])
            else:
                continue

            pred_mcg = _analytical_2cmt_concentration(
                t=t_obs,
                dose_mg=dose_mg,
                cl=cl_cand,
                v1=v1_cand,
                v2=v2_cand,
                q=q_cand,
            )
            pred_display = pred_mcg * mult
            sigma_i = max(1e-4, sigma_prop * pred_display + sigma_add)
            residual = (c_obs - pred_display) / sigma_i
            likelihood_term += 0.5 * (residual ** 2)

        return float(prior_term + likelihood_term)

    # Initial guess eta = [0, 0]
    res = minimize(
        objective,
        x0=np.array([0.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(-1.5, 1.5), (-1.2, 1.2)],  # Cap multiplier in [0.22, 4.48]
    )

    best_eta_cl = float(res.x[0]) if res.success else 0.0
    best_eta_v1 = float(res.x[1]) if res.success else 0.0

    post_state = compute_individualized_pk_parameters(
        drug_name=drug_name,
        weight_kg=weight_kg,
        age_years=age_years,
        serum_creatinine=serum_creatinine,
        egfr_ml_min=egfr_ml_min,
        alt=alt,
        ast=ast,
        cyp2c9_status=cyp2c9_status,
        eta_cl=best_eta_cl,
        eta_v1=best_eta_v1,
    )

    # Compute observed RMSE
    errors = []
    for obs in observations:
        if hasattr(obs, "time_h") and hasattr(obs, "concentration"):
            t_obs = float(obs.time_h)
            c_obs = float(obs.concentration)
        elif isinstance(obs, dict):
            t_obs = float(obs.get("time_h", obs.get("time", 0.0)))
            c_obs = float(obs.get("concentration", obs.get("conc", 0.0)))
        elif isinstance(obs, (list, tuple)) and len(obs) >= 2:
            t_obs = float(obs[0])
            c_obs = float(obs[1])
        else:
            continue
        pred_mcg = _analytical_2cmt_concentration(
            t=t_obs,
            dose_mg=dose_mg,
            cl=post_state.clearance,
            v1=post_state.v_central,
            v2=post_state.v_peripheral,
            q=post_state.intercompartmental_clearance,
        )
        errors.append((c_obs - (pred_mcg * mult)) ** 2)
    rmse = float(np.sqrt(np.mean(errors))) if errors else 0.0

    return BayesianTDMResult(
        eta_cl=best_eta_cl,
        eta_v1=best_eta_v1,
        cl_posterior=post_state.clearance,
        v1_posterior=post_state.v_central,
        cl_prior=prior_state.clearance,
        v1_prior=prior_state.v_central,
        rmse_observed=rmse,
        log_posterior_density=float(-res.fun),
        calibrated_patient_state=post_state,
    )
