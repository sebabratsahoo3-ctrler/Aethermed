"""
Phase 3 — Validate the PK/PD engine against published PK values for three
well-characterized drugs, and report RMSE. This is the step that turns
"physics-informed Neural ODE" from a claim into a validated component.

Literature values below are population averages from standard clinical
pharmacology references — cite the specific source in EXPERIMENTS.md when
you pin final numbers, since exact values vary by reference/population.
These are starting points, not verified-to-the-decimal.

What this actually validates
-----------------------------
`TwoCompartmentPK` / `simulate_two_compartment` is a genuine 2-compartment
model: k12/k21 (inter-compartmental distribution) are required to be
strictly positive, so a "1-compartment" model isn't directly expressible.
Literature only gives us `clearance` and `v_central` here (no k12/k21 per
drug yet — see the placeholder note below), so this script deliberately
sets the inter-compartmental clearance Q to a value that is negligible
relative to CL. That decouples the peripheral compartment for practical
purposes, so the central-compartment concentration is well approximated
by the closed-form mono-exponential solution C1(t) = (Dose/V1) * exp(-k10*t)
with k10 = CL/V1.

This lets us actually run the `torchdiffeq` solver end-to-end and check
two independent things:
  1. RMSE of the simulated concentration curve vs. the closed-form
     mono-exponential solution -> validates the ODE solver / integration
     pipeline itself (should be ~0 by construction, since Q is negligible).
  2. A terminal-phase half-life recovered by log-linear regression on the
     *simulated* trajectory (not recomputed algebraically from CL/V) vs.
     the literature half-life -> validates that k10 = CL/V, as run through
     the solver, reproduces the expected elimination kinetics.

k12/k21 magnitude and true patient covariates are still placeholders (see
comment below) — replace with literature/DrugBank values before treating
this as a completed Phase 3 validation.

Usage:
    python -m src.pk_model.train
"""
import sys

import numpy as np
import torch

from src.pk_model.compartmental_ode import (
    DoseEvent,
    DoseSchedule,
    PatientState,
    PKModelError,
    simulate_two_compartment,
)

# NOTE: these are illustrative population-average starting points, sourced
# from standard PK references (e.g. Winter's Basic Clinical Pharmacokinetics).
# Before citing these in a thesis/interview, verify each against the
# specific DrugBank/ChEMBL record pulled in Phase 1 and record the exact
# source + value in docs/EXPERIMENTS.md — do not present placeholders as
# validated ground truth.
KNOWN_DRUG_PK = {
    "warfarin": {
        "half_life_hours": 40.0,   # long half-life, highly protein-bound
        "v_central_L": 10.0,
        "clearance_L_per_hr": 0.193,
    },
    "digoxin": {
        "half_life_hours": 36.0,
        "v_central_L": 7.3,       # per kg averaged to a reference 70kg adult -> ~510L total; central compartment is smaller
        "clearance_L_per_hr": 7.0,
    },
    "vancomycin": {
        "half_life_hours": 6.0,
        "v_central_L": 18.0,
        "clearance_L_per_hr": 4.5,
    },
}

# Reference bolus dose (arbitrary mass units). The half-life recovery and
# RMSE-vs-analytic checks are dose-invariant for a linear model, so the
# specific value doesn't matter, only that dosing is nonzero.
DOSE_AMOUNT = 100.0

# Number of terminal-phase points (fraction of the simulated grid) used to
# fit the log-linear elimination-rate regression. Using only the tail
# avoids contamination from any residual (negligible-but-nonzero)
# distribution-phase curvature.
TERMINAL_FRACTION = 0.5

# Number of simulated grid points per drug.
N_POINTS = 300


def build_validation_patient(drug_name: str, known: dict) -> PatientState:
    """Builds a `PatientState` whose k10 = CL/V1 matches the literature
    values, with a *negligible* (but strictly positive, as required by
    `TwoCompartmentPK`) inter-compartmental clearance so the peripheral
    compartment doesn't materially affect the central-compartment curve.

    v_peripheral and intercompartmental_clearance are NOT literature
    values — they are placeholders chosen so Q/V1 and Q/V2 are ~0. Real
    k12/k21 need their own literature lookup per drug before this becomes
    a full validated 2-compartment model.
    """
    v_central = known["v_central_L"]
    negligible_q = 1e-6 * known["clearance_L_per_hr"]  # PLACEHOLDER — ~0 distribution

    return PatientState(
        patient_id=f"validation-{drug_name}",
        weight_kg=70.0,        # reference-adult placeholder covariate (unused by the linear ODE)
        age_years=50.0,        # placeholder covariate
        serum_creatinine=1.0,  # placeholder covariate
        clearance=known["clearance_L_per_hr"],
        v_central=v_central,
        v_peripheral=v_central,  # PLACEHOLDER — irrelevant given negligible Q
        intercompartmental_clearance=negligible_q,
    )


def rmse(predicted: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((predicted - target) ** 2)))


def recover_terminal_half_life(times: np.ndarray, concentrations: np.ndarray) -> float:
    """Fits a log-linear regression to the terminal portion of a
    concentration-time curve and returns the implied half-life.

    This is computed directly from the *simulated* trajectory (i.e. it
    exercises the solver), not re-derived algebraically from CL/V.
    """
    n_tail = max(int(len(times) * TERMINAL_FRACTION), 2)
    t_tail = times[-n_tail:]
    c_tail = concentrations[-n_tail:]

    valid = c_tail > 0
    if valid.sum() < 2:
        raise PKModelError(
            "Not enough positive concentration points in the terminal phase "
            "to fit an elimination-rate regression."
        )

    slope, _intercept = np.polyfit(t_tail[valid], np.log(c_tail[valid]), 1)
    k_recovered = -slope
    if k_recovered <= 0:
        raise PKModelError(
            f"Recovered elimination rate constant is non-positive ({k_recovered!r}); "
            "terminal-phase fit failed."
        )
    return float(np.log(2.0) / k_recovered)


def validate_drug(drug_name: str, known: dict) -> dict:
    """Simulates the 2-compartment ODE (with negligible distribution) via
    `simulate_two_compartment`, then checks:
      - RMSE of the simulated central-compartment concentration vs. the
        closed-form mono-exponential solution (solver correctness check).
      - The terminal half-life recovered from the simulated curve vs. the
        literature half-life (elimination-kinetics check).
    """
    patient = build_validation_patient(drug_name, known)
    dose_schedule = DoseSchedule(boluses=[DoseEvent(time=0.0, amount=DOSE_AMOUNT)])

    horizon_hours = 10.0 * known["half_life_hours"]
    time_steps = np.linspace(0.0, horizon_hours, N_POINTS).tolist()

    amounts, concentrations = simulate_two_compartment(patient, dose_schedule, time_steps)
    c1_simulated = concentrations[:, 0]

    k10 = patient.clearance / patient.v_central
    t = torch.tensor(time_steps, dtype=torch.float32)
    c1_analytic = (DOSE_AMOUNT / patient.v_central) * torch.exp(-k10 * t)

    solver_rmse = rmse(c1_simulated, c1_analytic)

    recovered_half_life = recover_terminal_half_life(
        np.asarray(time_steps), c1_simulated.detach().cpu().numpy()
    )
    error = abs(recovered_half_life - known["half_life_hours"])

    return {
        "drug": drug_name,
        "literature_half_life_hours": known["half_life_hours"],
        "recovered_half_life_hours": recovered_half_life,
        "absolute_error_hours": error,
        "solver_rmse_vs_analytic": solver_rmse,
    }


def run_validation_suite() -> list[dict]:
    results = []
    for drug_name, known in KNOWN_DRUG_PK.items():
        try:
            result = validate_drug(drug_name, known)
        except PKModelError as exc:
            print(f"{drug_name}: validation FAILED ({exc})")
            continue
        results.append(result)
        print(
            f"{drug_name}: literature t1/2={result['literature_half_life_hours']}h, "
            f"recovered={result['recovered_half_life_hours']:.2f}h, "
            f"error={result['absolute_error_hours']:.4f}h, "
            f"solver RMSE vs. analytic={result['solver_rmse_vs_analytic']:.6g}"
        )
    return results


def main() -> int:
    print("Validating compartmental PK ODE against known drug half-lives...\n")
    print(
        "NOTE: inter-compartmental rate constants (k12/k21) are set to a "
        "negligible placeholder in this scaffold, not literature values — "
        "see comments in this file. This validates the k10 = CL/V identity "
        "as executed through the actual torchdiffeq solver (via RMSE against "
        "the closed-form mono-exponential solution) and the resulting "
        "elimination half-life, not the full multi-exponential PK profile "
        "shape a real (non-negligible) 2-compartment fit would have. Replace "
        "placeholders with literature values before treating this as a "
        "completed Phase 3 validation.\n"
    )
    results = run_validation_suite()

    if not results:
        print("\nNo drugs validated successfully.")
        return 1

    errors = [r["absolute_error_hours"] for r in results]
    rmses = [r["solver_rmse_vs_analytic"] for r in results]
    print(f"\nMean absolute half-life error across {len(results)} drugs: {np.mean(errors):.4f} hours")
    print(f"Mean solver RMSE vs. analytic across {len(results)} drugs: {np.mean(rmses):.6g}")

    return 0


if __name__ == "__main__":
    sys.exit(main())