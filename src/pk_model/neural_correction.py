"""
neural_correction.py
=====================

AetherMed :: PK/PD Digital Twin

A physics-informed Neural ODE correction layer on top of the linear
2-compartment PK model in `compartmental_ode.py`.

Design
------
`NeuralPKCorrection` learns a patient-specific residual on top of the
population/individualized linear PK dynamics. The residual is
constrained by construction to respect mass balance: the network
outputs a single scalar *inter-compartmental exchange flux* correction
`f(t, A1, A2, covariates)` that is *added* to A2's derivative and
*subtracted* from A1's derivative (or vice versa). Because the same
quantity is added to one compartment and subtracted from the other,
the correction can never create or destroy total drug mass
(`A1 + A2` is invariant under the correction term alone) — it can only
reshape *how* mass moves between compartments relative to the linear
model's assumptions.

This is deliberately more restrictive than an unconstrained residual
network (which could add/remove mass and produce physically impossible
trajectories); the tradeoff is intentional for a clinical digital twin,
where mass-conservation is a hard physiological constraint we don't
want a data-driven correction to violate.

`HybridPKODE` composes the base `TwoCompartmentPK` dynamics with this
correction into a single `torchdiffeq`-compatible module.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

try:
    from torchdiffeq import odeint
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise ImportError(
        "The 'torchdiffeq' package is required for neural_correction. "
        "Install it with `pip install torchdiffeq`."
    ) from exc

from src.pk_model.compartmental_ode import (
    DoseSchedule,
    PatientState,
    PKModelError,
    TwoCompartmentPK,
    _STATE_DIM,
    _make_infusion_rate_fn,
)

LOGGER = logging.getLogger(__name__)


class NeuralPKCorrection(nn.Module):
    """Learns a mass-conserving inter-compartmental flux correction.

    The network maps (time, state, patient covariates) to a single
    scalar flux `f`. The correction applied to the derivative is:

        d_correction/dt = [-f, +f]

    i.e. it can only move mass between the central and peripheral
    compartments faster or slower than the linear model predicts; it
    cannot inject or remove total drug mass. An optional `flux_scale`
    bounds the magnitude of the correction (via `tanh`) so it acts as a
    bounded perturbation around the physics-based model rather than a
    free-form dynamics learner.

    Parameters
    ----------
    covariate_dim:
        Number of patient covariates fed into the network alongside
        (t, A1, A2). Set to 0 to use only time and state.
    hidden_dim:
        Width of the hidden layers.
    flux_scale:
        Maximum magnitude (mass/time) of the learned correction flux.
    """

    def __init__(
        self,
        covariate_dim: int = 3,
        hidden_dim: int = 32,
        flux_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if covariate_dim < 0:
            raise PKModelError(f"covariate_dim must be >= 0, got {covariate_dim!r}")
        if hidden_dim <= 0:
            raise PKModelError(f"hidden_dim must be positive, got {hidden_dim!r}")
        if flux_scale <= 0:
            raise PKModelError(f"flux_scale must be positive, got {flux_scale!r}")

        input_dim = 1 + _STATE_DIM + covariate_dim  # t, A1, A2, covariates...
        self.covariate_dim = covariate_dim
        self.flux_scale = flux_scale

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self._covariates: Optional[Tensor] = None

    def set_covariates(self, covariates: Optional[Tensor]) -> None:
        """Sets the (fixed, per-simulation) patient covariate tensor used
        as auxiliary network input. Call before integrating."""
        if covariates is not None:
            if covariates.numel() != self.covariate_dim:
                raise PKModelError(
                    f"Expected {self.covariate_dim} covariates, got tensor with "
                    f"{covariates.numel()} elements."
                )
            covariates = covariates.to(dtype=torch.float32)
        self._covariates = covariates

    def flux(self, t: Tensor, state: Tensor) -> Tensor:
        """Computes the scalar exchange-flux correction f(t, state, cov)."""
        if state.shape[-1] != _STATE_DIM:
            raise PKModelError(
                f"Expected state of dimension {_STATE_DIM}, got shape {tuple(state.shape)}"
            )

        t_expanded = t.reshape(1).to(dtype=torch.float32) if t.dim() == 0 else t.to(dtype=torch.float32)
        features = [t_expanded, state.to(dtype=torch.float32)]

        if self.covariate_dim > 0:
            if self._covariates is None:
                raise PKModelError(
                    "NeuralPKCorrection was configured with covariate_dim="
                    f"{self.covariate_dim} but no covariates were set. "
                    "Call set_covariates() before integrating."
                )
            features.append(self._covariates)

        x = torch.cat(features, dim=-1)
        raw_flux = self.net(x).squeeze(-1)
        # Bound the correction so it acts as a perturbation, not a
        # replacement for the physics-based dynamics.
        return self.flux_scale * torch.tanh(raw_flux)

    def forward(self, t: Tensor, state: Tensor) -> Tensor:
        """Returns the mass-conserving correction to d(state)/dt."""
        f = self.flux(t, state)
        return torch.stack([-f, f], dim=-1)


class HybridPKODE(nn.Module):
    """Combines the linear physics ODE with the learned correction into a
    single `torchdiffeq`-compatible module.

        d(state)/dt = base_dynamics(t, state) + correction(t, state)

    Mass conservation of the correction term means any mass-balance
    violation in the hybrid trajectory can only come from the base
    (physically-derived) dynamics or from the exogenous infusion term —
    not from the neural component.
    """

    def __init__(self, base_dynamics: TwoCompartmentPK, correction: NeuralPKCorrection) -> None:
        super().__init__()
        self.base_dynamics = base_dynamics
        self.correction = correction

    def forward(self, t: Tensor, state: Tensor) -> Tensor:
        return self.base_dynamics(t, state) + self.correction(t, state)

    def concentrations(self, amounts: Tensor) -> Tensor:
        return self.base_dynamics.concentrations(amounts)

    def mass_balance_residual(self, amounts_traj: Tensor, infused_mass: Tensor) -> Tensor:
        """Diagnostic: total mass (A1 + A2) minus cumulative infused mass
        minus cumulative eliminated mass, per timepoint. In a purely
        physics-based model without correction this should track the
        elimination term exactly; large deviations indicate the neural
        correction (or numerical error) is doing more than
        redistributing mass between compartments.

        This is a lightweight sanity check, not a full audit — true
        elimination accounting would integrate k10 * A1 over time.
        """
        total_mass = amounts_traj.sum(dim=-1)
        return total_mass - total_mass[0] - infused_mass


def build_hybrid_model(
    patient: PatientState,
    dose_schedule: DoseSchedule,
    correction: Optional[NeuralPKCorrection] = None,
) -> HybridPKODE:
    """Convenience constructor wiring a patient's base PK dynamics to a
    (new or provided) `NeuralPKCorrection` module, with covariates set.
    """
    patient.validate()
    infusion_rate_fn = _make_infusion_rate_fn(dose_schedule)
    base_dynamics = TwoCompartmentPK.from_patient_state(patient, infusion_rate_fn=infusion_rate_fn)

    if correction is None:
        correction = NeuralPKCorrection(covariate_dim=3)
    correction.set_covariates(patient.covariate_tensor())

    return HybridPKODE(base_dynamics=base_dynamics, correction=correction)


def simulate_drug_concentration(
    patient_state: PatientState,
    dose_schedule: DoseSchedule,
    time_steps: Sequence[float],
    correction: Optional[NeuralPKCorrection] = None,
    solver: str = "dopri5",
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> Tensor:
    """Top-level API: simulates the (optionally neural-corrected) drug
    concentration trajectory for a patient under a given dosing regimen.

    Parameters
    ----------
    patient_state:
        Patient covariates and PK parameters.
    dose_schedule:
        Bolus and/or infusion dosing regimen.
    time_steps:
        Strictly increasing sequence of times (starting at 0) at which to
        report concentrations.
    correction:
        An optional pre-trained (or freshly initialized) `NeuralPKCorrection`.
        If omitted, a freshly initialized (untrained) correction module is
        used, which is equivalent to a small random perturbation around
        the pure physics model — pass a trained module for meaningful
        patient-specific correction.
    solver, rtol, atol:
        Passed through to `torchdiffeq.odeint`.

    Returns
    -------
    Tensor of shape (len(time_steps), 2): concentration trajectories
    `[C_central, C_peripheral]` at each requested time step.

    Raises
    ------
    PKModelError
        For invalid patient state, dosing schedule, or time steps, or if
        integration fails.
    """
    patient_state.validate()
    dose_schedule.validate()

    time_list = [float(t) for t in time_steps]
    if len(time_list) < 2:
        raise PKModelError("time_steps must contain at least 2 points.")
    if any(t2 <= t1 for t1, t2 in zip(time_list, time_list[1:])):
        raise PKModelError("time_steps must be strictly increasing.")

    hybrid_model = build_hybrid_model(patient_state, dose_schedule, correction=correction)

    event_times = dose_schedule.event_times()
    all_times = sorted(set(time_list) | set(event_times))

    bolus_by_time = {}
    for bolus in dose_schedule.boluses:
        bolus_by_time[float(bolus.time)] = bolus_by_time.get(float(bolus.time), 0.0) + bolus.amount

    state = torch.zeros(_STATE_DIM, dtype=torch.float32)
    requested_set = set(time_list)
    current_time = all_times[0]

    if current_time in bolus_by_time:
        state = state + torch.tensor([bolus_by_time[current_time], 0.0], dtype=torch.float32)

    collected: list[Tuple[float, Tensor]] = []
    if current_time in requested_set:
        collected.append((current_time, state.clone()))

    try:
        for next_time in all_times[1:]:
            segment_t = torch.tensor([current_time, next_time], dtype=torch.float32)
            traj = odeint(hybrid_model, state, segment_t, rtol=rtol, atol=atol, method=solver)
            state = traj[-1]

            if next_time in bolus_by_time:
                state = state + torch.tensor([bolus_by_time[next_time], 0.0], dtype=torch.float32)

            if next_time in requested_set:
                collected.append((next_time, state.clone()))

            current_time = next_time
    except Exception as exc:
        raise PKModelError(
            f"Hybrid ODE integration failed near t={current_time}: {exc}"
        ) from exc

    collected.sort(key=lambda pair: pair[0])
    amounts = torch.stack([s for _, s in collected], dim=0)
    concentrations = hybrid_model.concentrations(amounts)

    LOGGER.info(
        "Simulated %d time points for patient_id=%s under %d bolus(es) / %d infusion(s).",
        len(time_list),
        patient_state.patient_id,
        len(dose_schedule.boluses),
        len(dose_schedule.infusions),
    )

    return concentrations