"""
compartmental_ode.py
=====================

AetherMed :: PK/PD Digital Twin

A standard linear 2-compartment pharmacokinetic (PK) model expressed as a
`torchdiffeq`-compatible ODE module, plus supporting utilities for dose
scheduling (IV bolus and/or infusion) and patient covariate handling.

Model
-----
Amounts in the central (A1) and peripheral (A2) compartments evolve as:

    dA1/dt = -(k10 + k12) * A1 + k21 * A2 + R(t)
    dA2/dt =  k12 * A1 - k21 * A2

where:
    k10 = CL / V1   (elimination rate constant, central compartment)
    k12 = Q  / V1   (distribution rate constant, central -> peripheral)
    k21 = Q  / V2   (distribution rate constant, peripheral -> central)
    R(t) is the exogenous infusion rate (amount / time) at time t.

Concentrations are recovered as C1 = A1 / V1, C2 = A2 / V2.

Bolus doses are not part of the continuous ODE (they are Dirac impulses);
`simulate_two_compartment` integrates piecewise between dose times and
adds bolus mass to A1 at each event boundary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

try:
    from torchdiffeq import odeint
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise ImportError(
        "The 'torchdiffeq' package is required for compartmental_ode. "
        "Install it with `pip install torchdiffeq`."
    ) from exc


LOGGER = logging.getLogger(__name__)

_STATE_DIM: int = 2  # [A1 (central amount), A2 (peripheral amount)]


class PKModelError(Exception):
    """Raised for invalid PK parameters, states, or dosing schedules."""


# ----------------------------------------------------------------------
# Patient / dosing data structures
# ----------------------------------------------------------------------


@dataclass
class PatientState:
    """Physiological covariates and baseline PK parameters for one patient.

    PK parameters (`clearance`, `v_central`, `v_peripheral`,
    `intercompartmental_clearance`) may be population-typical values or
    values already individualized (e.g. via allometric scaling on weight
    and renal function upstream of this module).

    Attributes
    ----------
    patient_id:
        Identifier for logging/traceability.
    weight_kg, age_years, serum_creatinine:
        Raw covariates, retained for use by covariate models / the neural
        correction module even though they don't enter the linear ODE
        directly.
    clearance:
        Total body clearance, CL (volume/time), e.g. L/h.
    v_central:
        Central compartment volume, V1 (volume), e.g. L.
    v_peripheral:
        Peripheral compartment volume, V2 (volume), e.g. L.
    intercompartmental_clearance:
        Distributional clearance, Q (volume/time), e.g. L/h.
    """

    patient_id: str
    weight_kg: float
    age_years: float
    serum_creatinine: float
    clearance: float
    v_central: float
    v_peripheral: float
    intercompartmental_clearance: float

    def validate(self) -> None:
        for name in ("weight_kg", "age_years", "clearance", "v_central",
                      "v_peripheral", "intercompartmental_clearance"):
            value = getattr(self, name)
            if value is None or value <= 0:
                raise PKModelError(
                    f"PatientState.{name} must be positive, got {value!r} "
                    f"for patient_id={self.patient_id!r}."
                )
        if self.serum_creatinine is not None and self.serum_creatinine <= 0:
            raise PKModelError(
                f"serum_creatinine must be positive, got {self.serum_creatinine!r} "
                f"for patient_id={self.patient_id!r}."
            )

    def covariate_tensor(self) -> Tensor:
        """Returns covariates as a tensor, useful as auxiliary input to a
        neural correction network."""
        return torch.tensor(
            [self.weight_kg, self.age_years, self.serum_creatinine],
            dtype=torch.float32,
        )

    def rate_constants(self) -> Tuple[float, float, float]:
        """Returns (k10, k12, k21) derived from clearance/volume terms."""
        k10 = self.clearance / self.v_central
        k12 = self.intercompartmental_clearance / self.v_central
        k21 = self.intercompartmental_clearance / self.v_peripheral
        return k10, k12, k21


@dataclass
class DoseEvent:
    """A single IV bolus dose.

    Attributes
    ----------
    time:
        Time of administration (same time units as the simulation clock).
    amount:
        Bolus mass added instantaneously to the central compartment (A1).
    """

    time: float
    amount: float

    def validate(self) -> None:
        if self.time < 0:
            raise PKModelError(f"DoseEvent.time must be >= 0, got {self.time!r}")
        if self.amount <= 0:
            raise PKModelError(f"DoseEvent.amount must be positive, got {self.amount!r}")


@dataclass
class InfusionSegment:
    """A constant-rate IV infusion active over [start, end).

    Attributes
    ----------
    start, end:
        Infusion window (simulation time units).
    rate:
        Infusion rate (amount / time), added continuously to dA1/dt while
        the segment is active.
    """

    start: float
    end: float
    rate: float

    def validate(self) -> None:
        if self.end <= self.start:
            raise PKModelError(
                f"InfusionSegment.end ({self.end}) must be > start ({self.start})"
            )
        if self.rate < 0:
            raise PKModelError(f"InfusionSegment.rate must be >= 0, got {self.rate!r}")


@dataclass
class DoseSchedule:
    """A patient's full dosing regimen: any mix of boluses and infusions.

    Attributes
    ----------
    boluses:
        List of `DoseEvent` instances.
    infusions:
        List of `InfusionSegment` instances.
    """

    boluses: List[DoseEvent] = field(default_factory=list)
    infusions: List[InfusionSegment] = field(default_factory=list)

    def validate(self) -> None:
        if not self.boluses and not self.infusions:
            raise PKModelError("DoseSchedule must contain at least one bolus or infusion.")
        for bolus in self.boluses:
            bolus.validate()
        for infusion in self.infusions:
            infusion.validate()

    def event_times(self) -> List[float]:
        """Returns sorted, de-duplicated time points at which the ODE's
        right-hand side changes discontinuously (bolus times and infusion
        start/end boundaries) — used to integrate piecewise."""
        times = {0.0}
        for bolus in self.boluses:
            times.add(float(bolus.time))
        for infusion in self.infusions:
            times.add(float(infusion.start))
            times.add(float(infusion.end))
        return sorted(times)

    def infusion_rate_at(self, t: float) -> float:
        """Total instantaneous infusion rate active at time `t`."""
        rate = 0.0
        for infusion in self.infusions:
            if infusion.start <= t < infusion.end:
                rate += infusion.rate
        return rate


# ----------------------------------------------------------------------
# ODE module
# ----------------------------------------------------------------------


class TwoCompartmentPK(nn.Module):
    """Linear 2-compartment PK dynamics, compatible with `torchdiffeq.odeint`.

    The module's `forward(t, state)` signature matches what `odeint`
    expects: `state` is a tensor `[A1, A2]` (amounts), and the return
    value is `d(state)/dt`.

    Infusion input is supplied as a callable `rate_fn(t) -> Tensor` so the
    same module can be reused across dosing segments without re-instantiation.
    """

    def __init__(
        self,
        k10: float,
        k12: float,
        k21: float,
        v_central: float,
        v_peripheral: float,
        infusion_rate_fn: Optional[callable] = None,
    ) -> None:
        super().__init__()
        for name, value in (("k10", k10), ("k12", k12), ("k21", k21),
                             ("v_central", v_central), ("v_peripheral", v_peripheral)):
            if value <= 0:
                raise PKModelError(f"{name} must be positive, got {value!r}")

        # Registered as buffers (not parameters) since these are fixed
        # patient-specific physiology, not being learned here. The
        # neural correction module (see neural_correction.py) is what
        # gets trained.
        self.register_buffer("k10", torch.tensor(float(k10)))
        self.register_buffer("k12", torch.tensor(float(k12)))
        self.register_buffer("k21", torch.tensor(float(k21)))
        self.register_buffer("v_central", torch.tensor(float(v_central)))
        self.register_buffer("v_peripheral", torch.tensor(float(v_peripheral)))

        self._infusion_rate_fn = infusion_rate_fn or (lambda t: torch.tensor(0.0))

    @classmethod
    def from_patient_state(
        cls,
        patient: PatientState,
        infusion_rate_fn: Optional[callable] = None,
    ) -> "TwoCompartmentPK":
        """Builds a `TwoCompartmentPK` from a `PatientState`'s derived
        rate constants."""
        patient.validate()
        k10, k12, k21 = patient.rate_constants()
        return cls(
            k10=k10,
            k12=k12,
            k21=k21,
            v_central=patient.v_central,
            v_peripheral=patient.v_peripheral,
            infusion_rate_fn=infusion_rate_fn,
        )

    def forward(self, t: Tensor, state: Tensor) -> Tensor:
        if state.shape[-1] != _STATE_DIM:
            raise PKModelError(
                f"Expected state of dimension {_STATE_DIM} ([A1, A2]), got shape {tuple(state.shape)}"
            )
        a1, a2 = state[..., 0], state[..., 1]
        infusion = self._infusion_rate_fn(t)

        da1_dt = -(self.k10 + self.k12) * a1 + self.k21 * a2 + infusion
        da2_dt = self.k12 * a1 - self.k21 * a2

        return torch.stack([da1_dt, da2_dt], dim=-1)

    def concentrations(self, amounts: Tensor) -> Tensor:
        """Converts a trajectory of amounts [..., 2] -> concentrations [..., 2]."""
        v = torch.stack([self.v_central, self.v_peripheral])
        return amounts / v


# ----------------------------------------------------------------------
# Simulation driver
# ----------------------------------------------------------------------


def _make_infusion_rate_fn(schedule: DoseSchedule) -> callable:
    def rate_fn(t: Tensor) -> Tensor:
        # torchdiffeq may call this with a 0-dim tensor at arbitrary times
        # (including intermediate RK stages), so we evaluate in Python
        # float space and re-wrap.
        t_val = float(t.detach().cpu()) if isinstance(t, Tensor) else float(t)
        return torch.tensor(schedule.infusion_rate_at(t_val), dtype=torch.float32)

    return rate_fn


def simulate_two_compartment(
    patient: PatientState,
    dose_schedule: DoseSchedule,
    time_steps: Sequence[float],
    solver: str = "dopri5",
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """Integrates the linear 2-compartment PK model over `time_steps`,
    applying bolus doses and infusions from `dose_schedule`.

    Because bolus doses are impulses, integration proceeds piecewise:
    the timeline is split at every dose/infusion boundary, each segment
    is integrated with `torchdiffeq.odeint`, and any bolus at a segment
    boundary is added to A1 before continuing.

    Parameters
    ----------
    patient:
        Patient covariates and PK parameters.
    dose_schedule:
        Bolus and/or infusion dosing regimen.
    time_steps:
        Strictly increasing sequence of times (including t=0) at which
        the caller wants the solution reported.
    solver:
        `torchdiffeq` solver name (e.g. "dopri5", "rk4", "euler").
    rtol, atol:
        Solver tolerances (ignored by fixed-step solvers like "euler"/"rk4").

    Returns
    -------
    (amounts, concentrations):
        Two tensors of shape (len(time_steps), 2): amounts in [A1, A2]
        and corresponding concentrations [C1, C2].

    Raises
    ------
    PKModelError
        For invalid inputs (non-monotonic time steps, empty schedule, etc.).
    """
    patient.validate()
    dose_schedule.validate()

    time_list = [float(t) for t in time_steps]
    if len(time_list) < 2:
        raise PKModelError("time_steps must contain at least 2 points.")
    if any(t2 <= t1 for t1, t2 in zip(time_list, time_list[1:])):
        raise PKModelError("time_steps must be strictly increasing.")
    if time_list[0] != 0.0:
        LOGGER.warning(
            "time_steps does not start at 0.0 (starts at %.4f); simulation "
            "clock assumes t=0 is the start of dosing.",
            time_list[0],
        )

    infusion_rate_fn = _make_infusion_rate_fn(dose_schedule)
    ode_module = TwoCompartmentPK.from_patient_state(patient, infusion_rate_fn=infusion_rate_fn)

    # Build the full set of query points plus internal event boundaries,
    # then integrate piecewise so bolus impulses land exactly on a grid point.
    event_times = dose_schedule.event_times()
    all_times = sorted(set(time_list) | set(event_times))

    state = torch.zeros(_STATE_DIM, dtype=torch.float32)
    bolus_by_time = {}
    for bolus in dose_schedule.boluses:
        bolus_by_time[float(bolus.time)] = bolus_by_time.get(float(bolus.time), 0.0) + bolus.amount

    collected_states: List[Tensor] = []
    requested_set = set(time_list)
    current_time = all_times[0]

    if current_time in bolus_by_time:
        state = state + torch.tensor([bolus_by_time[current_time], 0.0], dtype=torch.float32)
    if current_time in requested_set:
        collected_states.append((current_time, state.clone()))

    try:
        for next_time in all_times[1:]:
            segment_t = torch.tensor([current_time, next_time], dtype=torch.float32)
            traj = odeint(ode_module, state, segment_t, rtol=rtol, atol=atol, method=solver)
            state = traj[-1]

            if next_time in bolus_by_time:
                state = state + torch.tensor([bolus_by_time[next_time], 0.0], dtype=torch.float32)

            if next_time in requested_set:
                collected_states.append((next_time, state.clone()))

            current_time = next_time
    except Exception as exc:
        raise PKModelError(f"ODE integration failed near t={current_time}: {exc}") from exc

    collected_states.sort(key=lambda pair: pair[0])
    amounts = torch.stack([s for _, s in collected_states], dim=0)
    concentrations = ode_module.concentrations(amounts)

    return amounts, concentrations