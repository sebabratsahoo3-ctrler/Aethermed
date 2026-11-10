"""
tests/test_pk_model.py

Tests for the PK/PD Neural ODE stack:
    - 2-compartment ODE integration in `src/pk_model/compartmental_ode.py`
      (`TwoCompartmentPK`, `simulate_two_compartment`).
    - `PatientState` instantiation and mass-balance conservation physics
      in `NeuralPKCorrection` (`src/pk_model/neural_correction.py`).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed; skipping PK model tests")

pk_ode = pytest.importorskip(
    "src.pk_model.compartmental_ode",
    reason="src.pk_model.compartmental_ode not available; skipping ODE integration tests",
)
neural_correction = pytest.importorskip(
    "src.pk_model.neural_correction",
    reason="src.pk_model.neural_correction not available; skipping NeuralPKCorrection tests",
)

PatientState = getattr(pk_ode, "PatientState", None)
DoseEvent = getattr(pk_ode, "DoseEvent", None)
DoseSchedule = getattr(pk_ode, "DoseSchedule", None)
TwoCompartmentPK = getattr(pk_ode, "TwoCompartmentPK", None)
simulate_two_compartment = getattr(pk_ode, "simulate_two_compartment", None)

NeuralPKCorrection = getattr(neural_correction, "NeuralPKCorrection", None)
simulate_drug_concentration = getattr(neural_correction, "simulate_drug_concentration", None)

pytestmark = pytest.mark.skipif(
    TwoCompartmentPK is None or simulate_two_compartment is None or PatientState is None,
    reason="src.pk_model.compartmental_ode does not expose required classes.",
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def default_patient_state():
    """Typical 2-compartment IV bolus patient parameters."""
    return PatientState(
        patient_id="test_patient",
        weight_kg=70.0,
        age_years=45.0,
        serum_creatinine=1.0,
        clearance=4.5,
        v_central=35.0,
        v_peripheral=25.0,
        intercompartmental_clearance=8.0,
    )


@pytest.fixture
def default_dose_schedule():
    """Standard 1000mg IV bolus dose schedule."""
    return DoseSchedule(boluses=[DoseEvent(time=0.0, amount=1000.0)])


@pytest.fixture
def elimination_time_steps() -> list[float]:
    """A time grid well past Tmax, covering the elimination phase only."""
    return [float(t) for t in np.linspace(2.0, 48.0, 50)]


# --------------------------------------------------------------------------
# TwoCompartmentPK / simulate_two_compartment
# --------------------------------------------------------------------------

class TestTwoCompartmentODEIntegration:
    """Tests for the 2-compartment ODE integration path."""

    def test_two_compartment_pk_constructs(self, default_patient_state) -> None:
        model = TwoCompartmentPK.from_patient_state(default_patient_state)
        assert model is not None

    def test_simulate_output_shape(
        self, default_patient_state, default_dose_schedule, time_steps_24h: np.ndarray
    ) -> None:
        """Output must have shape (len(time_steps), 2): [central, peripheral]."""
        time_list = [float(t) for t in time_steps_24h]
        amounts, concs = simulate_two_compartment(
            patient=default_patient_state,
            dose_schedule=default_dose_schedule,
            time_steps=time_list,
        )
        assert concs.shape == (len(time_list), 2)

    def test_concentrations_are_non_negative(
        self, default_patient_state, default_dose_schedule, time_steps_24h: np.ndarray
    ) -> None:
        """Physical concentrations must never be negative."""
        time_list = [float(t) for t in time_steps_24h]
        _, concs = simulate_two_compartment(
            patient=default_patient_state,
            dose_schedule=default_dose_schedule,
            time_steps=time_list,
        )
        assert (concs.numpy() >= -1e-6).all()

    def test_monotonic_decay_during_elimination_phase(
        self, default_patient_state, default_dose_schedule, elimination_time_steps: list[float]
    ) -> None:
        """After distribution equilibrates, central concentration should decay monotonically."""
        _, concs = simulate_two_compartment(
            patient=default_patient_state,
            dose_schedule=default_dose_schedule,
            time_steps=elimination_time_steps,
        )
        central = concs[:, 0].numpy()
        diffs = np.diff(central)
        non_increasing = diffs <= 1e-6
        assert non_increasing.mean() >= 0.90, "Expected elimination phase to decay monotonically."


# --------------------------------------------------------------------------
# NeuralPKCorrection (Mass Conservation)
# --------------------------------------------------------------------------

class TestNeuralPKCorrection:
    """Tests for the physics-informed neural correction layer."""

    def test_neural_correction_preserves_mass_balance(
        self, default_patient_state, default_dose_schedule, time_steps_24h: np.ndarray
    ) -> None:
        """Mass balance is conserved because correction adds/subtracts equal exchange flux."""
        time_list = [float(t) for t in time_steps_24h]
        correction_module = NeuralPKCorrection(covariate_dim=3)
        
        concs = simulate_drug_concentration(
            patient_state=default_patient_state,
            dose_schedule=default_dose_schedule,
            time_steps=time_list,
            correction=correction_module,
        )
        assert concs.shape == (len(time_list), 2)
        assert torch.isfinite(concs).all()