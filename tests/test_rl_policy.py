"""
tests/test_rl_policy.py

Tests for the Offline RL dosing policy stack:
    - Continuous action normalization/denormalization ([0, 200] mg <-> [-1, 1]).
    - Loading the trained ``CQLAgent`` checkpoint (``models/cql_policy.pt``).
    - Running ``OffPolicyEvaluator`` (``src/rl_policy/evaluate_ope.py``) and
      validating the resulting ``OPEResult`` (V_beta, V_FQE, V_WIS, V_DR).

ASSUMED INTERFACE
-----------------
These modules were not available at the time this test file was written.
Tests are written against the documented public interface:

    # src/rl_policy/actions.py (or wherever normalize/denormalize live)
    def normalize_action(action_mg: float, low: float = 0.0, high: float = 200.0) -> float: ...
    def denormalize_action(action_norm: float, low: float = 0.0, high: float = 200.0) -> float: ...

    # src/rl_policy/cql_agent.py
    class CQLAgent:
        @classmethod
        def load(cls, checkpoint_path: str | Path) -> "CQLAgent": ...
        def act(self, observation) -> float: ...

    # src/rl_policy/evaluate_ope.py
    class OPEResult:
        V_beta: float   # behavior-policy value estimate
        V_FQE: float    # Fitted Q Evaluation estimate
        V_WIS: float    # Weighted Importance Sampling estimate
        V_DR: float     # Doubly Robust estimate

    class OffPolicyEvaluator:
        def __init__(self, agent, dataset): ...
        def evaluate(self) -> OPEResult: ...

If the real module's names/signatures differ, update the imports below
accordingly — the assertions encode the mathematical/range invariants a
correct implementation should satisfy (round-trip normalization, bounded
action ranges, finite/plausible OPE value estimates) regardless of exact
call signature.

Checkpoint-dependent tests (``TestCQLAgentCheckpoint`` and
``TestOffPolicyEvaluation``) are skipped automatically if
``models/cql_policy.pt`` does not exist in the working directory, so this
file can still be collected and partially run in environments where the
trained artifact hasn't been produced yet (e.g. CI unit-test stage before
a training job runs).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed; skipping RL policy tests")

actions_mod = pytest.importorskip(
    "src.rl_policy.actions",
    reason="src.rl_policy.actions not available; skipping action normalization tests",
)
normalize_action = getattr(actions_mod, "normalize_action", None)
denormalize_action = getattr(actions_mod, "denormalize_action", None)

pytestmark_actions = pytest.mark.skipif(
    normalize_action is None or denormalize_action is None,
    reason="src.rl_policy.actions does not expose normalize_action/denormalize_action",
)

ACTION_LOW_MG = 0.0
ACTION_HIGH_MG = 200.0

CHECKPOINT_PATH = Path("models/cql_policy.pt")


# --------------------------------------------------------------------------
# Action normalization / denormalization
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    normalize_action is None or denormalize_action is None,
    reason="normalize_action/denormalize_action not available",
)
class TestActionNormalization:
    """Tests for continuous dose <-> policy-action space conversions."""

    def test_min_dose_maps_to_negative_one(self) -> None:
        assert normalize_action(ACTION_LOW_MG) == pytest.approx(-1.0)

    def test_max_dose_maps_to_positive_one(self) -> None:
        assert normalize_action(ACTION_HIGH_MG) == pytest.approx(1.0)

    def test_midpoint_dose_maps_to_zero(self) -> None:
        midpoint_mg = (ACTION_LOW_MG + ACTION_HIGH_MG) / 2.0
        assert normalize_action(midpoint_mg) == pytest.approx(0.0, abs=1e-6)

    def test_negative_one_denormalizes_to_min_dose(self) -> None:
        assert denormalize_action(-1.0) == pytest.approx(ACTION_LOW_MG)

    def test_positive_one_denormalizes_to_max_dose(self) -> None:
        assert denormalize_action(1.0) == pytest.approx(ACTION_HIGH_MG)

    def test_zero_denormalizes_to_midpoint_dose(self) -> None:
        midpoint_mg = (ACTION_LOW_MG + ACTION_HIGH_MG) / 2.0
        assert denormalize_action(0.0) == pytest.approx(midpoint_mg)

    @pytest.mark.parametrize("dose_mg", [0.0, 25.0, 50.0, 100.0, 150.0, 200.0])
    def test_normalize_denormalize_roundtrip(self, dose_mg: float) -> None:
        """normalize -> denormalize should be a near-identity for in-range doses."""
        normalized = normalize_action(dose_mg)
        recovered = denormalize_action(normalized)
        assert recovered == pytest.approx(dose_mg, abs=1e-6)

    @pytest.mark.parametrize("action_norm", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_denormalize_normalize_roundtrip(self, action_norm: float) -> None:
        """denormalize -> normalize should be a near-identity for in-range actions."""
        dose_mg = denormalize_action(action_norm)
        recovered = normalize_action(dose_mg)
        assert recovered == pytest.approx(action_norm, abs=1e-6)

    def test_normalized_action_stays_within_unit_bounds_for_in_range_doses(self) -> None:
        doses = np.linspace(ACTION_LOW_MG, ACTION_HIGH_MG, 25)
        normalized = np.array([normalize_action(float(d)) for d in doses])
        assert np.all(normalized >= -1.0 - 1e-6)
        assert np.all(normalized <= 1.0 + 1e-6)

    def test_denormalized_action_stays_within_dose_bounds_for_unit_range(self) -> None:
        norms = np.linspace(-1.0, 1.0, 25)
        doses = np.array([denormalize_action(float(a)) for a in norms])
        assert np.all(doses >= ACTION_LOW_MG - 1e-6)
        assert np.all(doses <= ACTION_HIGH_MG + 1e-6)


# --------------------------------------------------------------------------
# CQLAgent checkpoint loading
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason=f"Trained checkpoint not found at {CHECKPOINT_PATH}; skipping checkpoint-dependent tests",
)
class TestCQLAgentCheckpoint:
    """Tests for loading and running the trained offline-RL dosing policy."""

    @pytest.fixture(scope="class")
    def cql_agent(self):
        cql_mod = pytest.importorskip(
            "src.rl_policy.cql_agent",
            reason="src.rl_policy.cql_agent not available",
        )
        CQLAgent = getattr(cql_mod, "CQLAgent", None)
        if CQLAgent is None:
            pytest.skip("src.rl_policy.cql_agent does not expose CQLAgent")
        return CQLAgent.load(CHECKPOINT_PATH)

    def test_checkpoint_loads_without_error(self, cql_agent) -> None:
        assert cql_agent is not None

    def test_agent_produces_action_for_observation(self, cql_agent) -> None:
        """Given a plausible observation vector, act() should return a finite,
        in-bounds normalized action."""
        dummy_obs = np.random.default_rng(0).normal(size=(16,)).astype(np.float32)
        action = cql_agent.act(dummy_obs)
        action_value = float(np.asarray(action).squeeze())
        assert np.isfinite(action_value)
        assert -1.0 - 1e-3 <= action_value <= 1.0 + 1e-3

    def test_agent_is_deterministic_for_same_observation_in_eval_mode(self, cql_agent) -> None:
        dummy_obs = np.random.default_rng(1).normal(size=(16,)).astype(np.float32)
        action1 = float(np.asarray(cql_agent.act(dummy_obs)).squeeze())
        action2 = float(np.asarray(cql_agent.act(dummy_obs)).squeeze())
        assert action1 == pytest.approx(action2)


# --------------------------------------------------------------------------
# Off-Policy Evaluation
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason=f"Trained checkpoint not found at {CHECKPOINT_PATH}; skipping OPE tests",
)
class TestOffPolicyEvaluation:
    """Tests for OffPolicyEvaluator / OPEResult value estimation."""

    @pytest.fixture(scope="class")
    def ope_result(self):
        cql_mod = pytest.importorskip("src.rl_policy.cql_agent", reason="cql_agent module unavailable")
        ope_mod = pytest.importorskip(
            "src.rl_policy.evaluate_ope", reason="evaluate_ope module unavailable"
        )
        dataset_mod = pytest.importorskip(
            "src.rl_policy.dataset", reason="dataset module unavailable"
        )

        CQLAgent = getattr(cql_mod, "CQLAgent", None)
        OffPolicyEvaluator = getattr(ope_mod, "OffPolicyEvaluator", None)
        ClinicalTrajectoryDataset = getattr(dataset_mod, "ClinicalTrajectoryDataset", None)

        if CQLAgent is None or OffPolicyEvaluator is None or ClinicalTrajectoryDataset is None:
            pytest.skip("Required RL policy classes not found in src.rl_policy.*")

        agent = CQLAgent.load(CHECKPOINT_PATH)
        eval_dataset = ClinicalTrajectoryDataset(
            num_patients=10, horizon=8, obs_dim=16, action_dim=1, seed=7
        ).generate()

        evaluator = OffPolicyEvaluator(agent=agent, dataset=eval_dataset)
        return evaluator.evaluate()

    def test_ope_result_has_all_value_estimates(self, ope_result) -> None:
        for field in ("V_beta", "V_FQE", "V_WIS", "V_DR"):
            assert hasattr(ope_result, field), f"OPEResult missing field: {field}"

    def test_ope_value_estimates_are_finite(self, ope_result) -> None:
        for field in ("V_beta", "V_FQE", "V_WIS", "V_DR"):
            value = getattr(ope_result, field)
            assert np.isfinite(value), f"{field} is not finite: {value}"

    def test_ope_value_estimates_are_reasonable_magnitude(self, ope_result) -> None:
        """Value estimates should be within a plausible reward-scale envelope;
        this is a loose sanity bound to catch gross numerical blow-ups
        (e.g. exploding importance weights in V_WIS), not a tight correctness
        check on the estimator's accuracy."""
        for field in ("V_beta", "V_FQE", "V_WIS", "V_DR"):
            value = getattr(ope_result, field)
            assert abs(value) < 1e6, f"{field} is implausibly large: {value}"

    def test_fqe_and_dr_estimates_are_not_wildly_divergent(self, ope_result) -> None:
        """FQE and Doubly-Robust estimates are both model-based/hybrid estimators
        and should not diverge by orders of magnitude on the same small dataset."""
        v_fqe = ope_result.V_FQE
        v_dr = ope_result.V_DR
        denom = max(abs(v_fqe), abs(v_dr), 1e-6)
        relative_diff = abs(v_fqe - v_dr) / denom
        assert relative_diff < 10.0, (
            f"V_FQE ({v_fqe}) and V_DR ({v_dr}) diverge implausibly on a small eval set"
        )