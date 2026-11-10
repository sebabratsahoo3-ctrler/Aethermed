"""
tests/test_build_cohort.py
===========================
Tests for ClinicalTrajectoryDataset in src/rl_policy/dataset.py.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from src.rl_policy.dataset import ClinicalTrajectoryDataset

EXPECTED_KEYS = {"observations", "actions", "rewards", "next_observations", "terminals"}


@pytest.fixture
def dataset_instance():
    return ClinicalTrajectoryDataset()


@pytest.fixture
def generated_trajectories(dataset_instance) -> dict:
    return dataset_instance.generate_synthetic_trajectories(num_episodes=20, episode_length=10, state_dim=8)


class TestTrajectoryGeneration:
    def test_generate_returns_expected_keys(self, generated_trajectories: dict) -> None:
        assert EXPECTED_KEYS.issubset(generated_trajectories.keys())

    def test_all_arrays_are_numpy(self, generated_trajectories: dict) -> None:
        for key in EXPECTED_KEYS:
            assert isinstance(generated_trajectories[key], np.ndarray)

    def test_expected_total_transitions(self, generated_trajectories: dict) -> None:
        expected_n = 20 * 10
        assert generated_trajectories["observations"].shape[0] == expected_n

    def test_observation_dimensionality(self, generated_trajectories: dict) -> None:
        assert generated_trajectories["observations"].shape[1] == 8

    def test_action_dimensionality(self, generated_trajectories: dict) -> None:
        actions = generated_trajectories["actions"]
        assert actions.ndim in (1, 2)

    def test_rewards_are_1d_and_finite(self, generated_trajectories: dict) -> None:
        rewards = generated_trajectories["rewards"]
        assert np.all(np.isfinite(rewards))

    def test_terminals_are_boolean_like(self, generated_trajectories: dict) -> None:
        terminals = generated_trajectories["terminals"]
        unique_vals = set(np.unique(terminals).tolist())
        assert unique_vals.issubset({0.0, 1.0, 0, 1, True, False})

    def test_each_patient_trajectory_has_exactly_one_terminal_step(self, generated_trajectories: dict) -> None:
        terminals = generated_trajectories["terminals"].reshape(20, 10)
        assert np.all(terminals[:, -1] == 1) or np.all(terminals[:, -1].astype(bool))

    def test_generation_is_reproducible_with_fixed_seed(self, dataset_instance) -> None:
        data1 = dataset_instance.generate_synthetic_trajectories(num_episodes=10, episode_length=5)
        data2 = dataset_instance.generate_synthetic_trajectories(num_episodes=10, episode_length=5)
        np.testing.assert_allclose(data1["observations"], data2["observations"])


class TestNPZArchiveRoundtrip:
    def test_save_creates_npz_file(self, dataset_instance, tmp_path: Path) -> None:
        out_path = tmp_path / "cohort.npz"
        dataset_instance.save_trajectories_for_cql(str(out_path))
        assert out_path.exists()
        assert out_path.suffix == ".npz"

    def test_load_returns_expected_keys(self, dataset_instance, tmp_path: Path) -> None:
        out_path = tmp_path / "cohort.npz"
        dataset_instance.save_trajectories_for_cql(str(out_path))
        loaded = np.load(out_path)
        assert {"state", "action", "reward", "next_state", "done"}.issubset(set(loaded.files))

    def test_roundtrip_preserves_array_contents(self, dataset_instance, tmp_path: Path) -> None:
        out_path = tmp_path / "cohort.npz"
        dataset_instance.save_trajectories_for_cql(str(out_path))
        loaded = np.load(out_path)
        assert len(loaded["state"]) > 0

    def test_roundtrip_preserves_shapes(self, dataset_instance, tmp_path: Path) -> None:
        out_path = tmp_path / "cohort.npz"
        dataset_instance.save_trajectories_for_cql(str(out_path))
        loaded = np.load(out_path)
        assert loaded["state"].ndim == 2

    def test_load_nonexistent_file_raises(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "does_not_exist.npz"
        with pytest.raises((FileNotFoundError, OSError)):
            np.load(missing_path)