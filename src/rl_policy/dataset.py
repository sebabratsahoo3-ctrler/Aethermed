"""
src/rl_policy/dataset.py

Synthetic offline trajectory generation and storage utilities for the
AetherMed 3.0 in-silico clinical trial digital twin RL policy module.

This module produces offline reinforcement-learning trajectories suitable
for training a Conservative Q-Learning (CQL) dosing policy, and persists
them in a `.npz` schema consumed by `src.rl_policy.train_cql`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


class ClinicalTrajectoryDataset:
    """Generates and persists synthetic offline RL trajectories for CQL training.

    Trajectories represent simulated patient episodes where the state is a
    physiological feature vector, the action is a raw drug dose in
    milligrams, and the reward reflects a synthetic clinical utility signal
    (efficacy net of a dose-dependent toxicity penalty).
    """

    #: Discrete dose choices (mg) sampled uniformly by the synthetic behavior policy.
    BEHAVIOR_DOSES: np.ndarray = np.array([0.0, 50.0, 100.0, 150.0, 200.0], dtype=np.float32)

    def __init__(self, data_dir: str = "data/processed/rl_trajectories") -> None:
        """Initialize the dataset handler.

        Args:
            data_dir: Directory used as the default location for reading/writing
                trajectory artifacts.
        """
        self.data_dir: str = data_dir
        self._trajectories: Optional[Dict[str, np.ndarray]] = None

    def generate_synthetic_trajectories(
        self,
        num_episodes: int = 500,
        episode_length: int = 10,
        state_dim: int = 5,
    ) -> Dict[str, np.ndarray]:
        """Generate synthetic offline trajectories via a uniform-random behavior policy.

        Args:
            num_episodes: Number of simulated patient episodes.
            episode_length: Number of timesteps per episode.
            state_dim: Dimensionality of the observation vector.

        Returns:
            Dictionary with keys 'observations' (N, state_dim), 'actions'
            (N, 1) raw mg doses, 'rewards' (N, 1), 'next_observations'
            (N, state_dim), and 'terminals' (N, 1), where N = num_episodes *
            episode_length and terminals is 1.0 on the final step of each
            episode, 0.0 otherwise.

        Raises:
            ValueError: If num_episodes, episode_length, or state_dim are non-positive.
        """
        if num_episodes <= 0 or episode_length <= 0 or state_dim <= 0:
            raise ValueError("num_episodes, episode_length, and state_dim must all be positive.")

        rng = np.random.default_rng(seed=42)

        total_steps = num_episodes * episode_length
        observations = np.zeros((total_steps, state_dim), dtype=np.float32)
        actions = np.zeros((total_steps, 1), dtype=np.float32)
        rewards = np.zeros((total_steps, 1), dtype=np.float32)
        next_observations = np.zeros((total_steps, state_dim), dtype=np.float32)
        terminals = np.zeros((total_steps, 1), dtype=np.float32)

        idx = 0
        for _episode in range(num_episodes):
            # Initialize a synthetic physiological state vector (e.g. vitals/labs).
            state = rng.normal(loc=0.0, scale=1.0, size=state_dim).astype(np.float32)

            for t in range(episode_length):
                # Behavior policy: uniform sampling over 5 discrete mg doses.
                dose = float(rng.choice(self.BEHAVIOR_DOSES))

                # Synthetic transition dynamics: dose nudges state toward a
                # healthier equilibrium (0-vector), with process noise.
                dose_effect = (dose / 200.0) * 0.5
                noise = rng.normal(loc=0.0, scale=0.1, size=state_dim).astype(np.float32)
                next_state = (state * 0.9 - state * dose_effect * 0.1 + dose_effect + noise).astype(
                    np.float32
                )

                # Synthetic reward: reward proximity of the next state to
                # equilibrium (efficacy) and penalize excessive dosing (toxicity).
                efficacy = -float(np.sum(np.abs(next_state)))
                toxicity_penalty = -0.01 * (dose**2) / 200.0
                reward = efficacy * 0.1 + toxicity_penalty

                is_terminal = 1.0 if t == episode_length - 1 else 0.0

                observations[idx] = state
                actions[idx, 0] = dose
                rewards[idx, 0] = reward
                next_observations[idx] = next_state
                terminals[idx, 0] = is_terminal

                state = next_state
                idx += 1

        trajectories: Dict[str, np.ndarray] = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_observations,
            "terminals": terminals,
        }
        self._trajectories = trajectories
        return trajectories

    def save_trajectories_for_cql(
        self,
        output_file: str = "data/processed/rl_trajectories/trajectories.npz",
    ) -> str:
        """Save generated trajectories to disk in the CQL-compatible `.npz` schema.

        If trajectories have not yet been generated in this instance, they
        are generated first using default parameters. The output directory
        is created automatically if it does not already exist.

        Args:
            output_file: Destination path for the `.npz` archive.

        Returns:
            The absolute path of the written file.

        Raises:
            OSError: If the output directory cannot be created.
        """
        if self._trajectories is None:
            self._trajectories = self.generate_synthetic_trajectories()

        output_path = Path(output_file)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(
                f"Failed to create output directory '{output_path.parent}': {exc}"
            ) from exc

        np.savez(
            output_path,
            state=self._trajectories["observations"],
            action=self._trajectories["actions"],
            reward=self._trajectories["rewards"],
            next_state=self._trajectories["next_observations"],
            done=self._trajectories["terminals"],
        )

        return str(output_path.resolve())


if __name__ == "__main__":
    dataset = ClinicalTrajectoryDataset()
    saved_path = dataset.save_trajectories_for_cql()
    print(f"Saved synthetic trajectories to: {saved_path}")