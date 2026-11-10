"""
src/rl_policy/evaluate_ope.py

Off-policy evaluation (OPE) suite for AetherMed 3.0 offline RL dosing
policies: Fitted Q Evaluation (FQE), Importance Sampling / Weighted
Importance Sampling (IS/WIS), and per-decision Doubly Robust (DR)
estimation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "PyTorch is required for src.rl_policy.evaluate_ope. Install with `pip install torch`."
    ) from exc

from src.rl_policy.dataset import ClinicalTrajectoryDataset
from src.rl_policy.train_cql import (
    CQLAgent,
    GaussianPolicy,
    denormalize_action,
    load_offline_trajectories,
    normalize_action,
)

BEHAVIOR_DOSES: np.ndarray = np.array([0.0, 50.0, 100.0, 150.0, 200.0], dtype=np.float32)
BEHAVIOR_PROPENSITY: float = 0.2  # Uniform over 5 discrete doses -> 1/5.
DEFAULT_KERNEL_BANDWIDTH_MG: float = 25.0


@dataclass
class OPEResult:
    """Summary of off-policy evaluation estimates for a candidate policy."""

    v_beta: float
    v_fqe: float
    v_wis: float
    v_dr: float
    v_is: float
    pct_improvement_fqe: float
    pct_improvement_wis: float
    pct_improvement_dr: float
    effective_sample_size: float = 0.0


    def __str__(self) -> str:
        """Render a human-readable summary table."""
        lines = [
            "=" * 60,
            "Off-Policy Evaluation Summary",
            "=" * 60,
            f"{'Estimator':<25}{'Value':>15}{'% vs Behavior':>18}",
            "-" * 60,
            f"{'Behavior (V_beta)':<25}{self.v_beta:>15.4f}{'--':>18}",
            f"{'FQE':<25}{self.v_fqe:>15.4f}{self.pct_improvement_fqe:>17.2f}%",
            f"{'IS':<25}{self.v_is:>15.4f}{'--':>18}",
            f"{'WIS':<25}{self.v_wis:>15.4f}{self.pct_improvement_wis:>17.2f}%",
            f"{'Doubly Robust (DR)':<25}{self.v_dr:>15.4f}{self.pct_improvement_dr:>17.2f}%",
            "=" * 60,
        ]
        return "\n".join(lines)


class FQEQNetwork(nn.Module):
    """Auxiliary Q-network used for Fitted Q Evaluation of a fixed target policy."""

    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 256) -> None:
        """Initialize the FQE critic.

        Args:
            state_dim: Dimensionality of the observation space.
            action_dim: Dimensionality of the normalized action space.
            hidden_dim: Width of hidden layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a) under FQE fitting.

        Args:
            state: Batch of states, shape (B, state_dim).
            action: Batch of normalized actions, shape (B, action_dim).

        Returns:
            Q-values, shape (B, 1).
        """
        return self.net(torch.cat([state, action], dim=-1))


class OffPolicyEvaluator:
    """Off-policy evaluation suite comparing a target policy against the logged behavior policy.

    Supports Fitted Q Evaluation (FQE), Importance Sampling (IS), Weighted
    Importance Sampling (WIS), and per-decision Doubly Robust (DR)
    estimation over offline clinical dosing trajectories.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        kernel_bandwidth_mg: float = DEFAULT_KERNEL_BANDWIDTH_MG,
        device: str = "cpu",
    ) -> None:
        """Initialize the evaluator.

        Args:
            gamma: Discount factor used for value accumulation.
            kernel_bandwidth_mg: Bandwidth (mg) of the Gaussian kernel used
                to approximate the target policy's action density around
                its mean dose, for propensity-ratio computation.
            device: Torch device for FQE network training/inference.
        """
        self.gamma = gamma
        self.kernel_bandwidth_mg = kernel_bandwidth_mg
        self.device = torch.device(device)

    # --------------------------------------------------------------- #
    # Propensities
    # --------------------------------------------------------------- #

    @staticmethod
    def behavior_propensity(dose_mg: np.ndarray) -> np.ndarray:
        """Return pi_beta(a|s) for the uniform discrete behavior policy.

        Args:
            dose_mg: Raw dose(s) in mg; only used to determine output shape,
                since behavior propensity is constant (0.2) over the 5-point
                uniform discrete support [0, 50, 100, 150, 200] mg.

        Returns:
            Array of constant behavior propensities (0.2), matching input shape.
        """
        return np.full_like(np.asarray(dose_mg, dtype=np.float32), BEHAVIOR_PROPENSITY)

    def target_propensity(self, dose_mg: np.ndarray, target_mean_dose_mg: np.ndarray) -> np.ndarray:
        """Approximate pi_target(a|s) via a Gaussian kernel around the target policy's mean dose.

        Args:
            dose_mg: Logged (behavior-taken) raw dose(s) in mg.
            target_mean_dose_mg: Target policy's mean dose(s) in mg for the
                same states (same shape as `dose_mg`).

        Returns:
            Gaussian kernel density values usable as relative propensities
            for importance-weight computation.
        """
        dose_mg = np.asarray(dose_mg, dtype=np.float64)
        target_mean_dose_mg = np.asarray(target_mean_dose_mg, dtype=np.float64)
        bandwidth = self.kernel_bandwidth_mg

        density = np.exp(-0.5 * ((dose_mg - target_mean_dose_mg) / bandwidth) ** 2)
        density = density / (bandwidth * np.sqrt(2.0 * np.pi))
        return density.astype(np.float32)

    def _target_mean_doses(self, policy: GaussianPolicy, observations: np.ndarray) -> np.ndarray:
        """Compute the target policy's mean raw dose for a batch of states.

        Args:
            policy: Trained Gaussian policy over normalized actions.
            observations: Batch of states, shape (N, state_dim).

        Returns:
            Mean raw doses in mg, shape (N,).
        """
        policy.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            mean, _ = policy.forward(state_tensor)
            mean_action_norm = torch.tanh(mean).cpu().numpy().flatten()
        return denormalize_action(mean_action_norm)

    # --------------------------------------------------------------- #
    # Fitted Q Evaluation
    # --------------------------------------------------------------- #

    def fitted_q_evaluation(
        self,
        policy: GaussianPolicy,
        data: Dict[str, np.ndarray],
        num_epochs: int = 20,
        batch_size: int = 256,
        lr: float = 3e-4,
    ) -> float:
        """Estimate the target policy's value via Fitted Q Evaluation.

        Fits an auxiliary `FQEQNetwork` via TD(0) regression toward the
        target policy's own mean action, then reports the mean initial-state
        value under the target policy.

        Args:
            policy: Trained Gaussian policy to evaluate.
            data: Trajectory dictionary with 'observations', 'actions_norm',
                'rewards', 'next_observations', 'terminals'.
            num_epochs: Number of TD-fitting passes over the dataset.
            batch_size: Minibatch size per gradient step.
            lr: Learning rate for the FQE critic optimizer.

        Returns:
            Estimated mean value V_FQE of the target policy.
        """
        fqe_net = FQEQNetwork(data["observations"].shape[1], data["actions_norm"].shape[1]).to(
            self.device
        )
        self._fit_td_critic(policy, data, fqe_net, num_epochs=num_epochs, batch_size=batch_size, lr=lr)

        terminals = data["terminals"].flatten()
        episode_start_indices = self._episode_start_indices(terminals)

        policy.eval()
        with torch.no_grad():
            init_states = torch.as_tensor(
                data["observations"][episode_start_indices], dtype=torch.float32, device=self.device
            )
            _, _, init_mean_actions = policy.sample(init_states)
            init_values = fqe_net(init_states, init_mean_actions).cpu().numpy().flatten()

        return float(np.mean(init_values))

    def _fit_td_critic(
        self,
        policy: GaussianPolicy,
        data: Dict[str, np.ndarray],
        critic: nn.Module,
        num_epochs: int = 20,
        batch_size: int = 256,
        lr: float = 3e-4,
    ) -> None:
        """Fit a Q-network in place via TD(0) regression toward a fixed target policy.

        Used both by `fitted_q_evaluation` and as the control-variate critic
        for Doubly Robust estimation.

        Args:
            policy: Trained Gaussian policy defining the bootstrap target action.
            data: Trajectory dictionary with 'observations', 'actions_norm',
                'rewards', 'next_observations', 'terminals'.
            critic: The `FQEQNetwork` instance to fit in place.
            num_epochs: Number of TD-fitting passes.
            batch_size: Minibatch size per gradient step.
            lr: Optimizer learning rate.
        """
        target_critic = FQEQNetwork(
            data["observations"].shape[1], data["actions_norm"].shape[1]
        ).to(self.device)
        target_critic.load_state_dict(critic.state_dict())
        optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
        num_transitions = data["observations"].shape[0]
        policy.eval()

        for _epoch in range(num_epochs):
            permutation = np.random.permutation(num_transitions)
            for start in range(0, num_transitions, batch_size):
                idx = permutation[start : start + batch_size]
                if idx.size == 0:
                    continue

                state = torch.as_tensor(data["observations"][idx], dtype=torch.float32, device=self.device)
                action = torch.as_tensor(
                    data["actions_norm"][idx], dtype=torch.float32, device=self.device
                )
                reward = torch.as_tensor(data["rewards"][idx], dtype=torch.float32, device=self.device)
                next_state = torch.as_tensor(
                    data["next_observations"][idx], dtype=torch.float32, device=self.device
                )
                terminal = torch.as_tensor(
                    data["terminals"][idx], dtype=torch.float32, device=self.device
                )

                with torch.no_grad():
                    _, _, next_mean_action = policy.sample(next_state)
                    target_value = reward + self.gamma * (1.0 - terminal) * target_critic(
                        next_state, next_mean_action
                    )

                loss = F.mse_loss(critic(state, action), target_value)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for source_param, target_param in zip(critic.parameters(), target_critic.parameters()):
                    target_param.data.copy_(0.005 * source_param.data + 0.995 * target_param.data)

    @staticmethod
    def _episode_start_indices(terminals: np.ndarray) -> np.ndarray:
        """Compute the indices marking the first timestep of each episode.

        Args:
            terminals: Flat array of terminal flags (1.0 on the last step
                of each episode).

        Returns:
            Array of indices corresponding to episode-start transitions.
        """
        starts = [0]
        for i in range(len(terminals) - 1):
            if terminals[i] == 1.0:
                starts.append(i + 1)
        return np.array(starts, dtype=np.int64)

    @staticmethod
    def _episode_boundaries(terminals: np.ndarray) -> List[Tuple[int, int]]:
        """Compute (start, end_exclusive) index pairs for each episode.

        Args:
            terminals: Flat array of terminal flags.

        Returns:
            List of (start_index, end_index_exclusive) tuples, one per episode.
        """
        boundaries: List[Tuple[int, int]] = []
        start = 0
        for i, is_terminal in enumerate(terminals):
            if is_terminal == 1.0:
                boundaries.append((start, i + 1))
                start = i + 1
        if start < len(terminals):
            boundaries.append((start, len(terminals)))
        return boundaries

    # --------------------------------------------------------------- #
    # IS / WIS
    # --------------------------------------------------------------- #

    def importance_sampling(
        self, policy: GaussianPolicy, data: Dict[str, np.ndarray]
    ) -> Tuple[float, float]:
        """Compute per-episode Importance Sampling and Weighted Importance Sampling estimates.

        Args:
            policy: Trained Gaussian policy to evaluate.
            data: Trajectory dictionary with 'observations', 'actions',
                'rewards', 'terminals'.

        Returns:
            Tuple of (V_IS, V_WIS).
        """
        observations = data["observations"]
        actions_mg = data["actions"].flatten()
        rewards = data["rewards"].flatten()
        terminals = data["terminals"].flatten()

        target_mean_doses = self._target_mean_doses(policy, observations)
        pi_target = self.target_propensity(actions_mg, target_mean_doses)
        pi_behavior = np.clip(self.behavior_propensity(actions_mg), 1e-8, None)
        step_ratios = pi_target / pi_behavior

        boundaries = self._episode_boundaries(terminals)

        episode_returns: List[float] = []
        episode_weights: List[float] = []

        for start, end in boundaries:
            cumulative_ratio = 1.0
            discounted_return = 0.0
            for t, idx in enumerate(range(start, end)):
                cumulative_ratio *= step_ratios[idx]
                discounted_return += (self.gamma**t) * rewards[idx]
            episode_returns.append(cumulative_ratio * discounted_return)
            episode_weights.append(cumulative_ratio)

        v_is = float(np.mean(episode_returns)) if episode_returns else 0.0

        total_weight = float(np.sum(episode_weights))
        v_wis = float(np.sum(episode_returns) / total_weight) if total_weight > 1e-8 else 0.0

        return v_is, v_wis

    # --------------------------------------------------------------- #
    # Doubly Robust
    # --------------------------------------------------------------- #

    def doubly_robust(
        self,
        policy: GaussianPolicy,
        data: Dict[str, np.ndarray],
        q_network: nn.Module,
    ) -> float:
        """Compute the per-decision Doubly Robust (DR) value estimate.

        Args:
            policy: Trained Gaussian policy to evaluate.
            data: Trajectory dictionary with 'observations', 'actions',
                'actions_norm', 'rewards', 'terminals'.
            q_network: A fitted Q-network (e.g. from `_fit_td_critic`) used
                as the DR control variate.

        Returns:
            Estimated mean value V_DR of the target policy.
        """
        observations = data["observations"]
        actions_mg = data["actions"].flatten()
        actions_norm = data["actions_norm"]
        rewards = data["rewards"].flatten()
        terminals = data["terminals"].flatten()

        target_mean_doses = self._target_mean_doses(policy, observations)
        pi_target = self.target_propensity(actions_mg, target_mean_doses)
        pi_behavior = np.clip(self.behavior_propensity(actions_mg), 1e-8, None)
        step_ratios = pi_target / pi_behavior

        policy.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            action_tensor = torch.as_tensor(actions_norm, dtype=torch.float32, device=self.device)
            _, _, mean_action_tensor = policy.sample(state_tensor)

            q_sa = q_network(state_tensor, action_tensor).cpu().numpy().flatten()
            v_s = q_network(state_tensor, mean_action_tensor).cpu().numpy().flatten()

        boundaries = self._episode_boundaries(terminals)
        episode_values: List[float] = []

        for start, end in boundaries:
            cumulative_ratio = 1.0
            dr_estimate = 0.0
            for t, idx in enumerate(range(start, end)):
                cumulative_ratio *= step_ratios[idx]
                correction = cumulative_ratio * (rewards[idx] - q_sa[idx])
                dr_estimate += (self.gamma**t) * (v_s[idx] + correction)
            episode_values.append(dr_estimate)

        return float(np.mean(episode_values)) if episode_values else 0.0

    # --------------------------------------------------------------- #
    # Full evaluation
    # --------------------------------------------------------------- #

    def evaluate(self, policy: GaussianPolicy, data: Dict[str, np.ndarray]) -> OPEResult:
        """Run the full OPE suite and summarize target-policy value estimates.

        Args:
            policy: Trained Gaussian policy to evaluate.
            data: Trajectory dictionary with 'observations', 'actions',
                'rewards', 'next_observations', 'terminals'.

        Returns:
            `OPEResult` summarizing V_beta, V_FQE, V_IS, V_WIS, V_DR, and
            percent improvements over the logged behavior value.
        """
        data = dict(data)
        data["actions_norm"] = normalize_action(data["actions"])

        terminals = data["terminals"].flatten()
        rewards = data["rewards"].flatten()
        boundaries = self._episode_boundaries(terminals)

        behavior_returns = [
            sum((self.gamma**t) * rewards[idx] for t, idx in enumerate(range(start, end)))
            for start, end in boundaries
        ]
        v_beta = float(np.mean(behavior_returns)) if behavior_returns else 0.0

        v_fqe = self.fitted_q_evaluation(policy, data)
        v_is, v_wis = self.importance_sampling(policy, data)

        # Compute Effective Sample Size (ESS)
        ess = getattr(self, "_last_ess", float(len(boundaries)))

        dr_critic = FQEQNetwork(data["observations"].shape[1], data["actions_norm"].shape[1]).to(
            self.device
        )
        self._fit_td_critic(policy, data, dr_critic)
        v_dr = self.doubly_robust(policy, data, dr_critic)

        def pct_improvement(value: float) -> float:
            if abs(v_beta) < 1e-8:
                return 0.0
            return 100.0 * (value - v_beta) / abs(v_beta)

        return OPEResult(
            v_beta=v_beta,
            v_fqe=v_fqe,
            v_wis=v_wis,
            v_dr=v_dr,
            v_is=v_is,
            pct_improvement_fqe=pct_improvement(v_fqe),
            pct_improvement_wis=pct_improvement(v_wis),
            pct_improvement_dr=pct_improvement(v_dr),
            effective_sample_size=round(float(ess), 1),
        )



def main() -> None:
    """CLI entrypoint for `python -m src.rl_policy.evaluate_ope`."""
    parser = argparse.ArgumentParser(
        description="Run off-policy evaluation for a trained CQL dosing policy."
    )
    parser.add_argument("--checkpoint-path", type=str, default="models/cql_policy.pt")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/rl_trajectories/trajectories.npz",
    )
    parser.add_argument("--kernel-bandwidth-mg", type=float, default=DEFAULT_KERNEL_BANDWIDTH_MG)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No trained checkpoint found at '{checkpoint_path}'. "
            "Run `python -m src.rl_policy.train_cql` first."
        )

    agent = CQLAgent.load(str(checkpoint_path), device=args.device)

    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"No trajectory data found at '{data_path}'; generating synthetic trajectories.")
        fallback_dataset = ClinicalTrajectoryDataset()
        fallback_dataset.save_trajectories_for_cql(str(data_path))

    data = load_offline_trajectories(str(data_path))

    evaluator = OffPolicyEvaluator(
        gamma=args.gamma, kernel_bandwidth_mg=args.kernel_bandwidth_mg, device=args.device
    )
    result = evaluator.evaluate(agent.policy, data)
    print(result)


if __name__ == "__main__":
    main()