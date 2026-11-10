"""
src/rl_policy/train_cql.py

Continuous-action Conservative Q-Learning (CQL) training pipeline for the
AetherMed 3.0 offline RL dosing policy.

A self-contained PyTorch SAC-style actor-critic (GaussianPolicy + twin
QNetworks) augmented with the CQL conservative penalty is used instead of
`d3rlpy`, because `d3rlpy`'s common CQL entrypoints assume a discrete
action space and raise assertion errors for continuous mg-dose actions.
All actions are handled internally as normalized floats in [-1.0, 1.0].
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "PyTorch is required for src.rl_policy.train_cql. Install with `pip install torch`."
    ) from exc


# --------------------------------------------------------------------------- #
# Action normalization helpers
# --------------------------------------------------------------------------- #

from src.rl_policy.actions import (
    DOSE_MIN_MG,
    DOSE_MAX_MG,
    DOSE_CENTER_MG,
    DOSE_SCALE_MG,
    DRUG_DOSE_RANGES,
    normalize_action,
    denormalize_action,
)



# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

_RL_NPZ_KEYS = {"state", "action", "reward", "next_state", "done"}


def load_offline_trajectories(data_dir: str) -> Dict[str, np.ndarray]:
    """Load offline RL trajectories from a directory or a direct `.npz` file path.

    Accepts either:
      * A path directly to a single `.npz` file containing RL trajectory
        arrays (keys: state, action, reward, next_state, done).
      * A directory path, which is scanned (non-recursively) for `.npz`
        files containing the expected RL keys. Non-RL artifacts in the same
        directory (e.g. `synthetic_cohort.csv`) are silently ignored rather
        than raising a spurious FileNotFoundError.

    Args:
        data_dir: Path to either a `.npz` file or a directory containing one
            or more `.npz` trajectory files.

    Returns:
        Dictionary with keys 'observations', 'actions', 'rewards',
        'next_observations', 'terminals' as concatenated numpy arrays.

    Raises:
        FileNotFoundError: If the path does not exist, or no valid `.npz`
            RL trajectory files are found within a directory.
        ValueError: If a discovered `.npz` file is missing required keys.
    """
    path = Path(data_dir)

    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_dir}")

    npz_files: List[Path]
    if path.is_file():
        if path.suffix != ".npz":
            raise ValueError(f"Expected a .npz file, got: {path}")
        npz_files = [path]
    else:
        # Directory: keep only .npz files carrying the expected RL keys,
        # ignoring non-RL artifacts such as synthetic_cohort.csv.
        candidates = sorted(Path(p) for p in glob.glob(str(path / "*.npz")))
        npz_files = []
        for candidate in candidates:
            try:
                with np.load(candidate) as npz:
                    if _RL_NPZ_KEYS.issubset(set(npz.files)):
                        npz_files.append(candidate)
            except (OSError, ValueError):
                continue

        if not npz_files:
            raise FileNotFoundError(
                f"No valid RL trajectory .npz files found in directory: {data_dir}"
            )

    observations_list: List[np.ndarray] = []
    actions_list: List[np.ndarray] = []
    rewards_list: List[np.ndarray] = []
    next_observations_list: List[np.ndarray] = []
    terminals_list: List[np.ndarray] = []

    for npz_file in npz_files:
        with np.load(npz_file) as npz:
            missing = _RL_NPZ_KEYS - set(npz.files)
            if missing:
                raise ValueError(f"File '{npz_file}' is missing required keys: {missing}")

            observations_list.append(npz["state"].astype(np.float32))
            actions_list.append(npz["action"].astype(np.float32))
            rewards_list.append(npz["reward"].astype(np.float32))
            next_observations_list.append(npz["next_state"].astype(np.float32))
            terminals_list.append(npz["done"].astype(np.float32))

    return {
        "observations": np.concatenate(observations_list, axis=0),
        "actions": np.concatenate(actions_list, axis=0),
        "rewards": np.concatenate(rewards_list, axis=0),
        "next_observations": np.concatenate(next_observations_list, axis=0),
        "terminals": np.concatenate(terminals_list, axis=0),
    }


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #

LOG_STD_MIN: float = -5.0
LOG_STD_MAX: float = 2.0


class GaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian policy producing continuous actions in [-1, 1]."""

    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 256) -> None:
        """Initialize the policy network.

        Args:
            state_dim: Dimensionality of the observation space.
            action_dim: Dimensionality of the normalized action space.
            hidden_dim: Width of hidden layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the pre-tanh Gaussian mean and log-std for the given states.

        Args:
            state: Batch of states, shape (B, state_dim).

        Returns:
            Tuple of (mean, log_std), each shape (B, action_dim).
        """
        features = self.net(state)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std_head(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a reparameterized, tanh-squashed action in [-1, 1].

        Args:
            state: Batch of states, shape (B, state_dim).

        Returns:
            Tuple of (action, log_prob, mean_action): `action` is a
            stochastic tanh-squashed sample, `log_prob` includes the tanh
            Jacobian correction (shape (B, 1)), and `mean_action` is the
            deterministic tanh-squashed mean.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        raw_action = normal.rsample()
        action = torch.tanh(raw_action)

        log_prob = normal.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action


class QNetwork(nn.Module):
    """State-action value network for continuous (normalized) actions."""

    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 256) -> None:
        """Initialize the Q network.

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
        """Compute Q(s, a).

        Args:
            state: Batch of states, shape (B, state_dim).
            action: Batch of normalized actions, shape (B, action_dim).

        Returns:
            Q-values, shape (B, 1).
        """
        return self.net(torch.cat([state, action], dim=-1))


# --------------------------------------------------------------------------- #
# CQL Agent
# --------------------------------------------------------------------------- #

@dataclass
class CQLConfig:
    """Hyperparameter configuration for the continuous CQL agent."""

    state_dim: int = 5
    action_dim: int = 1
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    cql_alpha: float = 5.0
    num_cql_actions: int = 10
    device: str = "cpu"


class CQLAgent:
    """Self-contained PyTorch continuous-action Conservative Q-Learning agent.

    Implements a SAC-style actor-critic backbone (`GaussianPolicy` + twin
    `QNetwork`s with target networks) augmented with the CQL conservative
    penalty, operating entirely on normalized actions in [-1, 1]. This
    avoids the discrete-action assumptions of `d3rlpy` for this continuous
    mg-dose action space.
    """

    def __init__(self, config: Optional[CQLConfig] = None) -> None:
        """Initialize the agent, its networks, and optimizers.

        Args:
            config: Agent hyperparameters. Uses `CQLConfig()` defaults if
                not provided.
        """
        self.config = config or CQLConfig()
        device = torch.device(self.config.device)
        self.device = device

        self.policy = GaussianPolicy(
            self.config.state_dim, self.config.action_dim, self.config.hidden_dim
        ).to(device)

        self.q1 = QNetwork(self.config.state_dim, self.config.action_dim, self.config.hidden_dim).to(device)
        self.q2 = QNetwork(self.config.state_dim, self.config.action_dim, self.config.hidden_dim).to(device)
        self.q1_target = QNetwork(
            self.config.state_dim, self.config.action_dim, self.config.hidden_dim
        ).to(device)
        self.q2_target = QNetwork(
            self.config.state_dim, self.config.action_dim, self.config.hidden_dim
        ).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.policy_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=self.config.q_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=self.config.q_lr)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Polyak-average source network parameters into the target network.

        Args:
            source: Network providing updated weights.
            target: Network to be smoothed toward `source`.
        """
        tau = self.config.tau
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single CQL training step on a batch of transitions.

        Args:
            batch: Dictionary with 'observations', 'actions' (normalized,
                shape (B, action_dim)), 'rewards', 'next_observations',
                'terminals', all as torch tensors on `self.device`.

        Returns:
            Dictionary of scalar diagnostics: q1_loss, q2_loss, policy_loss.
        """
        state = batch["observations"]
        action = batch["actions"]
        reward = batch["rewards"]
        next_state = batch["next_observations"]
        terminal = batch["terminals"]

        # --- Critic update: Bellman backup + CQL conservative penalty ---
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state)
            target_q1 = self.q1_target(next_state, next_action)
            target_q2 = self.q2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_value = reward + self.config.gamma * (1.0 - terminal) * target_q

        current_q1 = self.q1(state, action)
        current_q2 = self.q2(state, action)
        bellman_loss_q1 = F.mse_loss(current_q1, target_value)
        bellman_loss_q2 = F.mse_loss(current_q2, target_value)

        cql_penalty_q1 = self._cql_penalty(self.q1, state, action)
        cql_penalty_q2 = self._cql_penalty(self.q2, state, action)

        q1_loss = bellman_loss_q1 + self.config.cql_alpha * cql_penalty_q1
        q2_loss = bellman_loss_q2 + self.config.cql_alpha * cql_penalty_q2

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # --- Policy update (maximize min-Q, entropy-regularized) ---
        new_action, log_prob, _ = self.policy.sample(state)
        q1_pi = self.q1(state, new_action)
        q2_pi = self.q2(state, new_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (0.01 * log_prob - min_q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "policy_loss": float(policy_loss.item()),
        }

    def _cql_penalty(
        self, q_network: QNetwork, state: torch.Tensor, dataset_action: torch.Tensor
    ) -> torch.Tensor:
        """Compute the CQL conservative penalty: logsumexp(Q) over sampled actions minus dataset Q.

        Args:
            q_network: The Q-network being regularized.
            state: Batch of states, shape (B, state_dim).
            dataset_action: The logged (behavior) normalized action for each
                state in the batch, shape (B, action_dim).

        Returns:
            Scalar conservative penalty tensor (mean over the batch).
        """
        batch_size = state.shape[0]
        num_samples = self.config.num_cql_actions

        random_actions = (
            torch.rand(batch_size * num_samples, self.config.action_dim, device=self.device) * 2.0
            - 1.0
        )
        repeated_states = state.unsqueeze(1).repeat(1, num_samples, 1).view(
            batch_size * num_samples, -1
        )

        with torch.no_grad():
            policy_actions, _, _ = self.policy.sample(repeated_states)

        q_random = q_network(repeated_states, random_actions).view(batch_size, num_samples)
        q_policy = q_network(repeated_states, policy_actions).view(batch_size, num_samples)
        q_dataset = q_network(state, dataset_action).view(batch_size, 1)

        cat_q = torch.cat([q_random, q_policy], dim=1)
        logsumexp_q = torch.logsumexp(cat_q, dim=1, keepdim=True)

        penalty = (logsumexp_q - q_dataset).mean()
        return penalty

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> float:
        """Select a normalized action for a single state.

        Args:
            state: Observation vector, shape (state_dim,).
            deterministic: If True, use the policy mean action; otherwise
                sample stochastically.

        Returns:
            Normalized action in [-1, 1].
        """
        self.policy.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _, mean_action = self.policy.sample(state_tensor)
            chosen = mean_action if deterministic else action
        self.policy.train()
        return float(chosen.cpu().numpy().flatten()[0])

    def act(self, observation: np.ndarray) -> float:
        """Alias for select_action to support standardized evaluation harness."""
        return self.select_action(observation)

    def get_dose_recommendation(
        self,
        state: np.ndarray,
        drug: str = "warfarin",
        organ_risk: Optional[float] = None,
        deterministic: bool = True,
    ) -> float:
        """Computes a clinically safe recommended dose in mg for the given drug.
        If organ_risk > 0.80, applies critical toxicity dampening to the minimum safe dose.
        """
        a_norm = self.select_action(state, deterministic=deterministic)
        if organ_risk is not None and organ_risk > 0.80:
            # Critical toxicity hold: clamp to bottom 10% of therapeutic dose
            a_norm = -0.85
        return float(denormalize_action(a_norm, drug=drug))


    def save(self, path: str) -> None:
        """Save policy/critic state dicts and config to a checkpoint file.

        Args:
            path: Destination file path (e.g. 'models/cql_policy.pt'). The
                parent directory is created automatically if missing.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "config": self.config.__dict__,
            },
            output_path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CQLAgent":
        """Load an agent from a checkpoint file.

        Args:
            path: Path to a checkpoint saved via `save()`.
            device: Device to load the agent onto.

        Returns:
            A `CQLAgent` instance with restored weights.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        config_dict = dict(checkpoint["config"])
        config_dict["device"] = device
        config = CQLConfig(**config_dict)

        agent = cls(config)
        agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        agent.q1.load_state_dict(checkpoint["q1_state_dict"])
        agent.q2.load_state_dict(checkpoint["q2_state_dict"])
        return agent


# --------------------------------------------------------------------------- #
# Training entrypoint
# --------------------------------------------------------------------------- #

def _to_tensor_batch(
    data: Dict[str, np.ndarray], indices: np.ndarray, device: "torch.device"
) -> Dict[str, "torch.Tensor"]:
    """Slice a numpy trajectory dict at `indices` and convert to torch tensors.

    Args:
        data: Trajectory dictionary; must already contain 'actions_norm'.
        indices: Row indices to gather for this batch.
        device: Target torch device.

    Returns:
        Dictionary of torch tensors ready for `CQLAgent.update`.
    """
    return {
        "observations": torch.as_tensor(data["observations"][indices], dtype=torch.float32, device=device),
        "actions": torch.as_tensor(data["actions_norm"][indices], dtype=torch.float32, device=device),
        "rewards": torch.as_tensor(data["rewards"][indices], dtype=torch.float32, device=device),
        "next_observations": torch.as_tensor(
            data["next_observations"][indices], dtype=torch.float32, device=device
        ),
        "terminals": torch.as_tensor(data["terminals"][indices], dtype=torch.float32, device=device),
    }


def train(
    data_dir: str,
    backend: str = "torch",
    num_epochs: int = 50,
    batch_size: int = 256,
    checkpoint_path: str = "models/cql_policy.pt",
    device: str = "cpu",
) -> str:
    """Train a continuous CQL policy on offline clinical trajectories.

    Args:
        data_dir: Path to a `.npz` trajectory file or a directory of them.
        backend: Training backend. Only `"torch"` (self-contained continuous
            CQL/SAC) is supported; this avoids `d3rlpy`'s discrete-action
            assumptions for this continuous mg-dose action space.
        num_epochs: Number of passes over the dataset.
        batch_size: Minibatch size per gradient step.
        checkpoint_path: Where to save the trained policy checkpoint.
        device: Torch device string, e.g. "cpu" or "cuda".

    Returns:
        Path to the saved checkpoint.

    Raises:
        ValueError: If an unsupported backend is requested.
    """
    if backend != "torch":
        raise ValueError(
            f"Unsupported backend '{backend}'. This continuous mg-dose action "
            "space requires the self-contained 'torch' backend; discrete-action "
            "'d3rlpy' backends are not compatible."
        )

    raw = load_offline_trajectories(data_dir)
    raw["actions_norm"] = normalize_action(raw["actions"])

    state_dim = raw["observations"].shape[1]
    action_dim = raw["actions_norm"].shape[1]

    config = CQLConfig(state_dim=state_dim, action_dim=action_dim, device=device)
    agent = CQLAgent(config)

    num_transitions = raw["observations"].shape[0]
    torch_device = torch.device(device)

    for epoch in range(num_epochs):
        permutation = np.random.permutation(num_transitions)
        epoch_losses: List[Dict[str, float]] = []

        for start in range(0, num_transitions, batch_size):
            batch_indices = permutation[start : start + batch_size]
            if batch_indices.size == 0:
                continue
            batch = _to_tensor_batch(raw, batch_indices, torch_device)
            losses = agent.update(batch)
            epoch_losses.append(losses)

        if epoch_losses:
            mean_q1 = float(np.mean([loss["q1_loss"] for loss in epoch_losses]))
            mean_policy = float(np.mean([loss["policy_loss"] for loss in epoch_losses]))
            print(f"[epoch {epoch + 1}/{num_epochs}] q1_loss={mean_q1:.4f} policy_loss={mean_policy:.4f}")

    agent.save(checkpoint_path)
    print(f"Saved trained CQL checkpoint to: {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    """CLI entrypoint for `python -m src.rl_policy.train_cql`."""
    parser = argparse.ArgumentParser(description="Train a continuous CQL dosing policy offline.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/rl_trajectories/trajectories.npz",
        help="Path to a .npz trajectory file or a directory containing one.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="torch",
        choices=["torch"],
        help="Training backend. Defaults to the self-contained continuous 'torch' CQL implementation.",
    )
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-path", type=str, default="models/cql_policy.pt")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        backend=args.backend,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()