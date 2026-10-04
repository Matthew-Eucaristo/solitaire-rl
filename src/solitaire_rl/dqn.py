"""Masked DQN — a compact self-contained implementation (train_dqn.py driver).

Small MLP Q-network over the fixed 654-action space; illegal actions are
masked out of both exploration and the bootstrap max (Double-DQN style
argmax on the online net, evaluation on the target net).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from solitaire_rl.actions import N_ACTIONS
from solitaire_rl.env import KlondikeEnv


def build_q(obs_dim: int, hidden: int = 256) -> nn.Module:
    return nn.Sequential(
        nn.Linear(obs_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, N_ACTIONS),
    )


def masked_argmax(q: torch.Tensor, mask: np.ndarray) -> int:
    q = q.detach().cpu().numpy().astype(np.float64)
    q[~mask] = -np.inf
    return int(np.argmax(q))


class DQNPolicy:
    """Eval-time wrapper: masked argmax over a DQN checkpoint."""

    def __init__(self, net: nn.Module, obs_dim: int, device: str = "cpu") -> None:
        self.net = net.eval()
        self.device = device
        self._obs_dim = obs_dim

    @staticmethod
    def load(path: str, env_kwargs: dict | None = None) -> DQNPolicy:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obs_dim = ckpt["obs_dim"]
        net = build_q(obs_dim, ckpt.get("hidden", 256))
        net.load_state_dict(ckpt["state_dict"])
        return DQNPolicy(net, obs_dim)

    def act(self, env: KlondikeEnv) -> int:
        obs = torch.as_tensor(env._obs(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = self.net(obs)[0]
        return masked_argmax(q, env.action_masks())
