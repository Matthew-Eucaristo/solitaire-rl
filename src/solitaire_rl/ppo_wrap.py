"""Eval-time wrapper around a MaskablePPO checkpoint (sb3-contrib)."""

from __future__ import annotations

import numpy as np

from solitaire_rl.env import KlondikeEnv


class PPOPolicy:
    def __init__(self, model) -> None:
        self.model = model

    @staticmethod
    def load(path: str, env_kwargs: dict | None = None) -> PPOPolicy:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(path, device="cpu")
        return PPOPolicy(model)

    def act(self, env: KlondikeEnv) -> int:
        obs = env._obs()
        mask = env.action_masks()
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)


def masked_sample(model, obs: np.ndarray, mask: np.ndarray) -> int:
    """Deterministic masked argmax — used by eval and recordings."""
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    return int(action)
