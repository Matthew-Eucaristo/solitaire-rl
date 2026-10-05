"""Maskable QRDQN — sb3-contrib QRDQN with legal-move masks at rollout time.

sb3-contrib ships QRDQN without mask support, and in this env an illegal
action ends the episode instantly (OUTCOME_ILLEGAL), so both the ε-greedy
exploration AND the warmup random sampling must be mask-aware. This subclass
overrides ``_sample_action``; the TD bootstrap max is left unmasked
(ReplayBuffer does not store next-state masks) — a documented limitation.

Eval-time wrapper ``QRDQNPolicy`` masks the quantile-mean argmax.
"""

from __future__ import annotations

import numpy as np
import torch

from solitaire_rl.env import KlondikeEnv


class MaskableQRDQN:
    """Factory returning an sb3-contrib QRDQN subclass with masked sampling."""

    @staticmethod
    def build():
        from sb3_contrib import QRDQN

        class _Masked(QRDQN):
            def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
                masks = np.stack(self.env.env_method("action_masks"))
                if self.num_timesteps < learning_starts:
                    action = np.array(
                        [np.random.choice(np.flatnonzero(m)) for m in masks],
                        dtype=np.int64,
                    )
                else:
                    obs_t = torch.as_tensor(self._last_obs, device=self.policy.device)
                    with torch.no_grad():
                        qm = self.policy.quantile_net(obs_t).mean(dim=1).cpu().numpy()
                    qm[~masks] = -np.inf
                    action = qm.argmax(axis=1).astype(np.int64)
                    explore = np.random.rand(self.env.num_envs) < self.exploration_rate
                    for i in np.flatnonzero(explore):
                        action[i] = np.random.choice(np.flatnonzero(masks[i]))
                return action, action

        return _Masked


class QRDQNPolicy:
    """Eval-time wrapper: masked argmax over mean quantiles of a QRDQN ckpt."""

    def __init__(self, model) -> None:
        self.model = model

    @staticmethod
    def load(path: str, env_kwargs: dict | None = None) -> QRDQNPolicy:
        if path.endswith(".pt"):
            return QRDQNPolicy._load_packed(path, env_kwargs)
        from sb3_contrib import QRDQN

        model = QRDQN.load(path, device="cpu")
        model.policy.eval()
        return QRDQNPolicy(model)

    @staticmethod
    def _load_packed(path: str, env_kwargs: dict | None = None) -> QRDQNPolicy:
        """Load a fp16-packed policy (state_dict without the target net)."""
        import gymnasium as gym
        from sb3_contrib.qrdqn.policies import QRDQNPolicy as _SB3QRPolicy

        from solitaire_rl.actions import N_ACTIONS

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        env = KlondikeEnv(**(env_kwargs or {}))
        policy = _SB3QRPolicy(
            env.observation_space,
            gym.spaces.Discrete(N_ACTIONS),
            lr_schedule=lambda _: 0.0,
            net_arch=ckpt.get("net_arch", [256, 256]),
        )
        sd = {k: v.float() for k, v in ckpt["state_dict"].items()}
        policy.load_state_dict(sd, strict=False)
        policy.eval()
        return QRDQNPolicy(_PolicyShim(policy))

    def act(self, env: KlondikeEnv) -> int:
        obs = torch.as_tensor(env._obs(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            qm = self.model.policy.quantile_net(obs).mean(dim=1)[0].cpu().numpy()
        qm[~env.action_masks()] = -np.inf
        return int(np.argmax(qm))


class _PolicyShim:
    """Bare policy object exposing `.policy` like an sb3 model for eval."""

    def __init__(self, policy) -> None:
        self.policy = policy
