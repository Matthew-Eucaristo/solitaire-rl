"""Gymnasium environment for Klondike Solitaire.

Termination (episode ends as a loss unless `is_win`):
  * win           — all 52 cards on foundations
  * concede       — the agent chose the CONCEDE action (learned resignation)
  * no_moves      — no legal game moves remain
  * no_progress   — exact state signature repeated (deterministic cycle) or
                    `max_idle` moves without a progress event (reveal /
                    foundation / waste play)
  * truncated     — `max_steps` reached

Rewards — see REWARDS.md. Every shaped term is individually toggleable.
"""

from __future__ import annotations

from collections import deque
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from solitaire_rl.actions import (
    ACTION_CONCEDE,
    N_ACTIONS,
    action_to_move,
)
from solitaire_rl.moves import Move, MoveKind, apply_move, legal_moves
from solitaire_rl.obs import OBS_VARIANTS, encode_state, obs_dim
from solitaire_rl.state import DRAW1, VARIANTS, GameState, deal

# Outcome labels reported in info["outcome"].
OUTCOME_WIN = "win"
OUTCOME_CONCEDE = "concede"
OUTCOME_NO_MOVES = "no_moves"
OUTCOME_CYCLE = "no_progress_cycle"
OUTCOME_IDLE = "no_progress_idle"
OUTCOME_TRUNCATED = "truncated"
OUTCOME_ILLEGAL = "illegal_action"

# Training deals are sampled uniformly from [TRAIN_SEED_MIN, 2**31). Benchmark
# deals are seeds 0..999 and are never produced by this sampler — see
# DECISIONS.md.
TRAIN_SEED_MIN = 1_000_000

DEFAULT_REWARD_CONFIG: dict[str, float] = {
    "win": 1.0,  # bonus on winning the episode
    "foundation": 1.0 / 52.0,  # per card moved to a foundation
    "reveal": 0.15,  # per face-down card flipped
    "waste_play": 0.02,  # per waste card moved to the tableau
    "idle_after": 20,  # idle moves tolerated before `idle_penalty` applies
    "idle_penalty": -0.005,  # per move with no progress after `idle_after`
    "recycle_penalty": -0.02,  # per redeal (draw-1 has ~24 free draws/cycle)
    "illegal": -0.1,  # choosing a masked-out action
    "concede": 0.0,  # choosing to resign
    "loss": 0.0,  # any non-win termination
}


class KlondikeEnv(gym.Env):
    """Klondike Solitaire as a masked-action Gymnasium env."""

    metadata: ClassVar[dict] = {"render_modes": ["ansi"]}

    def __init__(
        self,
        variant: str = DRAW1,
        obs_variant: str = "pomdp",
        reward_mode: str = "shaped",
        reward_config: dict[str, float] | None = None,
        frame_stack: int = 8,
        max_steps: int = 500,
        max_idle: int = 150,
        max_redeals: int | None = None,
        train_seeds: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if obs_variant not in OBS_VARIANTS:
            raise ValueError(f"unknown obs_variant {obs_variant!r}")
        if reward_mode not in ("shaped", "sparse"):
            raise ValueError(f"unknown reward_mode {reward_mode!r}")
        self.variant = variant
        self.obs_variant = obs_variant
        self.reward_mode = reward_mode
        self.reward_config = {**DEFAULT_REWARD_CONFIG, **(reward_config or {})}
        self.frame_stack = max(1, frame_stack)
        self.max_steps = max_steps
        self.max_idle = max_idle
        self.max_redeals = max_redeals
        self.train_seeds = train_seeds
        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(N_ACTIONS)
        base_dim = obs_dim(obs_variant)
        low = -1.0 if obs_variant.startswith("compact") else 0.0
        self.observation_space = spaces.Box(
            low=low, high=1.0, shape=(self.frame_stack * base_dim,), dtype=np.float32
        )

        self.state: GameState | None = None
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._steps = 0
        self._idle = 0
        self._seen: set[tuple] = set()
        self._outcome: str | None = None
        self._legal_cache: list[Move] | None = None

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is None and options and "seed" in options:
            seed = options["seed"]
        if seed is None:
            if not self.train_seeds:
                raise ValueError("env created with train_seeds=False; pass an explicit seed")
            seed = int(self._rng.integers(TRAIN_SEED_MIN, 2**31 - 1))
        self.deal_seed = seed
        self.state = deal(seed, variant=self.variant, max_redeals=self.max_redeals)
        self._steps = 0
        self._idle = 0
        self._seen = {self.state.signature()}
        self._outcome = None
        self._legal_cache = None
        self._frames.clear()
        raw = self._raw_obs()
        for _ in range(self.frame_stack):
            self._frames.append(raw)
        return self._obs(), self._info()

    def step(self, action: int):
        assert self.state is not None, "call reset() first"
        if self._outcome is not None:
            raise RuntimeError("step() after episode end; call reset()")
        action = int(action)
        st = self.state
        reward = 0.0
        terminated = truncated = False
        cfg = self.reward_config

        if action == ACTION_CONCEDE:
            terminated = True
            self._outcome = OUTCOME_CONCEDE
            reward += cfg["concede"]
        else:
            legal = self._legal()
            legal_ids = self._legal_ids(legal)
            if action not in legal_ids:
                # Illegal (masked-out) action: penalize, end the episode.
                terminated = True
                self._outcome = OUTCOME_ILLEGAL
                reward += cfg["illegal"] + cfg["loss"]
                return self._finish(reward, terminated, truncated)

            move = action_to_move(action, st)
            outcome = apply_move(st, move)
            self._legal_cache = None
            self._steps += 1

            if self.reward_mode == "shaped":
                if outcome.to_foundation:
                    reward += cfg["foundation"]
                if outcome.revealed:
                    reward += cfg["reveal"]
                if outcome.from_waste and not outcome.to_foundation:
                    reward += cfg["waste_play"]
                if outcome.recycled:
                    reward += cfg["recycle_penalty"]

            if outcome.progress:
                self._idle = 0
            else:
                self._idle += 1
                if self.reward_mode == "shaped" and self._idle > cfg["idle_after"]:
                    reward += cfg["idle_penalty"]

            if st.is_win():
                terminated = True
                self._outcome = OUTCOME_WIN
                reward += cfg["win"]
            elif not self._legal():
                terminated = True
                self._outcome = OUTCOME_NO_MOVES
                reward += cfg["loss"]
            else:
                sig = st.signature()
                if sig in self._seen:
                    terminated = True
                    self._outcome = OUTCOME_CYCLE
                    reward += cfg["loss"]
                else:
                    self._seen.add(sig)
                    if self._idle >= self.max_idle:
                        terminated = True
                        self._outcome = OUTCOME_IDLE
                        reward += cfg["loss"]

            if not terminated and self._steps >= self.max_steps:
                truncated = True
                self._outcome = OUTCOME_TRUNCATED

        return self._finish(reward, terminated, truncated)

    def _finish(self, reward: float, terminated: bool, truncated: bool):
        raw = self._raw_obs()
        self._frames.append(raw)
        return self._obs(), reward, terminated, truncated, self._info()

    # ------------------------------------------------------------- helpers
    def _legal(self) -> list[Move]:
        if self._legal_cache is None:
            assert self.state is not None
            self._legal_cache = legal_moves(self.state)
        return self._legal_cache

    @staticmethod
    def _legal_ids(moves: list[Move]) -> set[int]:
        from solitaire_rl.actions import move_to_action

        return {move_to_action(m) for m in moves}

    def action_masks(self) -> np.ndarray:
        """Boolean action mask — required by sb3-contrib MaskablePPO."""
        mask = np.zeros(N_ACTIONS, dtype=bool)
        for a in self._legal_ids(self._legal()):
            mask[a] = True
        mask[ACTION_CONCEDE] = True
        return mask

    def legal_ids(self) -> list[int]:
        return [*sorted(self._legal_ids(self._legal())), ACTION_CONCEDE]

    def _raw_obs(self) -> np.ndarray:
        st = self.state
        assert st is not None
        moves = self._legal()
        can_draw = any(m.kind == MoveKind.DRAW for m in moves)
        return encode_state(
            st,
            self.obs_variant,
            idle_ratio=self._idle / max(1, self.max_idle),
            has_legal_move=bool(moves),
            can_draw=can_draw,
        )

    def _obs(self) -> np.ndarray:
        return np.concatenate(list(self._frames)).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        st = self.state
        assert st is not None
        return {
            "action_mask": self.action_masks(),
            "outcome": self._outcome,
            "seed": self.deal_seed,
            "moves": st.moves,
            "foundations": st.foundation_total(),
            "facedown": st.facedown_count(),
            "redeals": st.redeals,
            "is_win": st.is_win(),
        }

    # --------------------------------------------------------------- render
    def render(self, mode: str = "ansi") -> str:
        from solitaire_rl.render import render_text

        assert self.state is not None
        return render_text(self.state)
