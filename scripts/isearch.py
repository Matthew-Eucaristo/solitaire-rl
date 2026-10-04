#!/usr/bin/env python3
"""Information-set rollout policy for the Klondike POMDP.

For each decision the policy:
  1. Samples ``--worlds`` *determinizations* of the hidden information
     (face-down tableau cards + stock order), consistent with every card
     the agent has actually observed so far.
  2. For each candidate action — the top-K legal actions by the guiding
     PPO policy plus the heuristic's pick — clones each world and rolls
     it out with the fast deterministic heuristic for ``--horizon``
     steps.
  3. Plays the action with the best mean score (win=1, else
     foundations/52 as a progress proxy).

This is legitimate POMDP reasoning: the agent never sees the real hidden
cards, only samples from the multiset of cards it has not observed. It is
slow (~0.3s/move) so it is an eval-time policy, not a web-app one.

    .venv/bin/python scripts/isearch.py --policy ppo:runs/x.zip --deals 200 --worlds 8
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from solitaire_rl.actions import ACTION_CONCEDE, action_to_move
from solitaire_rl.env import KlondikeEnv
from solitaire_rl.moves import apply_move, legal_moves
from solitaire_rl.policies import HeuristicPolicy, load_policy
from solitaire_rl.state import GameState


def unseen_cards(state: GameState) -> list:
    """Cards the player cannot see: face-down tableau + stock contents."""
    hidden = [c for col in state.tableau for c in col.down]
    hidden += list(state.stock)
    return hidden


def determinize(state: GameState, rng: np.random.Generator) -> GameState:
    """Clone `state` with hidden cards replaced by a random permutation
    of the unseen multiset — a world consistent with the observation."""
    det = state.clone()
    perm = rng.permutation(len(unseen_cards(state)))
    pool = unseen_cards(state)
    i = 0
    for col in det.tableau:
        for j in range(len(col.down)):
            col.down[j] = pool[perm[i]]
            i += 1
    for j in range(len(det.stock)):
        det.stock[j] = pool[perm[i]]
        i += 1
    return det


class _Shim:
    """Minimal env surface HeuristicPolicy needs (state, _legal, _seen)."""

    def __init__(self, state: GameState) -> None:
        self.state = state
        self._seen: set = {state.signature()}
        self._legal_cache = None

    def _legal(self):
        if self._legal_cache is None:
            self._legal_cache = legal_moves(self.state)
        return self._legal_cache

    def step_done(self):
        self._seen.add(self.state.signature())
        self._legal_cache = None


def rollout(state: GameState, pol: HeuristicPolicy, horizon: int) -> float:
    """Heuristic rollout from `state` (mutated); returns 1.0 on win else
    foundations/52 progress."""
    shim = _Shim(state)
    for _ in range(horizon):
        if state.is_win():
            return 1.0
        a = pol.act(shim)
        if a == ACTION_CONCEDE:
            break
        m = action_to_move(a, state)
        if m is None:
            break
        apply_move(state, m)
        shim.step_done()
    return 1.0 if state.is_win() else state.foundation_total() / 52.0


def candidate_actions(env: KlondikeEnv, guide, k: int) -> list[int]:
    """Top-K legal actions by the guide policy's distribution + its argmax."""
    import torch

    obs = torch.as_tensor(env._obs(), dtype=torch.float32).unsqueeze(0)
    mask = torch.as_tensor(env.action_masks()).unsqueeze(0)
    with torch.no_grad():
        dist = guide.model.policy.get_distribution(obs)
        logits = dist.distribution.logits[0].clone()
    logits[~mask[0]] = -1e9
    top = torch.topk(logits, k=min(k, int(mask.sum()))).indices.tolist()
    cand = set(top)
    cand.add(guide.act(env))
    return [a for a in cand if env.action_masks()[a]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="guide policy spec")
    ap.add_argument("--obs", default="compact_hint")
    ap.add_argument("--deals", type=int, default=200)
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kw = dict(variant="draw1", obs_variant=args.obs, frame_stack=8,
              reward_mode="shaped", max_steps=500, train_seeds=False)
    guide = load_policy(args.policy, kw)
    hp = HeuristicPolicy()
    rng = np.random.default_rng(args.seed)
    env = KlondikeEnv(**kw)

    wins = 0
    moves_tot = 0
    t0 = time.perf_counter()
    for seed in range(args.deals):
        env.reset(seed=seed)
        while True:
            st = env.state
            cands = candidate_actions(env, guide, args.topk)
            if len(cands) <= 1:
                a = cands[0] if cands else ACTION_CONCEDE
            else:
                scores = {a: 0.0 for a in cands}
                for _ in range(args.worlds):
                    det = determinize(st, rng)
                    for a in cands:
                        if a == ACTION_CONCEDE:
                            continue
                        m = action_to_move(a, det)
                        if m is None:
                            continue
                        w = det.clone()
                        apply_move(w, m)
                        scores[a] += rollout(w, hp, args.horizon)
                best = max(scores, key=scores.get)
                # Defer to the guide when the search can't discriminate:
                # rollouts that converge to the same outcome carry no
                # signal, and the learned policy is the better tiebreak.
                guide_a = guide.act(env)
                a = best if scores[best] - scores.get(guide_a, 0.0) > 0.02 else guide_a
            _, _, term, trunc, _ = env.step(a)
            moves_tot += 1
            if term or trunc:
                if env._outcome == "win":
                    wins += 1
                break
        if (seed + 1) % 10 == 0:
            el = time.perf_counter() - t0
            print(f"{seed+1}/{args.deals}  wins={wins}  ({el:.0f}s)", flush=True)

    print(f"isearch worlds={args.worlds} topk={args.topk} "
          f"deals={args.deals} wins={wins} win_rate={wins/args.deals:.4f} "
          f"avg_moves={moves_tot/args.deals:.0f} wall={time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
