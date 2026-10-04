"""Evaluation harness — plays policies over a fixed set of deal seeds.

The benchmark deal set lives in benchmarks/deals.json (seeds 0..999 by
default). All comparisons happen on those same deals; training never sees
them (train seeds are sampled from [1_000_000, 2^31) — see DECISIONS.md).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass

from solitaire_rl.env import KlondikeEnv

BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "deals.json")


def load_benchmark_seeds(path: str | None = None) -> list[int]:
    path = path or os.path.normpath(BENCHMARK_PATH)
    with open(path) as f:
        return json.load(f)["seeds"]


@dataclass
class EpisodeResult:
    seed: int
    outcome: str
    moves: int
    foundations: int
    facedown_left: int
    redeals: int
    wall_ms: float
    total_reward: float = 0.0


def play_episode(policy, seed: int, env_kwargs: dict) -> EpisodeResult:
    """Play one episode of `policy` on deal `seed`. Deterministic for fixed
    (policy, seed, env_kwargs)."""
    env = KlondikeEnv(**env_kwargs)
    env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    t0 = time.perf_counter()
    total_reward = 0.0
    while True:
        action = policy.act(env)
        _, r, terminated, truncated, info = env.step(action)
        total_reward += r
        if terminated or truncated:
            break
    return EpisodeResult(
        seed=seed,
        outcome=info["outcome"],
        moves=info["moves"],
        foundations=info["foundations"],
        facedown_left=info["facedown"],
        redeals=info["redeals"],
        wall_ms=(time.perf_counter() - t0) * 1000,
        total_reward=total_reward,
    )


# ----------------------------------------------------------------------
# Multiprocessing workers — policy built once per worker process.
# ----------------------------------------------------------------------

_WORKER = None


def _init_worker(spec: str, env_kwargs: dict):
    global _WORKER
    from solitaire_rl.policies import load_policy

    _WORKER = load_policy(spec, env_kwargs, seed=12345)


def _run_one(args):
    seed, env_kwargs = args
    return play_episode(_WORKER, seed, env_kwargs)


def run_eval(
    spec: str,
    seeds: list[int],
    env_kwargs: dict | None = None,
    jobs: int = 1,
) -> dict:
    """Evaluate policy `spec` on `seeds`; returns the summary dict."""
    env_kwargs = env_kwargs or {"variant": "draw1", "obs_variant": "pomdp"}
    t0 = time.perf_counter()
    if jobs and jobs > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            jobs, initializer=_init_worker, initargs=(spec, env_kwargs)
        ) as pool:
            results: list[EpisodeResult] = pool.map(_run_one, [(s, env_kwargs) for s in seeds])
    else:
        _init_worker(spec, env_kwargs)
        results = [_run_one((s, env_kwargs)) for s in seeds]

    wins = [r for r in results if r.outcome == "win"]
    outcome_counts: dict[str, int] = {}
    for r in results:
        outcome_counts[r.outcome] = outcome_counts.get(r.outcome, 0) + 1

    summary = {
        "policy": spec,
        "env": env_kwargs,
        "n_deals": len(seeds),
        "seed_min": min(seeds),
        "seed_max": max(seeds),
        "wins": len(wins),
        "win_rate": len(wins) / len(seeds),
        "avg_moves": sum(r.moves for r in results) / len(results),
        "avg_foundations": sum(r.foundations for r in results) / len(results),
        "avg_wall_ms_per_game": sum(r.wall_ms for r in results) / len(results),
        "avg_reward": sum(r.total_reward for r in results) / len(results),
        "outcomes": outcome_counts,
        "wall_sec_total": time.perf_counter() - t0,
        "results": [vars(r) for r in results],
    }
    return summary


def print_summary(summary: dict) -> None:
    print(f"policy={summary['policy']}  env={summary['env']}")
    print(
        f"  deals={summary['n_deals']}  wins={summary['wins']}  "
        f"win_rate={summary['win_rate']:.4f}"
    )
    print(
        f"  avg_moves={summary['avg_moves']:.1f}  "
        f"avg_foundations={summary['avg_foundations']:.1f}  "
        f"avg_wall_ms={summary['avg_wall_ms_per_game']:.1f}"
    )
    print(f"  outcomes={summary['outcomes']}")
    print(f"  total_wall={summary['wall_sec_total']:.1f}s")
