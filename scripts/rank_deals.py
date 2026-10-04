#!/usr/bin/env python3
"""Rank training deals by difficulty for curriculum learning.

Plays the deterministic heuristic on candidate training seeds and buckets
them by outcome:

  * ``easy``   — heuristic wins (a dense win trajectory exists for RL to find)
  * ``medium`` — heuristic loses but reveals every face-down card (near-miss)
  * ``hard``   — heuristic loses with face-down cards remaining

Output: JSON ``{"easy": [...], "medium": [...], "hard": [...], "stats": {...}}``
usable as ``train_ppo.py --seed-pool <file>`` (a flat list or the whole dict —
``seed_pool`` accepts either; pass a filtered tier file for one phase).

    .venv/bin/python scripts/rank_deals.py --n 2000 --jobs 8 \
        --out benchmarks/deal_difficulty.json
"""

from __future__ import annotations

import argparse
import json
import time

from solitaire_rl.evaluator import run_eval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=1_000_000)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--variant", default="draw1", choices=["draw1", "draw3"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="benchmarks/deal_difficulty.json")
    args = ap.parse_args()

    seeds = list(range(args.lo, args.lo + args.n))
    t0 = time.perf_counter()
    summary = run_eval(
        "heuristic",
        seeds,
        {"variant": args.variant, "obs_variant": "pomdp"},
        jobs=args.jobs,
    )

    easy, medium, hard = [], [], []
    for r in summary["results"]:
        if r["outcome"] == "win":
            easy.append(r["seed"])
        elif r["facedown_left"] == 0:
            medium.append(r["seed"])
        else:
            hard.append(r["seed"])

    doc = {
        "variant": args.variant,
        "lo": args.lo,
        "n": args.n,
        "stats": {
            "easy": len(easy),
            "medium": len(medium),
            "hard": len(hard),
            "heuristic_win_rate": summary["win_rate"],
            "wall_s": round(time.perf_counter() - t0, 1),
        },
        "easy": easy,
        "medium": medium,
        "hard": hard,
    }
    with open(args.out, "w") as f:
        json.dump(doc, f)
    print(
        f"easy={len(easy)} medium={len(medium)} hard={len(hard)} "
        f"(heuristic win rate {summary['win_rate']:.3f}) -> {args.out} "
        f"wall={time.perf_counter() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
