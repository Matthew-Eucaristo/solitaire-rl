#!/usr/bin/env python3
"""Evaluate a policy on the fixed benchmark deals.

python eval.py --policy heuristic --deals 1000 --variant draw1 --jobs 8
python eval.py --policy dqn:runs/dqn/q_final.pt --deals 1000
python eval.py --policy ppo:runs/ppo/ppo_final.zip --deals 1000
"""

from __future__ import annotations

import argparse
import json
import os

from solitaire_rl.evaluator import load_benchmark_seeds, print_summary, run_eval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--policy", required=True, help="random | heuristic | dqn:<ckpt.pt> | ppo:<model.zip>"
    )
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument(
        "--deal-seeds", default=None, help="deals.json path (default benchmarks/deals.json)"
    )
    ap.add_argument("--variant", default="draw1", choices=["draw1", "draw3"])
    ap.add_argument(
        "--obs",
        default="pomdp",
        choices=["pomdp", "perfect", "compact", "compact_perfect", "compact_hint"],
    )
    ap.add_argument("--frame-stack", type=int, default=8)
    ap.add_argument("--reward-mode", default="shaped", choices=["shaped", "sparse"])
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=None, help="write summary JSON here")
    args = ap.parse_args()

    seeds = load_benchmark_seeds(args.deal_seeds)[: args.deals]
    env_kwargs = {
        "variant": args.variant,
        "obs_variant": args.obs,
        "frame_stack": args.frame_stack,
        "reward_mode": args.reward_mode,
        "max_steps": args.max_steps,
        "train_seeds": False,
    }
    summary = run_eval(args.policy, seeds, env_kwargs, jobs=args.jobs)
    print_summary(summary)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
