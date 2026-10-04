"""Play a few games with text rendering — M0 demo.

    python -m solitaire_rl.demo --episodes 5 [--policy heuristic] [--seed 0]
"""

from __future__ import annotations

import argparse

from solitaire_rl.env import KlondikeEnv
from solitaire_rl.policies import load_policy
from solitaire_rl.render import render_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--policy", default="heuristic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="draw1", choices=["draw1", "draw3"])
    ap.add_argument("--max-render-steps", type=int, default=500)
    args = ap.parse_args()

    env = KlondikeEnv(variant=args.variant)
    policy = load_policy(args.policy, {"variant": args.variant})

    for ep in range(args.episodes):
        seed = args.seed + ep
        env.reset(seed=seed)
        print("=" * 64)
        print(f"EPISODE {ep}  seed={seed}  variant={args.variant}  policy={args.policy}")
        print("=" * 64)
        print(render_text(env.state))
        steps = 0
        while True:
            action = policy.act(env)
            _, r, term, trunc, info = env.step(action)
            steps += 1
            if steps <= args.max_render_steps or term or trunc:
                print(f"\n-- move {info['moves']}  action={action}  r={r:+.3f} --")
                print(render_text(env.state))
            if term or trunc:
                print(f"\n>>> outcome={info['outcome']}  moves={info['moves']}  "
                      f"foundations={info['foundations']}/52")
                break


if __name__ == "__main__":
    main()
