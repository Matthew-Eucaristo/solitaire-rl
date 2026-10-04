"""Record episodes to GIFs — renders each step's board via PIL frames.

python -m solitaire_rl.record --policy heuristic --seeds 0 1 2 --out results/recordings
"""

from __future__ import annotations

import argparse
import os

from solitaire_rl.env import KlondikeEnv
from solitaire_rl.policies import load_policy
from solitaire_rl.render import render_frame

_ACTION_NAMES = {
    0: "draw",
    8: "waste->fdn",
    653: "concede",
}


def describe_action(a: int) -> str:
    if a in _ACTION_NAMES:
        return _ACTION_NAMES[a]
    if 1 <= a <= 7:
        return f"waste->t{a - 1}"
    if 9 <= a <= 15:
        return f"t{a - 9}->fdn"
    rel = a - 16
    src, rem = divmod(rel, 91)
    dst, j = divmod(rem, 13)
    return f"t{src}->t{dst}[{j}]"


def record_episode(
    policy,
    seed: int,
    env_kwargs: dict,
    out_path: str,
    fps: float = 4.0,
    max_frames: int = 600,
) -> dict:
    """Play one episode and write a GIF. Returns the episode summary."""
    env = KlondikeEnv(**env_kwargs)
    env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(seed)

    frames = [render_frame(env.state, footer=f"seed={seed} start")]
    total_reward = 0.0
    while True:
        action = policy.act(env)
        _, r, term, trunc, info = env.step(action)
        total_reward += r
        footer = f"seed={seed} a={describe_action(action)} r={r:+.2f} Σ={total_reward:+.2f}"
        frames.append(render_frame(env.state, footer=footer))
        if term or trunc:
            break

    outcome = info["outcome"]
    frames.append(render_frame(env.state, footer=f">>> {outcome} moves={info['moves']}"))

    # Subsample to stay under max_frames while keeping start/end.
    if len(frames) > max_frames:
        step = len(frames) / max_frames
        frames = [frames[int(i * step)] for i in range(max_frames - 1)] + [frames[-1]]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    return {
        "seed": seed,
        "outcome": outcome,
        "moves": info["moves"],
        "foundations": info["foundations"],
        "gif": out_path,
        "n_frames": len(frames),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="heuristic")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--variant", default="draw1", choices=["draw1", "draw3"])
    ap.add_argument(
        "--obs",
        default="pomdp",
        choices=["pomdp", "perfect", "compact", "compact_perfect"],
    )
    ap.add_argument("--out", default="results/recordings")
    ap.add_argument("--fps", type=float, default=4.0)
    args = ap.parse_args()

    env_kwargs = {"variant": args.variant, "obs_variant": args.obs}
    policy = load_policy(args.policy, env_kwargs)
    tag = args.policy.replace(":", "_").replace("/", "_")
    for seed in args.seeds:
        path = os.path.join(args.out, f"{tag}_{args.variant}_seed{seed}.gif")
        res = record_episode(policy, seed, env_kwargs, path, fps=args.fps)
        print(res)


if __name__ == "__main__":
    main()
