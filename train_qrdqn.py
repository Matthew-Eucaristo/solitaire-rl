#!/usr/bin/env python3
"""Train MaskableQRDQN (masked-rollout QRDQN) on Klondike.

    python train_qrdqn.py --steps 1000000 --variant draw1 --obs compact_hint \
        --device cpu --seed 0 --out runs/qrdqn_hint

Checkpoints + metrics.csv land in --out (runs/ is gitignored).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time

from stable_baselines3.common.vec_env import SubprocVecEnv

from solitaire_rl.env import KlondikeEnv
from solitaire_rl.qrdqn import MaskableQRDQN, QRDQNPolicy


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def make_env(env_kwargs: dict):
    def _f():
        return KlondikeEnv(**env_kwargs)

    return _f


def masked_eval(model, env_kwargs: dict, seeds: list[int]) -> float:
    env = KlondikeEnv(**env_kwargs)
    pol = QRDQNPolicy(model)
    wins = 0
    for s in seeds:
        env.reset(seed=s)
        while True:
            _, _, term, trunc, info = env.step(pol.act(env))
            if term or trunc:
                wins += int(info["outcome"] == "win")
                break
    return wins / len(seeds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--variant", default="draw1", choices=["draw1", "draw3"])
    ap.add_argument(
        "--obs",
        default="compact_hint",
        choices=["pomdp", "perfect", "compact", "compact_perfect", "compact_hint"],
    )
    ap.add_argument("--frame-stack", type=int, default=8)
    ap.add_argument("--reward-mode", default="shaped", choices=["shaped", "sparse"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--learning-starts", type=int, default=10_000)
    ap.add_argument("--train-freq", type=int, default=4)
    ap.add_argument("--target-update", type=int, default=10_000)
    ap.add_argument("--eps-final", type=float, default=0.02)
    ap.add_argument("--eps-frac", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=100_000)
    ap.add_argument("--eval-deals", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=200_000)
    args = ap.parse_args()

    out = args.out or f"runs/qrdqn_{args.variant}_{args.obs}_{args.seed}"
    os.makedirs(out, exist_ok=True)

    env_kwargs = {
        "variant": args.variant,
        "obs_variant": args.obs,
        "reward_mode": args.reward_mode,
        "frame_stack": args.frame_stack,
    }
    venv = SubprocVecEnv(
        [make_env({**env_kwargs, "seed": args.seed + i}) for i in range(args.n_envs)]
    )

    cls = MaskableQRDQN.build()
    model = cls(
        "MlpPolicy",
        venv,
        policy_kwargs={"net_arch": args.net_arch},
        learning_rate=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer,
        batch_size=args.batch,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        target_update_interval=args.target_update,
        exploration_final_eps=args.eps_final,
        exploration_fraction=args.eps_frac,
        seed=args.seed,
        device=args.device,
        verbose=0,
        tensorboard_log=os.path.join(out, "tb"),
    )

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**vars(args), "git": git_hash()}, f, indent=2)

    metrics_path = os.path.join(out, "metrics.csv")
    csv_file = open(metrics_path, "w", newline="")  # noqa: SIM115
    writer = csv.writer(csv_file)
    writer.writerow(["step", "eval_win_rate", "wall_s"])

    t0 = time.perf_counter()
    eval_seeds = list(range(args.eval_deals))
    done = 0
    while done < args.steps:
        chunk = min(args.eval_every, args.steps - done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        done += chunk
        wr = masked_eval(model, {**env_kwargs, "seed": None}, eval_seeds)
        writer.writerow([done, round(wr, 4), round(time.perf_counter() - t0, 1)])
        csv_file.flush()
        print(f"[{time.perf_counter() - t0:8.1f}s] step={done} eval_win@{len(eval_seeds)}={wr:.3f}")
        if done % args.ckpt_every == 0 or done == args.steps:
            model.save(os.path.join(out, f"qrdqn_{done}"))

    model.save(os.path.join(out, "qrdqn_final"))
    venv.close()
    csv_file.close()
    print(f"done wall={time.perf_counter() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
