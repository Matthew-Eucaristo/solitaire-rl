#!/usr/bin/env python3
"""Train MaskablePPO (sb3-contrib) on Klondike.

    python train_ppo.py --steps 1000000 --variant draw1 --obs pomdp \
        --device cpu --seed 0 --out runs/ppo_draw1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from solitaire_rl.env import KlondikeEnv


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
    wins = 0
    for s in seeds:
        obs, _ = env.reset(seed=s)
        while True:
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, _, term, trunc, info = env.step(int(action))
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
        default="pomdp",
        choices=["pomdp", "perfect", "compact", "compact_perfect"],
    )
    ap.add_argument("--frame-stack", type=int, default=8)
    ap.add_argument("--reward-mode", default="shaped", choices=["shaped", "sparse"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-envs", type=int, default=8)  # parallel env procs
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=100_000)
    ap.add_argument("--eval-deals", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=200_000)
    ap.add_argument(
        "--bc-init",
        default=None,
        help="optional BC checkpoint (runs/bc_*/bc.pt) to warm-start the actor",
    )
    args = ap.parse_args()

    out = args.out or f"runs/ppo_{args.variant}_{args.obs}_{args.seed}"
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

    model = MaskablePPO(
        "MlpPolicy",
        venv,
        policy_kwargs={"net_arch": [256, 256]},
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        seed=args.seed,
        device=args.device,
        verbose=0,
        tensorboard_log=os.path.join(out, "tb"),
    )

    if args.bc_init:
        import torch

        ckpt = torch.load(args.bc_init, map_location=args.device, weights_only=True)
        sd = ckpt["state_dict"]
        pol = model.policy
        with torch.no_grad():
            pol.mlp_extractor.policy_net[0].weight.copy_(sd["0.weight"])
            pol.mlp_extractor.policy_net[0].bias.copy_(sd["0.bias"])
            pol.mlp_extractor.policy_net[2].weight.copy_(sd["2.weight"])
            pol.mlp_extractor.policy_net[2].bias.copy_(sd["2.bias"])
            pol.action_net.weight.copy_(sd["4.weight"])
            pol.action_net.bias.copy_(sd["4.bias"])
        print(f"warm-started actor from {args.bc_init}")

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**vars(args), "git": git_hash()}, f, indent=2)

    metrics_path = os.path.join(out, "metrics.csv")
    csv_file = open(metrics_path, "w", newline="")  # noqa: SIM115
    writer = csv.writer(csv_file)
    writer.writerow(["step", "eval_win_rate", "wall_s"])

    import time

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
            model.save(os.path.join(out, f"ppo_{done}"))

    model.save(os.path.join(out, "ppo_final"))
    venv.close()
    csv_file.close()
    print(f"done wall={time.perf_counter() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
