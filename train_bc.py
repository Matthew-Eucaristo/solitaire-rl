#!/usr/bin/env python3
"""Behavior-cloning warm-start: pretrain the policy net on heuristic games.

This is documented imitation, not a result claim: the BC checkpoint is a
starting point for RL fine-tuning (train_dqn.py --init / train_ppo.py
--load). The BC policy itself is also evaluated and reported separately.

    .venv/bin/python train_bc.py --games 400 --epochs 3 --out runs/bc_init
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn

from solitaire_rl.env import KlondikeEnv
from solitaire_rl.policies import HeuristicPolicy, load_policy


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def collect(kw: dict, n_games: int, rng: np.random.Generator, teacher: str = "heuristic"):
    """Play teacher games; return (obs_list, action_list) pairs.

    teacher: 'heuristic', a load_policy spec, or 'ens:s1,s2,...' for
    majority-vote ensemble distillation (all voters share kw's obs).
    """
    if teacher == "heuristic":
        pols = [HeuristicPolicy()]
    elif teacher.startswith("ens:"):
        pols = [load_policy(s, kw) for s in teacher[4:].split(",")]
    else:
        pols = [load_policy(teacher, kw)]
    xs, ys = [], []
    env = KlondikeEnv(**kw)
    train_seed_lo = 1_000_000  # never benchmark seeds — same rule as RL training
    for _ in range(n_games):
        obs, _ = env.reset(seed=int(rng.integers(train_seed_lo, 2**31 - 1)))
        while True:
            votes = [p.act(env) for p in pols]
            a = max(set(votes), key=votes.count)
            xs.append(obs.copy())
            ys.append(a)
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument(
        "--activation",
        default="tanh",
        choices=["tanh", "relu"],
        help="activation for the BC net; 'tanh' matches MaskablePPO's policy_net "
        "so --bc-init is a true warm-start (the earlier ReLU BC was the "
        "documented warm-start failure mode).",
    )
    ap.add_argument("--variant", default="draw1", choices=["draw1", "draw3"])
    ap.add_argument(
        "--obs",
        default="pomdp",
        choices=["pomdp", "perfect", "compact", "compact_perfect", "compact_hint"],
    )
    ap.add_argument("--frame-stack", type=int, default=8)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--teacher",
        default="heuristic",
        help="policy to clone: 'heuristic' or any load_policy spec "
        "(e.g. 'ppo:runs/x/ppo_final.zip') — enables iterated amplification "
        "(clone the current champion, re-init PPO, improve again).",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = args.out or f"runs/bc_{args.variant}_{args.seed}"
    os.makedirs(out, exist_ok=True)
    kw = {
        "variant": args.variant,
        "obs_variant": args.obs,
        "frame_stack": args.frame_stack,
        "train_seeds": True,
    }

    t0 = time.perf_counter()
    xs, ys = collect(kw, args.games, rng, args.teacher)
    obs_dim = xs.shape[1]
    print(
        f"collected {len(xs)} (obs,action) pairs from {args.games} games "
        f"({time.perf_counter() - t0:.0f}s)"
    )

    act = nn.Tanh if args.activation == "tanh" else nn.ReLU
    net = nn.Sequential(
        nn.Linear(obs_dim, args.hidden),
        act(),
        nn.Linear(args.hidden, args.hidden),
        act(),
        nn.Linear(args.hidden, 654),
    ).to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    n = len(xs)
    for ep in range(1, args.epochs + 1):
        perm = rng.permutation(n)
        tot = correct = 0
        tot_loss = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            xb = torch.as_tensor(xs[idx]).to(args.device)
            yb = torch.as_tensor(ys[idx]).to(args.device)
            logits = net(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss) * len(idx)
            correct += int((logits.argmax(1) == yb).sum())
            tot += len(idx)
        print(f"epoch {ep}: loss={tot_loss / tot:.4f} acc={correct / tot:.3f}")

    path = os.path.join(out, "bc.pt")
    torch.save(
        {
            "state_dict": net.state_dict(),
            "obs_dim": obs_dim,
            "hidden": args.hidden,
            "kind": "bc",
            "git": git_hash(),
        },
        path,
    )
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(
            {
                **vars(args),
                "git": git_hash(),
                "n_pairs": int(n),
                "wall_s": round(time.perf_counter() - t0, 1),
            },
            f,
            indent=2,
        )
    print(f"saved {path} wall={time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
