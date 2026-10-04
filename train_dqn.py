#!/usr/bin/env python3
"""Train a masked Double-DQN on Klondike.

    python train_dqn.py --steps 200000 --variant draw1 --obs pomdp \
        --device cpu --seed 0 --out runs/dqn_draw1

Checkpoints + metrics.csv land in --out (runs/ is gitignored).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn

from solitaire_rl.actions import ACTION_CONCEDE
from solitaire_rl.dqn import build_q, masked_argmax
from solitaire_rl.env import KlondikeEnv
from solitaire_rl.evaluator import play_episode


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class Replay:
    """Fixed-size replay storing obs/next_obs as float16 (compact enough)."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.cap = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float16)
        self.nobs = np.zeros((capacity, obs_dim), dtype=np.float16)
        self.act = np.zeros(capacity, dtype=np.int32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.mask = np.zeros((capacity, 654), dtype=bool)  # next-state masks
        self.i = 0
        self.n = 0

    def push(self, o, a, r, no, d, nmask) -> None:
        i = self.i
        self.obs[i] = o.astype(np.float16)
        self.nobs[i] = no.astype(np.float16)
        self.act[i] = a
        self.rew[i] = r
        self.done[i] = d
        self.mask[i] = nmask
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch: int, rng: np.random.Generator):
        idx = rng.integers(0, self.n, size=batch)
        return (
            torch.as_tensor(self.obs[idx]).float(),
            torch.as_tensor(self.act[idx]).long(),
            torch.as_tensor(self.rew[idx]),
            torch.as_tensor(self.nobs[idx]).float(),
            torch.as_tensor(self.done[idx]),
            torch.as_tensor(self.mask[idx]),
        )


def evaluate_q(net, obs_dim, env_kwargs, seeds, device, n_max=100):
    """Greedy (eps=0) eval on a handful of seeds — returns win rate."""
    from solitaire_rl.dqn import DQNPolicy

    pol = DQNPolicy(net, obs_dim, device=device)
    wins = 0
    for s in seeds[:n_max]:
        if play_episode(pol, s, env_kwargs).outcome == "win":
            wins += 1
    return wins / min(n_max, len(seeds))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--variant", default="draw1", choices=["draw1", "draw3"])
    ap.add_argument(
        "--obs",
        default="pomdp",
        choices=["pomdp", "perfect", "compact", "compact_perfect", "compact_hint"],
    )
    ap.add_argument("--frame-stack", type=int, default=8)
    ap.add_argument("--reward-mode", default="shaped", choices=["shaped", "sparse"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--buffer", type=int, default=30_000)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay-frac", type=float, default=0.6)
    ap.add_argument("--train-every", type=int, default=4)
    ap.add_argument(
        "--target-every", type=int, default=2_000, help="env steps between target syncs"
    )
    ap.add_argument("--warmup", type=int, default=1_000)
    ap.add_argument(
        "--init", default=None, help="optional checkpoint to warm-start Q (e.g. runs/bc_init/bc.pt)"
    )
    ap.add_argument("--ckpt-every", type=int, default=50_000)
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--eval-deals", type=int, default=100)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = args.out or f"runs/dqn_{args.variant}_{args.obs}_{args.seed}"
    os.makedirs(out, exist_ok=True)

    env_kwargs = {
        "variant": args.variant,
        "obs_variant": args.obs,
        "reward_mode": args.reward_mode,
        "frame_stack": args.frame_stack,
        "seed": args.seed,
    }
    env = KlondikeEnv(**env_kwargs)
    obs_dim = env.observation_space.shape[0]

    q = build_q(obs_dim, args.hidden).to(args.device)
    if args.init:
        ckpt = torch.load(args.init, map_location=args.device, weights_only=True)
        q.load_state_dict(ckpt["state_dict"])
        print(f"warm-started Q from {args.init}")
    qt = build_q(obs_dim, args.hidden).to(args.device)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()
    replay = Replay(args.buffer, obs_dim)

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**vars(args), "git": git_hash(), "obs_dim": obs_dim}, f, indent=2)
    metrics_path = os.path.join(out, "metrics.csv")
    csv_file = open(metrics_path, "w", newline="")  # noqa: SIM115
    writer = csv.writer(csv_file)
    writer.writerow(
        ["step", "episode", "ep_return", "ep_len", "win", "eps", "loss", "eval_win_rate"]
    )

    def eps(step: int) -> float:
        frac = min(1.0, step / max(1, int(args.steps * args.eps_decay_frac)))
        return args.eps_start + (args.eps_end - args.eps_start) * frac

    obs, info = env.reset()
    ep_return = ep_len = wins = episodes = 0
    losses: list[float] = []
    last_target_sync = 0
    eval_seeds = list(range(args.eval_deals))
    eval_kwargs = {**env_kwargs, "seed": None}
    t0 = time.perf_counter()

    for step in range(1, args.steps + 1):
        e = eps(step)
        legal_mask = env.action_masks()
        if rng.random() < e:
            # Explore game moves only: CONCEDE stays in the mask for the
            # greedy policy but is never taken by epsilon exploration —
            # uniformly sampling it would end most episodes in a few steps.
            pool = np.flatnonzero(legal_mask)
            pool = pool[pool != ACTION_CONCEDE]
            action = int(rng.choice(pool))
        else:
            with torch.no_grad():
                qs = q(torch.as_tensor(obs).unsqueeze(0).to(args.device))[0]
            action = masked_argmax(qs, legal_mask)

        nobs, r, term, trunc, info = env.step(action)
        replay.push(obs, action, r, nobs, term or trunc, env.action_masks())
        obs = nobs
        ep_return += r
        ep_len += 1

        if term or trunc:
            if info["outcome"] == "win":
                wins += 1
            episodes += 1
            writer.writerow(
                [
                    step,
                    episodes,
                    round(ep_return, 3),
                    ep_len,
                    int(info["outcome"] == "win"),
                    round(e, 3),
                    round(float(np.mean(losses)) if losses else 0.0, 5),
                    "",
                ]
            )
            obs, info = env.reset()
            ep_return = ep_len = 0
            losses.clear()

        if step >= args.warmup and step % args.train_every == 0:
            b_o, b_a, b_r, b_no, b_d, b_m = replay.sample(args.batch, rng)
            b_o, b_no = b_o.to(args.device), b_no.to(args.device)
            b_a, b_r, b_d = b_a.to(args.device), b_r.to(args.device), b_d.to(args.device)
            qvals = q(b_o).gather(1, b_a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_online = q(b_no)
                next_online[~torch.as_tensor(b_m).to(args.device)] = -1e9
                next_a = next_online.argmax(1)
                next_q = qt(b_no)
                next_q[~torch.as_tensor(b_m).to(args.device)] = -1e9
                tgt = next_q.gather(1, next_a.unsqueeze(1)).squeeze(1)
                target = b_r + args.gamma * (1 - b_d) * tgt
            loss = loss_fn(qvals, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
            losses.append(loss.item())

        if step - last_target_sync >= args.target_every:
            qt.load_state_dict(q.state_dict())
            last_target_sync = step

        if step % args.ckpt_every == 0:
            torch.save(
                {"state_dict": q.state_dict(), "obs_dim": obs_dim, "hidden": args.hidden},
                os.path.join(out, f"q_{step}.pt"),
            )

        if step % args.eval_every == 0:
            wr = evaluate_q(q, obs_dim, eval_kwargs, eval_seeds, args.device)
            print(
                f"[{time.perf_counter() - t0:7.1f}s] step={step} eps={e:.3f} "
                f"eps_done={episodes} win={wins} eval@100={wr:.3f}"
            )
            writer.writerow([step, episodes, "", "", "", round(e, 3), "", wr])
            csv_file.flush()

    torch.save(
        {"state_dict": q.state_dict(), "obs_dim": obs_dim, "hidden": args.hidden},
        os.path.join(out, "q_final.pt"),
    )
    csv_file.close()
    print(f"done: episodes={episodes} train_wins={wins} wall={time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
