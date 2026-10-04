#!/usr/bin/env python3
"""Plot learning curves from a run's metrics.csv into results/<name>_curve.png.

    .venv/bin/python scripts/plot_metrics.py runs/dqn_draw1 results/m2_dqn_curve.png
"""

from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_csv")
    ap.add_argument("out_png")
    ap.add_argument("--ma", type=int, default=200, help="moving-average window (episodes)")
    args = ap.parse_args()

    steps, ep_ret, ep_len, win, eval_steps, eval_wr = [], [], [], [], [], []
    with open(args.metrics_csv) as f:
        for row in csv.DictReader(f):
            if row["ep_len"]:
                steps.append(int(row["step"]))
                ep_ret.append(float(row["ep_return"]))
                ep_len.append(int(row["ep_len"]))
                win.append(int(row["win"]))
            if row["eval_win_rate"]:
                eval_steps.append(int(row["step"]))
                eval_wr.append(float(row["eval_win_rate"]))

    def ma(xs: list[float], w: int) -> list[float]:
        out = []
        for i in range(len(xs)):
            lo = max(0, i - w + 1)
            out.append(sum(xs[lo : i + 1]) / (i - lo + 1))
        return out

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(steps, ma(ep_ret, args.ma), lw=1)
    axes[0].set_title("episode return (moving avg)")
    axes[0].set_xlabel("env steps")
    axes[1].plot(steps, ma(ep_len, args.ma), lw=1, color="darkorange")
    axes[1].set_title("episode length (moving avg)")
    axes[1].set_xlabel("env steps")
    axes[2].plot(steps, ma(win, args.ma), lw=1, color="seagreen", label="train win rate")
    axes[2].plot(eval_steps, eval_wr, "o-", color="black", label="greedy eval win rate")
    axes[2].set_title("win rate")
    axes[2].set_xlabel("env steps")
    axes[2].set_ylim(-0.02, 1.0)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=140)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
