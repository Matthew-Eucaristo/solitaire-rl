#!/usr/bin/env python3
"""Regenerate benchmarks/deals.json — seeds 0..999 by default."""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="benchmarks/deals.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {"seeds": list(range(args.start, args.start + args.n)), "variant": "draw1"},
            f,
        )
    print(f"wrote {args.n} seeds to {args.out}")


if __name__ == "__main__":
    main()
