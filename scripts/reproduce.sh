#!/usr/bin/env bash
# Re-run M0-M3 checks end-to-end on a clean checkout (< ~30 min CPU).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== setup ==="
uv sync --extra dev
PY=.venv/bin/python

echo "=== M0: tests + demo ==="
"$PY" -m pytest -q
"$PY" -m solitaire_rl.demo --episodes 2 --policy heuristic | head -40

echo "=== M1: baselines on 1000 fixed deals ==="
"$PY" eval.py --policy random    --deals 1000 --variant draw1 --jobs 8
"$PY" eval.py --policy heuristic --deals 1000 --variant draw1 --jobs 8
"$PY" eval.py --policy heuristic --deals 1000 --variant draw3 --jobs 8

echo "=== M2: masked DQN (short re-train for reproduce; full run config in runs/dqn_compact/config.json) ==="
"$PY" train_dqn.py --steps 30000 --obs compact --eval-every 15000 --eval-deals 50 --out runs/repro_dqn
"$PY" eval.py --policy dqn:runs/repro_dqn/q_final.pt --deals 200 --variant draw1 --obs compact --jobs 8

echo "=== M3: MaskablePPO (short re-train; full run config in runs/ppo_compact_fresh/config.json) ==="
"$PY" train_ppo.py --steps 100000 --n-envs 8 --obs compact --eval-every 50000 --eval-deals 50 --out runs/repro_ppo
"$PY" eval.py --policy ppo:runs/repro_ppo/ppo_final --deals 200 --variant draw1 --obs compact --jobs 8

echo "=== done ==="
