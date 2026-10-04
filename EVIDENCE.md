# Evidence

Exact commands and their real output. Nothing here is a result I did not
observe. All numbers were produced on this machine (MacBook Pro, Apple
silicon, 8 cores) unless stated otherwise.

## M0 — engine + env + tests

```
$ .venv/bin/python -m pytest -q
............................................                          [100%]
44 passed in 12.3s
```

Mask consistency: `tests/test_mask_consistency.py` verifies mask ≡ legal ∪
{CONCEDE} on 10,000 random mid-game states (0.56 s).

```
$ .venv/bin/python -m solitaire_rl.demo --episodes 5 --policy heuristic
(prints 5 fully rendered text games; ep 4 ended in a concede after 155 moves
with 4/52 foundations — see git history for full output)
```

Env throughput: ~16.4k steps/s for env-only stepping under a random policy.

## M1 — baselines on the fixed 1000-deal benchmark (seeds 0..999)

```
$ .venv/bin/python eval.py --policy random    --deals 1000 --variant draw1 --jobs 8 --out results/m1_random_draw1.json
  deals=1000  wins=0    win_rate=0.0000
  avg_moves=32.6   avg_foundations=2.6   avg_wall_ms=2.5
  outcomes={'no_progress_cycle': 1000}

$ .venv/bin/python eval.py --policy heuristic --deals 1000 --variant draw1 --jobs 8 --out results/m1_heuristic_draw1.json
  deals=1000  wins=414  win_rate=0.4140
  avg_moves=230.6  avg_foundations=26.9  avg_wall_ms=36.0
  outcomes={'no_progress_idle': 136, 'concede': 432, 'win': 414, 'truncated': 18}

$ .venv/bin/python eval.py --policy heuristic --deals 1000 --variant draw3 --jobs 8 --out results/m1_heuristic_draw3.json
  deals=1000  wins=118  win_rate=0.1180
  avg_moves=127.9  outcomes={'concede': 765, 'win': 118, 'no_progress_idle': 116, 'truncated': 1}

$ .venv/bin/python eval.py --policy random    --deals 1000 --variant draw3 --jobs 8 --out results/m1_random_draw3.json
  deals=1000  wins=0    win_rate=0.0000
  avg_moves=24.5   outcomes={'no_progress_cycle': 1000}
```

Committed JSONs: `results/m1_*.json`. Total eval wall time ~6 s for all
four runs.

## M2 — masked DQN

Two runs, same code; only `obs_variant` differs.

**one-hot POMDP obs (49,304 dims, frame_stack=8)** — `runs/dqn_draw1`:
0.000 eval win rate through 150k steps (avg foundations plateaued ~7);
killed and superseded by the compact run — root cause recorded in
ANALYSIS.md.

```
$ .venv/bin/python train_dqn.py --steps 250000 --variant draw1 --obs compact \
    --frame-stack 8 --device cpu --seed 0 --out runs/dqn_compact
[  39.9s] step=25000  eps=0.842 eps_done=795  win=0  eval@100=0.000
[  87.5s] step=50000  eps=0.683 eps_done=1540 win=0  eval@100=0.000
[ 132.2s] step=75000  eps=0.525 eps_done=2258 win=0  eval@100=0.000
[ 177.9s] step=100000 eps=0.367 eps_done=2842 win=0  eval@100=0.000
[ 219.6s] step=125000 eps=0.208 eps_done=3301 win=0  eval@100=0.000
[ 266.8s] step=150000 eps=0.050 eps_done=3649 win=1  eval@100=0.000
[ 312.1s] step=175000 eps=0.050 eps_done=3915 win=5  eval@100=0.010
[ 358.6s] step=200000 eps=0.050 eps_done=4173 win=9  eval@100=0.040
[ 400.1s] step=225000 eps=0.050 eps_done=4439 win=9  eval@100=0.000
[ 425.7s] step=250000 eps=0.050 eps_done=4700 win=10 eval@100=0.050
done: episodes=4700 train_wins=10 wall=426s
```

Full-benchmark eval of `q_final.pt` (greedy masked argmax):

```
$ .venv/bin/python eval.py --policy dqn:runs/dqn_compact/q_final.pt \
    --deals 1000 --variant draw1 --obs compact --jobs 8
  deals=1000  wins=58   win_rate=0.0580
  avg_moves=110.5  avg_foundations=13.4  avg_wall_ms=22.3
  outcomes={'no_progress_cycle': 942, 'win': 58}
```

Committed: `results/m2_dqn.json`, `results/m2_dqn_curve.png` (learning
curve from `runs/dqn_compact/metrics.csv`).

A continuation run (`runs/dqn_compact2`, warm-started from `q_final.pt`,
500k more steps at ε 0.1→0.02) is evaluated separately in ANALYSIS.md.

## M3 — MaskablePPO

(pending — see `results/m3_ppo.json` + `runs/ppo_draw1/metrics.csv`)
