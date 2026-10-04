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

(pending — see `results/m2_dqn.json` + `runs/dqn_draw1/metrics.csv`)

## M3 — MaskablePPO

(pending — see `results/m3_ppo.json` + `runs/ppo_draw1/metrics.csv`)
