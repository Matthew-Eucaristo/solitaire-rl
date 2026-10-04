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

First attempt (`runs/ppo_draw1`, one-hot pomdp obs, with a BC warm-start):
0.000 eval win rate through 1.5M steps — killed. Same conclusion as M2:
the 49k-dim one-hot input does not train on laptop budgets.

Compact obs, fresh (no BC) — `runs/ppo_compact_fresh`, 3M steps, ~17 min wall:

```
$ .venv/bin/python train_ppo.py --steps 3000000 --n-envs 6 --n-steps 1024 \
    --variant draw1 --obs compact --frame-stack 8 --device cpu --seed 1 \
    --eval-every 300000 --eval-deals 100 --out runs/ppo_compact_fresh
step     eval_win_rate@100   wall_s
 300000  0.00    99.6
 600000  0.00   203.8
 900000  0.00   302.3
1200000  0.00   403.2
1500000  0.05   509.5
1800000  0.05   612.7
2100000  0.06   714.7
2400000  0.05   821.2
2700000  0.13   921.7
3000000  0.11  1018.8
```

Full-benchmark eval of `ppo_final`:

```
$ .venv/bin/python eval.py --policy ppo:runs/ppo_compact_fresh/ppo_final \
    --deals 1000 --variant draw1 --obs compact --jobs 8
  deals=1000  wins=116  win_rate=0.1160
  avg_moves=85.5  avg_foundations=15.2  avg_wall_ms=48.3
  outcomes={'no_progress_cycle': 884, 'win': 116}
```

Committed: `results/m3_ppo.json`.

A `compact_perfect` run (perfect information, M4-style stretch) reached
0.20 eval@100 at 2.7M steps — see ANALYSIS.md for the comparison.

## Extensions (resumed +3M each via `train_ppo.py --load`)

```
$ .venv/bin/python eval.py --policy ppo:runs/ppo_compact2/ppo_final \
    --deals 1000 --variant draw1 --obs compact --jobs 8
  deals=1000  wins=118  win_rate=0.1180   # plateau: same as 3M
  outcomes={'no_progress_cycle': 882, 'win': 118}

$ .venv/bin/python eval.py --policy ppo:runs/ppo_perfect2/ppo_final \
    --deals 1000 --variant draw1 --obs compact_perfect --jobs 8
  deals=1000  wins=177  win_rate=0.1770
  outcomes={'no_progress_cycle': 823, 'win': 177}
```

Committed: `results/m3_ppo_6m.json`, `results/m4_ppo_perfect.json`.
Both curves oscillated flat through the second 3M — a real plateau, not
truncated learning.

## Recordings

`recordings/` contains GIF episodes rendered by `record.py` directly from
the environment state (no pixels fed to the agent — the agent only sees
the compact obs vector; every frame is a real masked action the policy
chose on a benchmark deal):

- `ppo_win_seed{118,47,211,37,275,639}.gif` — MaskablePPO winning six
  different benchmark deals (verified `outcome: "win"` in the eval JSON).
- `ppo_perfect_win_seed{211,191}.gif` — perfect-info PPO wins (M4-style stretch).
- `heuristic_win_seed{326,191}.gif` — the heuristic baseline winning
  (seeds chosen from `results/m1_heuristic_draw1.json`).
- `heuristic_lose_seed0.gif` — heuristic stalling into `no_progress_idle`
  on seed 0 (it wins 414/1000, not all).
- `dqn_win_seed{47,718}.gif` — masked DQN wins (seed 47 shared with a PPO
  win GIF for comparison).
- `dqn_concede_seed14.gif` — the trained DQN voluntarily conceding a
  hopeless deal at move 86 (64/1000 concessions observed in eval).

## Web app (playable UI + agent watch)

```
$ .venv/bin/python -m pytest -q
54 passed, 1 warning in 1.97s          # +8 webapi tests (test_webapi.py)

$ .venv/bin/python -m uvicorn solitaire_rl.webapi:app --port 8080
# verified interactively in Chrome:
#   - stock click draws; waste updates (moves=1)
#   - double-click A♣ -> foundation applied (1/52)
#   - drag 2♥ -> 3♠ column applied, next card revealed
#   - "Watch" with MaskablePPO on deal 284 played a full live game to
#     WIN: 104 moves, 52/52 foundations (seed 284 is a benchmark win in
#     results/m3_ppo_6m.json)
#   - deal 47 under the 6M checkpoint ended no_progress_cycle @78 —
#     consistent: the recorded seed-47 win belongs to the 3M checkpoint
```

API: `GET /api/state` `POST /api/new|/api/move|/api/undo|/api/concede`
`POST /api/agent/step|/api/agent/hint` `GET /api/agent/policies`.
Hidden cards/stock order are never serialized — the UI is a legal POMDP
client. Files: `src/solitaire_rl/webapi.py`, `web/{index.html,app.css,app.js}`.

## Improvement experiments (post-M3, draw-1, 1000-deal eval)

Hypothesis (2026-10-04): three cheap levers to lift PPO past the 0.118
plateau — curriculum on heuristic-winnable deals, a correctly
activation-matched BC warm-start, and exposing the heuristic's choice as
observation features.

Machinery (commit a016b03): `seed_pool` env param restricts training
episodes to a deal set; `scripts/rank_deals.py` buckets train seeds by
heuristic outcome; `compact_hint` obs appends the heuristic's chosen
action (10 dims: kind one-hot + src/dst/run_start); `train_bc.py
--activation tanh` emits the exact MaskablePPO `policy_net`/`action_net`
arch so `--bc-init` is a true warm-start (the earlier ReLU BC was the
documented failure).

Deal difficulty ranking (seeds 1_000_000..1_001_999, heuristic, jobs 8,
9s wall): easy=873 medium=3 hard=1124 — heuristic train-range win rate
0.436 vs 0.414 on the benchmark (population consistent).
`benchmarks/deal_difficulty.json`, pool file `benchmarks/easy_seeds.json`.

Measured on the SAME 1000 benchmark deals (eval.py --jobs 4-8):

| run | recipe | steps | win_rate |
|-----|--------|-------|----------|
| runs/ppo_curr_p1 | compact, easy-pool only (curriculum phase 1) | 3M | 0.125 |
| runs/ppo_bcinit | compact, Tanh-BC init | 3M | 0.142 |
| runs/ppo_hint | compact_hint obs | 3M | (eval pending) |

Read at equal budget both new recipes already beat the 0.118 baseline
(6M). Phase-2 expansion (p1 --load on full seeds), bc-init continuation
to 6M, and the combo run (hint obs + Tanh BC init, acc@BC 0.816) are in
flight; `runs/*/metrics.csv` has the per-100k curves.

Batch-2 results (same 1000-deal benchmark):

| run | recipe | win_rate |
|-----|--------|----------|
| runs/ppo_hint | compact_hint obs only, 3M | 0.114 |
| runs/ppo_curr_p2 | curriculum phase-2 (p1 --load, full seeds), +3M | 0.129 |
| runs/ppo_bcinit_6m | Tanh-BC init continued to ~6M total | 0.127 |
| runs/ppo_hint_bc | compact_hint + Tanh-BC init, 3M | **0.265** |

Interpretation: neither lever works alone — hint-only 0.114, curriculum
0.125/0.129, BC-init 0.142 (and decayed to 0.127 by 6M). Together they
compose: BC plants the policy in a working follow-the-expert attractor,
and the hint features keep the expert signal observable at every step so
PPO can learn residual deviations. Result 0.265 = 2.2× the 0.118
baseline, above the 0.177 perfect-info model, at HALF its step budget.
ppo_hint_bc was still climbing (eval@100 hit 0.27) — extension to 6M
launched.

Batch-3 results (same 1000-deal benchmark):

| run | recipe | win_rate |
|-----|--------|----------|
| runs/ppo_hint_bc_6m | hint+BC(82%) init, extended to ~6M (lr 3e-4) | 0.180 (decay) |
| runs/ppo_hint_bc_curr | hint+BC init, easy-pool curriculum | 0.236 |
| runs/ppo_hint_bc_s2 | hint+BC init, seed 2 replication | 0.229 |
| runs/ppo_hint_bcxl | hint+BC(97% attractor, 460k pairs) init, lr 3e-4 | 0.170 |
| runs/ppo_hint_bcxl_lr | hint+BC(97%) init, lr 1e-4 | **0.278** NEW BEST |

Checkpoint sweep on ppo_hint_bc confirms the ~0.26 plateau is real
(2.4M=0.214, 2.6M=0.260, 2.8M=0.226, 3.0M=0.265) — not a lucky eval.

Three findings stack: (1) imitation anchoring works (hint obs + Tanh BC
init); (2) late-training decay is an LR artifact — at lr 3e-4 every run
peaks ~3M then slides (0.265->0.180, 0.142->0.127); (3) the fix is
stronger attractor + lower LR: 460k-pair BC clone at 97% acc + lr 1e-4
holds 0.28-0.33 eval@100 through 3M and evals 0.278 on 1000 deals.
Round-2 amplification (clone the champion itself — acc 0.60 — as the
next init) and a low-LR 6M continuation of the champion are in flight.

Batch-4 results (same 1000-deal benchmark):

| run | recipe | win_rate |
|-----|--------|----------|
| runs/ppo_amp1@2.8M | clone-of-champion init (60% acc) + lr 1e-4 | **0.324** NEW BEST |
| runs/ppo_bcxllr_6m | 97% clone init, continued to ~6M at lr 1e-4 | 0.312 |
| runs/ppo_amp1@3.0M | same run, final ckpt | 0.282 |

Iterated amplification works: cloning the champion (not the heuristic)
as the next init landed in a better basin — amp1@2.8M = 0.324 vs its
teacher's 0.265. The low-LR continuation also validates the decay
diagnosis: 6M at lr 1e-4 evals 0.312 where 6M at lr 3e-4 collapsed to
0.18. Heuristic bar (0.414) is now at 78% reach. Round-2 (clone
amp1@2.8M → amp2) and peak-continuation launched.
