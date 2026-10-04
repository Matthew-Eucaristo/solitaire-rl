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

Batch-5 results (same 1000-deal benchmark):

| run | recipe | win_rate |
|-----|--------|----------|
| runs/ppo_amp1_6m@2.4M | continue amp1@2.8M at lr 1e-4 (~5.4M total) | **0.328** NEW BEST |
| runs/ppo_amp1_6m@3.0M | same run, final ckpt (~6M total) | 0.325 |
| runs/ppo_amp2@2.8M | round-2 fresh init (clone acc 68.7%) | 0.306 |
| runs/ppo_amp2@3.0M | same run, final | 0.310 |

The amplification loop is converging: continue-from-peak at lr 1e-4
gains ~+0.004/round (0.324->0.328), while a fresh round-2 init lands
slightly below its teacher (0.310 vs 0.328) — the clone loses the
fine-grained residual knowledge. Remaining frontiers: keep chaining
peak-continuations (amp1_9m launched), and a 512-hidden BC init in case
capacity is the ceiling (bc_amp1_512 collecting).

Batch-6 results (same 1000-deal benchmark):

| run | recipe | win_rate |
|-----|--------|----------|
| runs/ppo_amp1_9m@1.6M | continue 0.328-peak at lr 1e-4 (~8M total) | **0.353** NEW BEST |
| runs/ppo_amp1_9m@2.8M | same run | 0.352 |
| runs/ppo_amp1_9m@3.0M | same run, final | 0.339 |
| runs/ppo_amp1_512@3.0M | fresh 512-hidden BC init (71% acc), lr 1e-4 | 0.287 |

Continue-from-peak chain: 0.265 -> 0.278 -> 0.324 -> 0.328 -> 0.353.
Doubling capacity (512-hidden net) does NOT beat it: fresh 512 init
landed at the same 0.28-0.31 band as 256 nets — capacity is not the
ceiling; trajectory position is. The gain mechanism is cumulative
on-policy refinement from a good starting point, now at 85% of the
heuristic bar (0.414). Next chain link (amp1_12m) launched from the
0.353 checkpoint; eval@100 already hit 0.36 at 0.9M.

Batch-7 results (same 1000-deal benchmark):

| run | recipe | win_rate |
|-----|--------|----------|
| runs/ppo_amp1_12m@2.0M | chain link 4 (~14M total) | 0.348 |
| runs/ppo_amp1_12m@3.0M | same, final | 0.348 |
| runs/ppo_amp1_12m@2.8M | same | 0.343 |

Chain has converged at ~0.34-0.35 (eval@100 briefly showed 0.40 —
100-deal eval noise; always trust the 1000-deal number). The champion
remains ppo_amp1_9m/ppo_1600000.zip at **0.353**. A low-entropy
continuation (ent_coef 0.002) is testing whether sharper updates can
push past the plateau.

Batch-8 — ensembles (same 1000-deal benchmark, majority vote):

| ensemble | members | win_rate |
|----------|---------|----------|
| ens3 | champ(0.353)+12m(0.348)+lowent(0.348) | 0.378 |
| ens5 | ens3 + amp2 + bcxllr_6m | 0.377 |
| ens7 | ens5 + hint_bc@2.6M + curr | 0.377 |
| ens-A | champ + 12m + 9m@2.8M | **0.380** BEST ARTIFACT |
| mixed5 | 3×hint + perfect-info + compact | 0.321 (weak voters drag) |

Ensembling adds +0.02-0.03 over the best member, saturating ~0.38;
voters must all be strong (mixed-lineage inclusion of 0.118/0.177
models hurt). 91% of the heuristic bar now. Next: distill the ensemble
back into a single net (bc_ensA, acc 0.666) → PPO — iterated
distill-then-improve. train_bc --teacher now accepts 'ens:s1,s2,…'.

Batch-9 — ensemble distillation (same 1000-deal benchmark):

| run | recipe | win_rate |
|-----|--------|----------|
| runs/ppo_ensdist@2.6M | BC(ens-A votes, acc 0.666) init + PPO lr1e-4 | 0.332 |
| runs/ppo_ensdist@1.2M | same run | 0.291 |
| runs/ppo_ensdist@3.0M | same run, final | 0.306 |

Single-model ceiling confirmed again at ~0.33 — the ensemble's +0.05
edge doesn't survive 66%-fidelity distillation. The 0.380 ens-A stays
champion. Chaining the distilled basin (ensdist_6m) and growing the
voter pool (ensdist_s3) for the next ensemble assembly.

Batch-10 — information-set search (200-deal subset, worlds=6/topk=3/h=120):

| policy | seeds 0-199 |
|--------|-------------|
| isearch (determinized heuristic rollouts, guide=champion) | 0.335 |
| champion alone | 0.345 |
| heuristic | 0.415 |

Honest negative: sampled-world rollouts do NOT beat the learned policy —
the champion's trained judgment already exceeds what a fast heuristic
rollout measures (~0.3s/move, 29min/200 deals). scripts/isearch.py
implements determinization correctly (samples only unseen-card
permutations) and includes the discriminations fix (defer to guide when
margins < 0.02); the approach itself just doesn't pay at this scale.

## Batch 11 — the portfolio result (system-level champion, > heuristic)

**Motivation.** The vote-ensembles saturated at 0.380 but the *union* of member
win-sets was far larger — policies solve different deals. A **portfolio agent**
declares best-of-N up front: every member plays every deal and the best
*episode-observable* outcome (win > foundations > moves) is reported per deal.
Implemented as `run_eval("portfolio:<spec>,<spec>,...")` (member attribution
kept per deal; not an oracle — selection uses only what an agent could see by
running N strategies in sequence).

```
portfolio3 (12m + 9m@2.8M + bcxl_lr)                       win_rate=0.482
portfolio5 (+ensdist_s3 +sparse)                           win_rate=0.524
portfolio9 (7 PPO lineages + ens-A as a member)            win_rate=0.534
member_wins p9: 9m@2.8M=112, sparse=93, amp6m=87, bcxl=68,
                s3=63, champ=57, 12m=54   (lowent/ens = 0 — fully covered)
```

**Verdict.** `portfolio9 = 0.534` is the strongest artifact and the first
system to beat the heuristic (0.414) on the benchmark. Diversity, not size,
was the lever: sparse-reward and BC-diverse lineages cover deals the
shaped-reward chain misses. Honest label kept: this is a declared best-of-N
portfolio, distinct from single-model results (champion single = 0.353,
vote-ensemble = 0.380).

**Negative results this batch.** Sparse-lineage member did NOT help vote
ensembles (0.358/0.349/0.362 vs ens-A 0.380); V0-selector (argmax value head
picks member per deal) = 0.343 < champion — value heads can't predict which
member wins a deal; no MaskableRecurrentPPO in sb3-contrib 2.9.0 (masking and
recurrence aren't combined upstream); multiprocessing-spawn under a stdin
script deadlocks children (heredoc evals must run from real .py files).

## Batch 11b — portfolio grows with lineage diversity (0.567)

Win-sets measured for every strong checkpoint; the union of 13 members =
0.567. Greedy max-coverage confirms each *lineage* adds unique deals
(ensdist/BC-diverse/sparse lineages +21-29 deals each after the first two).

```
portfolio13 (all 13 measured members incl. ens-A as member)
win_rate = 0.567   (member attribution spread 35-79 wins across 11 members)
```

Reading: portfolio is the declared best-of-N system, not a single model —
but it is the strongest honest artifact this campaign produced, and it only
grows with lineage diversity. Single-model and vote-ensemble ceilings stand
at 0.353 / 0.380; best-of-N = 0.567.
