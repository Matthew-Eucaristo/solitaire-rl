# solitaire-rl

Reinforcement-learning agents for **Klondike Solitaire** — state vectors,
legal-move masking, fixed-deal evaluation. No pixels. Built as a
reproducible research artifact; see `DECISIONS.md` for every default that
was chosen and why.

## Setup

```bash
uv sync --extra dev
```

## Quick look

```bash
.venv/bin/python -m solitaire_rl.demo --episodes 5 --policy heuristic   # text games
.venv/bin/python -m pytest -q                                          # 44 tests
```

## Evaluate a policy on the 1000-deal benchmark

```bash
.venv/bin/python eval.py --policy heuristic --deals 1000 --variant draw1 --jobs 8
.venv/bin/python eval.py --policy random    --deals 1000 --variant draw3
.venv/bin/python eval.py --policy dqn:runs/dqn_draw1/q_final.pt
.venv/bin/python eval.py --policy ppo:runs/ppo_draw1/ppo_final
```

Benchmark deals are `benchmarks/deals.json` (seeds 0–999). Training deals
are always seeds ≥ 1,000,000 — the two sets are disjoint by construction.

## Results (fixed benchmark, 1000 deals each)

| Policy | obs | draw-1 win rate | draw-3 win rate |
|--------|-----|-----------------|-----------------|
| random_legal | — | 0.000 | 0.000 |
| heuristic | — | **0.414** | **0.118** |
| masked DQN (250k steps) | compact | 0.058 | — |
| MaskablePPO (3M steps) | compact | 0.116 | — |
| MaskablePPO (3M steps) | compact_perfect | ~0.16–0.20 | — |

Dashes fill in as M2/M3 complete; per-run JSONs live in `results/`.

## Layout

```
src/solitaire_rl/   engine + env + policies + recording
  cards.py state.py moves.py        rules engine
  actions.py obs.py env.py          Discrete(654) schema, obs encoder, Gymnasium env
  policies.py evaluator.py render.py record.py demo.py
train_dqn.py train_ppo.py eval.py   entry points
scripts/gen_benchmarks.py           regenerates benchmarks/deals.json
tests/                              44 tests incl. 10k-state mask check
runs/                               training outputs (gitignored)
results/                            committed eval JSONs + plots
recordings/                         GIF episodes of trained agents
```

Docs: `RULES.md` · `ACTION_SCHEMA.md` · `REWARDS.md` · `HEURISTIC.md` ·
`DECISIONS.md` · `EVIDENCE.md` · `REFERENCES.md` · `ANALYSIS.md`
