# Committed checkpoints (best run of each policy)

| File | Policy | Benchmark (1000 deals, draw-1) |
|------|--------|-------------------------------|
| `dqn_compact_250k.pt` | masked DQN, compact obs | 0.058 |
| `ppo_compact_6m.zip` | MaskablePPO, compact POMDP | 0.118 |
| `ppo_perfect_6m.zip` | MaskablePPO, compact perfect-info | 0.177 |
| `ppo_amp1_9m_1p6m.zip` | amplification chain champion | 0.353 |
| `ppo_amp1_9m_2p8m.zip` | amplification chain member | 0.352 |
| `ppo_amp1_12m.zip` | amplification chain member | 0.348 |
| `ppo_divC.zip` | heuristic-clone lineage (seed 13) | 0.295 |
| `ppo_divD.zip` | ensA-clone lineage (seed 17) | 0.315 |
| `ppo_sparse.zip` | sparse-reward lineage | 0.289 |
| `ppo_amp512.zip` | 512×2-arch lineage | 0.287 |
| `ppo_bcxl.zip` | BC-XL clone lineage | 0.278 |
| `ppo_bcxllr6m.zip` | BC-XL 6M lineage | 0.312 |
| `ppo_hintbc.zip` | hint-obs + BC lineage | 0.265 |
| `qrdqn_400000.pt` | MaskableQRDQN, hint obs @400k (fp16-packed, target net dropped) | 0.067 |

Load a single model via
`eval.py --policy ppo:checkpoints/ppo_amp1_9m_1p6m.zip --obs compact_hint`
(`--obs compact` / `compact_perfect` for the original three).

## Canonical portfolio (best-of-N, declared) — 0.579

```bash
.venv/bin/python eval.py --obs compact_hint --deals 1000 --jobs 4 --policy \
  'portfolio:ppo:checkpoints/ppo_amp1_9m_1p6m.zip,ppo:checkpoints/ppo_divC.zip,ppo:checkpoints/ppo_divD.zip,ppo:checkpoints/ppo_sparse.zip,ppo:checkpoints/ppo_amp512.zip,ppo:checkpoints/ppo_bcxl.zip,ppo:checkpoints/ppo_bcxllr6m.zip,ppo:checkpoints/ppo_hintbc.zip,qrdqn:checkpoints/qrdqn_400000.pt'
```

Every member plays every deal; the best episode-observable outcome (win >
foundations > moves) is reported per deal — a declared best-of-N system.
0.579 measured: 571 canonical-8 wins + 8 net deals from the QRDQN member
(the fp16-packed member wins 64 solo, including wins fp32 missed). The full 25-member pure-RL portfolio (members
under the gitignored `runs/`) reached 0.620 (0.625 including QRDQN); adding
`heuristic` as a member reaches 0.640 (0.645 with QRDQN).
