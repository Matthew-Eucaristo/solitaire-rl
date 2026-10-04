# Committed checkpoints (best run of each policy)

| File | Policy | Benchmark (1000 deals, draw-1) |
|------|--------|-------------------------------|
| `dqn_compact_250k.pt` | masked DQN, compact obs | 0.058 |
| `ppo_compact_6m.zip` | MaskablePPO, compact POMDP | 0.118 |
| `ppo_perfect_6m.zip` | MaskablePPO, compact perfect-info | 0.177 |

Load via `eval.py --policy dqn:checkpoints/dqn_compact_250k.pt --obs compact`
or `ppo:checkpoints/ppo_compact_6m.zip --obs compact` /
`ppo:checkpoints/ppo_perfect_6m.zip --obs compact_perfect`.
