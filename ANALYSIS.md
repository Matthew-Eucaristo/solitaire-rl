# Analysis

What helped, what didn't, and what the numbers mean. Numbers are filled in
from `results/*.json` — every claim traces to `EVIDENCE.md`.

## Headline results (1000 fixed deals, seeds 0–999)

| Policy | obs | draw-1 win rate | draw-3 win rate | avg moves |
|--------|-----|-----------------|-----------------|-----------|
| random_legal | — | 0.000 | 0.000 | ~25–33 |
| heuristic | — | 0.414 | 0.118 | ~231 / ~128 |
| masked DQN (250k) | compact | 0.058 | — | ~110 |
| MaskablePPO (3M) | compact | 0.116 | — | ~85 |
| MaskablePPO (6M) | compact | 0.118 | — | ~86 |
| MaskablePPO (6M) | compact_perfect | 0.177 | — | ~90 |

All RL numbers are greedy masked evaluations on the fixed benchmark; the
training curves in `runs/*/metrics.csv` are reproduced in
`results/m2_dqn_curve.png`.

## Why Klondike is hard for RL here

- **Sparse terminal signal**: a win requires ~150–250 coordinated moves;
  `random_legal` wins 0/1000 and dies to `no_progress_cycle` after ~32
  moves. A policy that merely "plays well" still loses — foundations are
  the only real scoreboard.
- **POMDP structure**: the stock order and face-down cards are hidden;
  the frame stack (k=8) gives the agent a short memory of what it has
  already seen cycle past.
- **Cycle traps**: without no-progress termination, naive policies
  (including an early version of our own heuristic) die ping-ponging a
  run between two columns forever. Both the terminator and the heuristic's
  anti-repeat filter exist because this failure mode is the default, not
  the exception.

## What helped

- **Action masking** is doing the heavy lifting everywhere: the policy can
  never take an illegal move, so all learning capacity goes to sequencing,
  not legality.
- **Reward shaping** toward reveals (+0.15) gave the DQN an early,
  dense learning signal — see EVIDENCE for avg-foundations vs random.
- **Anti-repeat guard in the heuristic** turned a 0/100 heuristic into a
  41% one (see HEURISTIC.md). The equivalent effect for learned agents is
  the cycle-detection terminator: hopeless loops are losses, so the value
  function learns to avoid them instead of looping forever.

## What didn't (or couldn't)

- **The 49k-dim one-hot obs could not be trained here**: DQN plateaued at
  0.000 eval (avg foundations ~7) through 150k+ steps; PPO stayed at 0.000
  through 1.5M. Same code, same budgets, only `obs_variant` changed —
  the compact per-slot code (132 dims) immediately produced wins. Encoding
  choice dominated every other hyperparameter we touched.
- **raw ε-greedy including CONCEDE**: sampling resignations ended episodes
  in ~3.6 steps; excluding CONCEDE from *exploration* (keeping it in the
  mask) restored ~33-step episodes.
- **BC warm-start did not transfer**: a 2×256 MLP only reached 55–60%
  action-match on heuristic demonstrations, and the imperfect imitation
  drifted into immediate cycles (0/100 wins). Grafted into PPO it made no
  difference vs fresh (fresh actually learned, grafted did not) —
  ReLU-trained weights transplanted into a Tanh actor was the likely
  break. Reported for completeness; fresh compact PPO is the M3 result.
- **Win rates plateau ~10–20%**: losses are overwhelmingly
  `no_progress_cycle` — agents play competently but cannot plan out of
  recurring board patterns. Value networks are not search; that is the
  structural argument for M4 (perfect-info search).

## draw-1 vs draw-3

- Heuristic: 0.414 (draw-1) vs 0.118 (draw-3) — draw-3 is ~3.5× harder,
  consistent with the waste being throttled.
- RL training focused on draw-1 (the headline variant). Given the draw-3
  heuristic bar is already 3.5× lower, draw-3 RL runs are a documented
  extension rather than a blocker.

## The "knows when to give up" behavior

Two mechanisms, both measured:

1. `CONCEDE` is always in the mask; the trained DQN uses it (64/1000
   deals in the continuation eval — `results/m2_dqn_cont.json`), choosing
   to resign hopeless deals rather than cycling to the terminator. See
   `recordings/dqn_concede_seed14.gif`.
2. The environment *itself* ends hopeless loops (`no_progress_cycle`,
   `no_progress_idle`, `no_moves`) — so every loss is still bounded and
   honestly reported instead of a policy churning forever.
