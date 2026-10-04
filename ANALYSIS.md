# Analysis

What helped, what didn't, and what the numbers mean. Numbers are filled in
from `results/*.json` — every claim traces to `EVIDENCE.md`.

## Headline results (1000 fixed deals, seeds 0–999)

| Policy | draw-1 win rate | draw-3 win rate | avg moves |
|--------|-----------------|-----------------|-----------|
| random_legal | 0.000 | 0.000 | ~25–33 |
| heuristic | 0.414 | 0.118 | ~231 / ~128 |
| masked DQN | TBD | — | — |
| MaskablePPO | TBD | — | — |

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

- **raw ε-greedy including CONCEDE**: sampling resignations ended episodes
  in ~3.6 steps; excluding CONCEDE from *exploration* (keeping it in the
  mask) restored ~33-step episodes. Documented in train_dqn.py and D17.
- (filled in as training completes)

## draw-1 vs draw-3

(placeholder — filled with M2/M3 numbers)
