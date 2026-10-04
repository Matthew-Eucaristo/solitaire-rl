# Rewards

`reward_mode="sparse"`: only the win bonus (`+1`).

`reward_mode="shaped"` (default) — every term toggleable via `reward_config`:

| Term | Default | Applies when |
|------|---------|--------------|
| `win` | +1.0 | episode ends in a win |
| `foundation` | +1/52 ≈ 0.019 | each card moved to a foundation |
| `reveal` | +0.15 | a face-down card is flipped |
| `waste_play` | +0.02 | a waste card is played (tableau or foundation) |
| `idle_penalty` | -0.005 | per move after `idle_after=20` consecutive no-progress moves |
| `recycle_penalty` | -0.02 | each waste→stock redeal |
| `illegal` | -0.1 | an out-of-mask action is submitted (episode ends) |
| `concede` | 0.0 | voluntary give-up |
| `loss` | 0.0 | no_moves / no_progress_* terminations |

"Progress" = a move that flips a face-down card, sends a card to a
foundation, or plays a card from the waste (`MoveOutcome.progress`).

## Rationale

- Foundations are the terminal objective, so the shaped reward is dominated
  by reveal (+0.15) — information gain in a POMDP — with smaller bonuses for
  the other terminal-consistent signals.
- `idle_penalty` and `recycle_penalty` discourage the two degenerate
  strategies a pure maximizer learns first: cycling the stock forever and
  shuffling tableau cards without progress. Termination rules already cap
  this behavior; the penalties make the gradient signal agree with the
  terminator.
- `illegal` ends the episode at -0.1. Agents only see masked actions during
  eval, but the penalty exists for training-time exploration bugs.
