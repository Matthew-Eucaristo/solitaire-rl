# Heuristic policy

`HeuristicPolicy` is the deterministic bar the RL agents are compared
against. It scores only mask-legal moves and is fully seeded through the
environment.

## Priority order (first applicable tier wins)

1. **P1 reveal** — any move that flips a face-down card (moving the last
   face-up card off a column).
2. **P2 safe foundation** — send a card to its foundation when the move is
   *safe* (see below).
3. **P3 waste → tableau** — play the waste card onto a tableau column
   (arbitrarily picks the first legal target).
4. **P4 productive tableau→tableau** — the best T2T move by
   `_t2t_score`; a T2T move only counts as productive when it:
   - `run_start == 0`: moves a whole column and thereby exposes the column's
     face-down stack *or* empties it for a King; or
   - `run_start > 0`: unburies cards whose new top either can go to a
     foundation now or can itself be placed on a third tableau column —
     i.e., the move genuinely advances the game instead of shuttling a run.
5. **P5 unsafe foundation** — send a card to its foundation even when unsafe,
   but only once the stock is empty (no more information can arrive).
6. **P6 draw** — turn the stock (or redeal).
7. **P7 leftover** — any remaining legal move (e.g. King into an empty
   column it would only vacate).
8. **P8 concede** — resign.

## The "safe" rule, explicitly

A card of rank `r` may go to its foundation safely when:

- `r <= 2` (Aces and twos can never be needed on the tableau), or
- **both** opposite-color foundations are already at height ≥ `r - 1`
  (the two cards this card could descend onto are already home, so moving it
  up can never strand anything).

## Anti-repeat guard

Before scoring, every candidate move is checked for whether its *post-move*
board signature is already in `env._seen`. Moves that would recreate an
earlier state are skipped; if **every** legal move repeats a past state the
heuristic concedes — this is the "knows when to give up" mechanism, and is
why the heuristic never dies on `no_progress_cycle`.

## Measured (M1, 1000 fixed deals, seeds 0..999)

| Variant | Win rate | Typical failure |
|---------|----------|-----------------|
| draw1 | **0.414** | concede 432, no_progress_idle 136, truncated 18 |
| draw3 | **0.118** | concede 765, no_progress_idle 116 |
