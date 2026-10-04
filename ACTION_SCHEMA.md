# Action schema — Discrete(654)

Fixed-size action space. Illegal actions are always masked (never filtered
from the space); `env.action_masks()` returns the exact legal set plus
`CONCEDE`, which is always legal.

| Range      | Count | Meaning |
|------------|-------|---------|
| 0          | 1     | `DRAW` — draw from stock, or redeal when stock is empty |
| 1–7        | 7     | waste → tableau column 0–6 |
| 8          | 1     | waste → foundation |
| 9–15       | 7     | tableau top of column `a-9` → foundation |
| 16–652     | 637   | tableau → tableau: `a = 16 + src*91 + dst*13 + j`, where `j` is the index of the moved card in the source column's `up` list (`j=0` moves the whole run) |
| 653        | 1     | `CONCEDE` — give up the deal; always legal |

## Invariants (tested on 10k random states)

- `action_masks()` ≡ legal moves ∪ {CONCEDE}: every legal move has a mask
  entry, every mask entry decodes to a legal move, and `apply_move` succeeds
  on the decoded move.
- Indices decode unambiguously: `src ∈ 0..6`, `dst ∈ 0..6`, `j ∈ 0..12`
  (`MAX_UP = 13`; a tableau column can never hold more than 13 face-up cards
  in a legal game — 7 original cards can grow only by stacking alternating
  runs, and the run itself is capped by deck size).
- `CONCEDE` is never produced by `legal_moves()` — it is added to the mask
  on top, so the agent can always resign even from a dead position.
