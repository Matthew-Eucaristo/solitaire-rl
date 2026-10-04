# Klondike rules, as implemented

Standard 52-card deck, 7 tableau columns, 4 foundations, stock + waste.

## Deal

Column `i` (0-indexed) gets `i + 1` cards. Exactly the top card of each column
starts face-up; the rest are face-down. Remaining 24 cards form the stock.
Waste and foundations start empty.

## Moves

- **Tableau → tableau**: a face-up card, or an ordered face-up *run*
  (descending ranks alternating colors), may move onto a tableau card of
  **one higher rank and opposite color**. An empty column accepts only a
  **King-led** card or run.
- **→ foundation**: ascending in suit, starting from the Ace.
- **Waste → tableau / foundation**: only the top waste card may be played.
- **Stock → waste**: draw-1 (headline) or draw-3 (variant). When the stock is
  empty, waste is recycled back into the stock in reverse order
  (`stock = waste[::-1]`), which preserves the original draw order.
  Redeals are **unlimited** by default (`max_redeals=None`, configurable).

## Win / loss

- **Win**: all 52 cards on foundations (`GameState.is_win`).
- The environment additionally ends episodes on hopeless conditions that are
  not rules losses but indicate no possible progress:
  `no_moves` (no legal move at all, stock empty),
  `no_progress_cycle` (exact board state repeats — same tableau/waste/foundations/stock),
  `no_progress_idle` (150 consecutive moves without progress), and
  `truncated` at `max_steps` (500). `concede` is a voluntary give-up action.

## Ambiguities resolved

- Redeal order: `stock = waste[::-1]` — cycling the full deck preserves the
  original sequence of draws, matching standard Klondike play.
- Partial-run moves are legal: any face-up card may be moved with its run,
  not only whole columns (standard Microsoft Klondike behavior).
- Draw-3: the player may play only the top waste card; buried waste cards
  become reachable again after a redeal or by playing cards above them.
