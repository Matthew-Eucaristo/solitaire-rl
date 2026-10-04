"""Legal-move generation and move application for Klondike.

Move legality (see RULES.md):
  * Tableau -> tableau: a card or an ordered run of face-up cards may move
    onto a card of one higher rank and opposite color. An empty column only
    accepts a King-led run.
  * -> foundation: ascending in suit, starting from Ace.
  * Stock -> waste: draw `draw_size` cards (only the waste top is playable).
    When the stock is empty the waste flips back (redeal), subject to
    `max_redeals` (None = unlimited).
  * Waste top -> tableau/foundation.

`legal_moves` enumerates every currently legal *game* move. CONCEDE is an
agent-level action added by the environment's action mask — it is not a game
move and never appears here.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from solitaire_rl.cards import Card, other_color
from solitaire_rl.state import GameState


class MoveKind(enum.IntEnum):
    DRAW = 0
    WASTE_TO_TABLEAU = 1
    WASTE_TO_FOUNDATION = 2
    TABLEAU_TO_FOUNDATION = 3
    TABLEAU_TO_TABLEAU = 4


@dataclass(frozen=True, slots=True)
class Move:
    """A legal game move.

    src/dst are tableau column indices where relevant. `run_start` is the
    index into the source column's `up` list: the moved run is
    `up[run_start:]` (0 = whole face-up pile).
    """

    kind: MoveKind
    src: int = -1
    dst: int = -1
    run_start: int = -1


@dataclass(slots=True)
class MoveOutcome:
    """What an applied move changed — drives rewards and progress tracking."""

    revealed: bool = False  # a face-down card was flipped
    to_foundation: bool = False
    from_waste: bool = False  # a waste card entered the tableau/foundations
    recycled: bool = False
    drew: bool = False
    emptied_column: bool = False

    @property
    def progress(self) -> bool:
        """Events that count as real progress (resets the idle counter)."""
        return self.revealed or self.to_foundation or self.from_waste


def _can_place_on_tableau(card: Card, col_has_cards: bool, top: Card | None) -> bool:
    if not col_has_cards:
        return card.rank == 13  # empty column: King only
    assert top is not None
    return top.rank == card.rank + 1 and other_color(top, card)


def legal_moves(state: GameState) -> list[Move]:
    """All legal game moves in `state`, in a fixed enumeration order."""
    moves: list[Move] = []

    # Draw / redeal.
    can_redeal = state.waste and (state.max_redeals is None or state.redeals < state.max_redeals)
    if state.stock or can_redeal:
        moves.append(Move(MoveKind.DRAW))

    # Waste plays.
    if state.waste:
        w = state.waste[-1]
        if w.rank == state.foundation_height(w.suit) + 1:
            moves.append(Move(MoveKind.WASTE_TO_FOUNDATION, dst=w.suit))
        for d, col in enumerate(state.tableau):
            if _can_place_on_tableau(w, bool(col.up), col.top):
                moves.append(Move(MoveKind.WASTE_TO_TABLEAU, dst=d))

    # Tableau plays.
    for c, col in enumerate(state.tableau):
        if not col.up:
            continue
        top = col.up[-1]
        if top.rank == state.foundation_height(top.suit) + 1:
            moves.append(Move(MoveKind.TABLEAU_TO_FOUNDATION, src=c, dst=top.suit))
        for j, card in enumerate(col.up):
            for d, dst_col in enumerate(state.tableau):
                if d == c:
                    continue
                if _can_place_on_tableau(card, bool(dst_col.up), dst_col.top):
                    moves.append(Move(MoveKind.TABLEAU_TO_TABLEAU, src=c, dst=d, run_start=j))
    return moves


def apply_move(state: GameState, move: Move) -> MoveOutcome:
    """Apply a legal move in place. Returns what changed."""
    out = MoveOutcome()

    if move.kind == MoveKind.DRAW:
        if state.stock:
            n = min(state.draw_size, len(state.stock))
            for _ in range(n):
                state.waste.append(state.stock.pop())
            out.drew = True
        else:
            if not state.waste:
                raise ValueError("draw with empty stock and waste")
            if state.max_redeals is not None and state.redeals >= state.max_redeals:
                raise ValueError("redeals exhausted")
            state.stock = state.waste[::-1]
            state.waste = []
            state.redeals += 1
            out.recycled = True

    elif move.kind == MoveKind.WASTE_TO_FOUNDATION:
        card = state.waste.pop()
        if card.rank != state.foundation_height(card.suit) + 1:
            raise ValueError("illegal foundation move")
        state.foundations[card.suit].append(card)
        out.to_foundation = out.from_waste = True

    elif move.kind == MoveKind.WASTE_TO_TABLEAU:
        card = state.waste.pop()
        col = state.tableau[move.dst]
        if not _can_place_on_tableau(card, bool(col.up), col.top):
            raise ValueError("illegal waste->tableau move")
        col.up.append(card)
        out.from_waste = True

    elif move.kind == MoveKind.TABLEAU_TO_FOUNDATION:
        col = state.tableau[move.src]
        card = col.up.pop()
        if card.rank != state.foundation_height(card.suit) + 1:
            raise ValueError("illegal foundation move")
        state.foundations[card.suit].append(card)
        out.to_foundation = True
        if not col.up and col.down:
            col.up.append(col.down.pop())
            out.revealed = True
        out.emptied_column = not col.up and not col.down

    elif move.kind == MoveKind.TABLEAU_TO_TABLEAU:
        src, dst = state.tableau[move.src], state.tableau[move.dst]
        run = src.up[move.run_start :]
        if not run:
            raise ValueError("empty run")
        if not _can_place_on_tableau(run[0], bool(dst.up), dst.top):
            raise ValueError("illegal tableau->tableau move")
        dst.up.extend(run)
        del src.up[move.run_start :]
        if not src.up and src.down:
            src.up.append(src.down.pop())
            out.revealed = True
        out.emptied_column = not src.up and not src.down

    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown move kind {move.kind}")

    state.moves += 1
    return out
