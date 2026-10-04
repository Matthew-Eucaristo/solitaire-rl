"""Game state and deal generation for Klondike.

Layout after a standard deal (see RULES.md):
  * 7 tableau columns; column i (0-indexed) holds i+1 cards. The first i cards
    are face-down; the last card is face-up.
  * The remaining 24 cards form the stock.
  * Waste and the four foundations start empty.

List-end convention: for every pile, `pile[-1]` is the "top" card — the card
drawn next (stock), most recently played (waste/foundations), or the card new
cards are played onto (tableau `up`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from solitaire_rl.cards import Card, new_deck

NUM_TABLEAU = 7
STOCK_SIZE = 24

DRAW1 = "draw1"
DRAW3 = "draw3"
VARIANTS = (DRAW1, DRAW3)


@dataclass(slots=True)
class TableauColumn:
    """One tableau column: face-down cards plus a face-up run.

    Invariant: if `down` is non-empty then `up` is non-empty — as soon as the
    last face-up card leaves a column, the top face-down card is flipped.
    """

    down: list[Card] = field(default_factory=list)
    up: list[Card] = field(default_factory=list)

    @property
    def top(self) -> Card | None:
        return self.up[-1] if self.up else None


@dataclass(slots=True)
class GameState:
    """Complete mutable state of one Klondike game."""

    tableau: list[TableauColumn]
    stock: list[Card]  # stock[-1] is the next card drawn
    waste: list[Card]  # waste[-1] is the playable waste card
    foundations: list[list[Card]]  # one pile per suit
    draw_size: int = 1
    max_redeals: int | None = None  # None = unlimited redeals
    redeals: int = 0
    moves: int = 0

    # ------------------------------------------------------------------
    def clone(self) -> GameState:
        return GameState(
            tableau=[TableauColumn(list(c.down), list(c.up)) for c in self.tableau],
            stock=list(self.stock),
            waste=list(self.waste),
            foundations=[list(f) for f in self.foundations],
            draw_size=self.draw_size,
            max_redeals=self.max_redeals,
            redeals=self.redeals,
            moves=self.moves,
        )

    # ------------------------------------------------------------------
    def foundation_height(self, suit: int) -> int:
        """Rank of the top foundation card of `suit` (0 when empty)."""
        return len(self.foundations[suit])

    def foundation_total(self) -> int:
        return sum(len(f) for f in self.foundations)

    def facedown_count(self) -> int:
        return sum(len(c.down) for c in self.tableau)

    def is_win(self) -> bool:
        return self.foundation_total() == 52

    # ------------------------------------------------------------------
    def signature(self) -> tuple:
        """Hashable fingerprint of the *full* board state (including hidden
        stock order). `redeals` is deliberately excluded: a position that
        repeats after a redeal is still a cycle. Used for cycle detection
        only — never exposed in observations.
        """
        return (
            tuple((len(c.down), tuple(c.up)) for c in self.tableau),
            tuple(self.stock),
            tuple(self.waste),
            tuple(tuple(f) for f in self.foundations),
        )


def deal(seed: int, variant: str = DRAW1, max_redeals: int | None = None) -> GameState:
    """Deal a fresh Klondike game. `seed` fully determines the deal."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    rng = random.Random(seed)
    deck = new_deck()
    rng.shuffle(deck)

    tableau: list[TableauColumn] = []
    pos = 0
    for i in range(NUM_TABLEAU):
        n = i + 1
        column_cards = deck[pos : pos + n]
        pos += n
        tableau.append(TableauColumn(down=column_cards[:-1], up=column_cards[-1:]))

    stock = deck[pos:]
    # stock[-1] must be the first card drawn; the deck's top (deck[-1] after
    # the tableau deal consumed from the front) should be drawn first.
    stock.reverse()

    return GameState(
        tableau=tableau,
        stock=stock,
        waste=[],
        foundations=[[] for _ in range(4)],
        draw_size=1 if variant == DRAW1 else 3,
        max_redeals=max_redeals,
    )
