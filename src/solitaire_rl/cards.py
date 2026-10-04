"""Card primitives for Klondike Solitaire.

A card is an immutable (rank, suit) pair. Ranks are 1..13 (Ace..King).
Suits are 0..3: clubs, diamonds, hearts, spades. Clubs/spades are black,
diamonds/hearts are red.
"""

from __future__ import annotations

from dataclasses import dataclass

CLUBS = 0
DIAMONDS = 1
HEARTS = 2
SPADES = 3

NUM_SUITS = 4
NUM_RANKS = 13  # Ace=1 .. King=13
DECK_SIZE = 52

RED_SUITS = frozenset({DIAMONDS, HEARTS})
SUIT_SYMBOLS = ("♣", "♦", "♥", "♠")
RANK_CHARS = ".A23456789TJQK"  # index by rank; T=10


@dataclass(frozen=True, slots=True)
class Card:
    """An immutable playing card."""

    rank: int  # 1 (Ace) .. 13 (King)
    suit: int  # 0=clubs 1=diamonds 2=hearts 3=spades

    def __post_init__(self) -> None:
        if not (1 <= self.rank <= NUM_RANKS):
            raise ValueError(f"bad rank {self.rank}")
        if not (0 <= self.suit < NUM_SUITS):
            raise ValueError(f"bad suit {self.suit}")

    @property
    def is_red(self) -> bool:
        return self.suit in RED_SUITS

    @property
    def id(self) -> int:
        """Unique id in 0..51, used for one-hot encodings."""
        return self.suit * NUM_RANKS + (self.rank - 1)

    @staticmethod
    def from_id(card_id: int) -> Card:
        if not (0 <= card_id < DECK_SIZE):
            raise ValueError(f"bad card id {card_id}")
        return Card(rank=card_id % NUM_RANKS + 1, suit=card_id // NUM_RANKS)

    def __str__(self) -> str:
        return f"{RANK_CHARS[self.rank]}{SUIT_SYMBOLS[self.suit]}"


def new_deck() -> list[Card]:
    """Return a freshly ordered 52-card deck."""
    return [Card.from_id(i) for i in range(DECK_SIZE)]


def other_color(card_a: Card, card_b: Card) -> bool:
    """True when the two cards have opposite colors."""
    return card_a.is_red != card_b.is_red
