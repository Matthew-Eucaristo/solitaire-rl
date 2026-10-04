"""Observation encoders — see OBSERVATION.md for the exact vector layout.

Two variants:

  * ``pomdp``  — only what a player may see: face-up tableau cards, face-down
    counts, the waste pile (contents and order — a player has seen every waste
    card as it was drawn), foundation heights, stock *count*, and scalars.
    The stock order and face-down identities are NOT observable.
  * ``perfect`` — pomdp plus the full stock order and face-down card
    identities (perfect information; suitable for search baselines).

Every value lies in [0, 1] (one-hots are exactly 0/1), so observations can be
stored losslessly enough as uint8 for replay buffers.
"""

from __future__ import annotations

import numpy as np

from solitaire_rl.state import GameState

MAX_UP = 13  # max face-up run length in a tableau column
MAX_DOWN = 6  # max face-down cards dealt into a column
MAX_STOCK = 24
MAX_WASTE = 24
CARD_DIM = 53  # 52 cards + empty slot

TABLEAU_DIM = 7 * MAX_UP * CARD_DIM  # 4823
WASTE_DIM = MAX_WASTE * CARD_DIM  # 1272
FOUNDATION_DIM = 4 * 14  # one-hot of top rank (0=empty .. 13=King)
DOWN_COUNT_DIM = 7
SCALAR_DIM = 4
POMDP_DIM = TABLEAU_DIM + DOWN_COUNT_DIM + WASTE_DIM + FOUNDATION_DIM + 1 + SCALAR_DIM

STOCK_ORDER_DIM = MAX_STOCK * CARD_DIM  # perfect-info extras
DOWN_CONTENT_DIM = 7 * MAX_DOWN * CARD_DIM
PERFECT_EXTRA_DIM = STOCK_ORDER_DIM + DOWN_CONTENT_DIM
PERFECT_DIM = POMDP_DIM + PERFECT_EXTRA_DIM

OBS_VARIANTS = ("pomdp", "perfect")


def _write_card(vec: np.ndarray, offset: int, card) -> None:
    idx = offset + (card.id if card is not None else 52)
    vec[idx] = 1.0


def encode_state(
    state: GameState,
    variant: str = "pomdp",
    *,
    idle_ratio: float = 0.0,
    has_legal_move: bool = True,
    can_draw: bool = True,
) -> np.ndarray:
    """Encode `state` as a flat float32 vector in [0, 1]."""
    if variant not in OBS_VARIANTS:
        raise ValueError(f"unknown obs variant {variant!r}")
    dim = POMDP_DIM if variant == "pomdp" else PERFECT_DIM
    obs = np.zeros(dim, dtype=np.float32)
    pos = 0

    # --- tableau face-up cards: 7 cols x 13 slots x 53 one-hot -------------
    for c in range(7):
        up = state.tableau[c].up
        for s in range(MAX_UP):
            _write_card(obs, pos + s * CARD_DIM, up[s] if s < len(up) else None)
        pos += MAX_UP * CARD_DIM

    # --- face-down counts (0..6 -> /6) ------------------------------------
    for c in range(7):
        obs[pos + c] = len(state.tableau[c].down) / MAX_DOWN
    pos += DOWN_COUNT_DIM

    # --- waste pile: 24 slots x 53 (order preserved) ----------------------
    for s in range(MAX_WASTE):
        _write_card(obs, pos + s * CARD_DIM, state.waste[s] if s < len(state.waste) else None)
    pos += WASTE_DIM

    # --- foundations: 4 x 14 one-hot of top rank --------------------------
    for suit in range(4):
        obs[pos + suit * 14 + state.foundation_height(suit)] = 1.0
    pos += FOUNDATION_DIM

    # --- stock count -------------------------------------------------------
    obs[pos] = len(state.stock) / MAX_STOCK
    pos += 1

    # --- scalars -----------------------------------------------------------
    obs[pos + 0] = float(np.clip(idle_ratio, 0.0, 1.0))
    obs[pos + 1] = 1.0 if has_legal_move else 0.0
    obs[pos + 2] = 1.0 if can_draw else 0.0
    obs[pos + 3] = min(state.redeals, 10) / 10.0
    pos += SCALAR_DIM

    if variant == "perfect":
        for s in range(MAX_STOCK):
            _write_card(obs, pos + s * CARD_DIM, state.stock[s] if s < len(state.stock) else None)
        pos += STOCK_ORDER_DIM
        for c in range(7):
            down = state.tableau[c].down
            for s in range(MAX_DOWN):
                _write_card(obs, pos + s * CARD_DIM, down[s] if s < len(down) else None)
            pos += MAX_DOWN * CARD_DIM

    assert pos == dim, (pos, dim)
    return obs
