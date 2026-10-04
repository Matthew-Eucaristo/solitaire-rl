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

# --- compact encoding ------------------------------------------------------
# Per-slot scalar code instead of one-hot: EMPTY=-1.0, face-down/unknown
# =-0.5, otherwise card.id / 52 in [0, 1]. Orders of magnitude fewer dims;
# the right inductive step for small MLPs on a laptop budget.
CODE_EMPTY = -1.0
CODE_DOWN = -0.5
COMPACT_COL_DIM = MAX_UP + 1  # 13 up-slot codes + down count
COMPACT_POMDP_DIM = (
    7 * COMPACT_COL_DIM + MAX_WASTE + 4 + 1 + 5  # = 132
)
COMPACT_PERFECT_EXTRA_DIM = MAX_STOCK + 7 * MAX_DOWN  # = 66
COMPACT_PERFECT_DIM = COMPACT_POMDP_DIM + COMPACT_PERFECT_EXTRA_DIM  # = 198

# --- hint encoding -----------------------------------------------------------
# The deterministic heuristic's chosen action as extra feature dims: the
# policy sees the expert's suggestion and learns when to deviate from it.
# kind one-hot (6) + src/6 + dst/6 + run_start/12 = 10 dims.
HINT_KINDS = 6  # draw, w2t, w2f, t2f, t2t, concede
HINT_DIM = HINT_KINDS + 3  # = 10
COMPACT_HINT_DIM = COMPACT_POMDP_DIM + HINT_DIM  # = 142

OBS_VARIANTS = ("pomdp", "perfect", "compact", "compact_perfect", "compact_hint")


def obs_dim(variant: str) -> int:
    return {
        "pomdp": POMDP_DIM,
        "perfect": PERFECT_DIM,
        "compact": COMPACT_POMDP_DIM,
        "compact_perfect": COMPACT_PERFECT_DIM,
        "compact_hint": COMPACT_HINT_DIM,
    }[variant]


def _code(card) -> float:
    if card is None:
        return CODE_EMPTY
    return card.id / 52.0


def _encode_compact(
    state: GameState,
    perfect: bool,
    *,
    idle_ratio: float,
    has_legal_move: bool,
    can_draw: bool,
) -> np.ndarray:
    dim = COMPACT_PERFECT_DIM if perfect else COMPACT_POMDP_DIM
    obs = np.full(dim, CODE_EMPTY, dtype=np.float32)
    pos = 0
    for c in range(7):
        col = state.tableau[c]
        for s in range(MAX_UP):
            obs[pos + s] = _code(col.up[s]) if s < len(col.up) else CODE_EMPTY
        pos += MAX_UP
        obs[pos] = len(col.down) / MAX_DOWN
        pos += 1
    for s in range(MAX_WASTE):
        obs[pos + s] = _code(state.waste[s]) if s < len(state.waste) else CODE_EMPTY
    pos += MAX_WASTE
    for suit in range(4):
        obs[pos + suit] = state.foundation_height(suit) / 13.0
    pos += 4
    obs[pos] = len(state.stock) / MAX_STOCK
    pos += 1
    obs[pos + 0] = float(np.clip(idle_ratio, 0.0, 1.0))
    obs[pos + 1] = 1.0 if has_legal_move else 0.0
    obs[pos + 2] = 1.0 if can_draw else 0.0
    obs[pos + 3] = min(state.redeals, 10) / 10.0
    obs[pos + 4] = min(state.moves, 500) / 500.0
    pos += 5
    if perfect:
        for s in range(MAX_STOCK):
            obs[pos + s] = _code(state.stock[s]) if s < len(state.stock) else CODE_EMPTY
        pos += MAX_STOCK
        for c in range(7):
            down = state.tableau[c].down
            for s in range(MAX_DOWN):
                obs[pos + s] = _code(down[s]) if s < len(down) else CODE_EMPTY
            pos += MAX_DOWN
    assert pos == dim, (pos, dim)
    return obs


def encode_hint(action: int) -> np.ndarray:
    """Encode an action id (the heuristic's suggestion) as HINT_DIM floats.

    kind one-hot [draw, w2t, w2f, t2f, t2t, concede], then src/dst/run_start
    normalized to [0, 1] (0 when the kind has no such field).
    """
    vec = np.zeros(HINT_DIM, dtype=np.float32)
    if action == 0:
        vec[0] = 1.0
    elif 1 <= action <= 7:
        vec[1] = 1.0
        vec[HINT_KINDS + 1] = (action - 1) / 6.0
    elif action == 8:
        vec[2] = 1.0
    elif 9 <= action <= 15:
        vec[3] = 1.0
        vec[HINT_KINDS + 0] = (action - 9) / 6.0
    elif action == 653:
        vec[5] = 1.0
    elif 16 <= action < 16 + 7 * 91:
        vec[4] = 1.0
        rel = action - 16
        src, rem = divmod(rel, 91)
        dst, j = divmod(rem, 13)
        vec[HINT_KINDS + 0] = src / 6.0
        vec[HINT_KINDS + 1] = dst / 6.0
        vec[HINT_KINDS + 2] = j / 12.0
    else:
        raise ValueError(f"action {action} out of range")
    return vec


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
    hint_action: int = 0,
) -> np.ndarray:
    """Encode `state` as a flat float32 vector in [0, 1] (compact: [-1, 1])."""
    if variant not in OBS_VARIANTS:
        raise ValueError(f"unknown obs variant {variant!r}")
    if variant == "compact_hint":
        base = _encode_compact(
            state,
            perfect=False,
            idle_ratio=idle_ratio,
            has_legal_move=has_legal_move,
            can_draw=can_draw,
        )
        return np.concatenate([base, encode_hint(hint_action)])
    if variant.startswith("compact"):
        return _encode_compact(
            state,
            perfect=variant == "compact_perfect",
            idle_ratio=idle_ratio,
            has_legal_move=has_legal_move,
            can_draw=can_draw,
        )
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
