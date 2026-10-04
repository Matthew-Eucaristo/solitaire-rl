"""Fixed action enumeration — see ACTION_SCHEMA.md for the full spec.

Discrete(654), stable across versions:

    0           DRAW — draw from stock, or redeal waste->stock when empty
    1..7        WASTE -> TABLEAU column t   (id = 1 + t)
    8           WASTE -> FOUNDATION (suit implied by the waste top card)
    9..15       TABLEAU c top -> FOUNDATION (id = 9 + c)
    16..652     TABLEAU src -> TABLEAU dst moving run up[j:]
                id = 16 + src*91 + dst*13 + j   (j in 0..12)
    653         CONCEDE — always legal; ends the episode as a loss
"""

from __future__ import annotations

import numpy as np

from solitaire_rl.moves import Move, MoveKind, legal_moves
from solitaire_rl.state import GameState

ACTION_DRAW = 0
ACTION_WASTE_TO_TABLEAU_BASE = 1  # + dst col (0..6) -> 1..7
ACTION_WASTE_TO_FOUNDATION = 8
ACTION_TABLEAU_TO_FOUNDATION_BASE = 9  # + src col -> 9..15
ACTION_TABLEAU_TO_TABLEAU_BASE = 16  # src*91 + dst*13 + j -> 16..652
ACTION_CONCEDE = 653
N_ACTIONS = 654

MAX_RUN_LEN = 13  # a descending alternating run can hold at most K..A


def move_to_action(move: Move) -> int:
    if move.kind == MoveKind.DRAW:
        return ACTION_DRAW
    if move.kind == MoveKind.WASTE_TO_TABLEAU:
        return ACTION_WASTE_TO_TABLEAU_BASE + move.dst
    if move.kind == MoveKind.WASTE_TO_FOUNDATION:
        return ACTION_WASTE_TO_FOUNDATION
    if move.kind == MoveKind.TABLEAU_TO_FOUNDATION:
        return ACTION_TABLEAU_TO_FOUNDATION_BASE + move.src
    if move.kind == MoveKind.TABLEAU_TO_TABLEAU:
        return ACTION_TABLEAU_TO_TABLEAU_BASE + move.src * 91 + move.dst * 13 + move.run_start
    raise ValueError(f"cannot encode {move}")


def action_to_move(action: int, state: GameState) -> Move | None:
    """Decode an action id into a Move in `state`, or None when CONCEDE.

    Raises ValueError for structurally impossible ids. Legality in the
    current state is the caller's concern (the env checks the mask).
    """
    if action == ACTION_CONCEDE:
        return None
    if action == ACTION_DRAW:
        return Move(MoveKind.DRAW)
    if ACTION_WASTE_TO_TABLEAU_BASE <= action <= ACTION_WASTE_TO_TABLEAU_BASE + 6:
        return Move(MoveKind.WASTE_TO_TABLEAU, dst=action - ACTION_WASTE_TO_TABLEAU_BASE)
    if action == ACTION_WASTE_TO_FOUNDATION:
        if not state.waste:
            raise ValueError("waste->foundation with empty waste")
        w = state.waste[-1]
        return Move(MoveKind.WASTE_TO_FOUNDATION, dst=w.suit)
    if ACTION_TABLEAU_TO_FOUNDATION_BASE <= action <= ACTION_TABLEAU_TO_FOUNDATION_BASE + 6:
        c = action - ACTION_TABLEAU_TO_FOUNDATION_BASE
        if not state.tableau[c].up:
            raise ValueError("tableau->foundation from empty column")
        top = state.tableau[c].up[-1]
        return Move(MoveKind.TABLEAU_TO_FOUNDATION, src=c, dst=top.suit)
    rel = action - ACTION_TABLEAU_TO_TABLEAU_BASE
    if 0 <= rel < 7 * 91:
        src, rem = divmod(rel, 91)
        dst, j = divmod(rem, 13)
        return Move(MoveKind.TABLEAU_TO_TABLEAU, src=src, dst=dst, run_start=j)
    raise ValueError(f"action {action} out of range")


def legal_action_ids(state: GameState) -> list[int]:
    """Action ids of every legal game move (CONCEDE excluded)."""
    return [move_to_action(m) for m in legal_moves(state)]


def action_mask(state: GameState) -> np.ndarray:
    """Boolean mask over Discrete(654): legal moves + CONCEDE."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for a in legal_action_ids(state):
        mask[a] = True
    mask[ACTION_CONCEDE] = True
    return mask
