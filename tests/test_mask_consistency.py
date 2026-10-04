"""The big one: action_masks() must equal the legal-move set exactly.

We sample random *reachable* states (random playouts under a light heuristic
bias so we also see mid/late-game states), then check:
  1. mask ids == legal-move ids (plus CONCEDE)
  2. every legal move applies without error on a clone
  3. every legal move decodes back through action_to_move identically
"""

import os
import random

import numpy as np
import pytest

from solitaire_rl.actions import (
    ACTION_CONCEDE,
    action_mask,
    action_to_move,
    legal_action_ids,
    move_to_action,
)
from solitaire_rl.moves import MoveKind, apply_move, legal_moves
from solitaire_rl.state import deal

N_STATES = int(os.environ.get("MASK_STATES", "10000"))


def _random_states(n: int, seed: int = 0):
    """Yield (state,) sampled mid-game from random playouts."""
    rng = random.Random(seed)
    produced = 0
    game = 0
    while produced < n:
        st = deal(rng.randrange(10_000_000), variant=rng.choice(["draw1", "draw3"]))
        seen_sig = set()
        while produced < n:
            moves = legal_moves(st)
            if not moves:
                break
            yield st
            produced += 1
            # bias toward non-draw moves so we reach deep states
            non_draw = [m for m in moves if m.kind != MoveKind.DRAW]
            pick_from = non_draw if non_draw and rng.random() < 0.7 else moves
            m = rng.choice(pick_from)
            apply_move(st, m)
            sig = st.signature()
            if sig in seen_sig or st.is_win():
                break
            seen_sig.add(sig)
        game += 1


@pytest.mark.slow
def test_mask_matches_legal_on_random_states():
    checked = 0
    for st in _random_states(N_STATES):
        mask = action_mask(st)
        assert mask.shape == (654,)
        ids = set(np.flatnonzero(mask))
        legal = set(legal_action_ids(st)) | {ACTION_CONCEDE}
        assert ids == legal, f"mask mismatch: extra={ids - legal} missing={legal - ids}"
        # every legal move applies cleanly on a clone and round-trips
        for m in legal_moves(st):
            aid = move_to_action(m)
            assert mask[aid]
            decoded = action_to_move(aid, st)
            assert decoded == m
            st2 = st.clone()
            apply_move(st2, m)
        checked += 1
    assert checked == N_STATES


def test_mask_size_and_concede():
    st = deal(0)
    mask = action_mask(st)
    assert mask.dtype == bool
    assert mask[ACTION_CONCEDE]
    assert mask.sum() == len(legal_action_ids(st)) + 1


def test_illegal_ids_never_masked():
    for st in _random_states(200, seed=1):
        mask = action_mask(st)
        legal = set(legal_action_ids(st)) | {ACTION_CONCEDE}
        for a in range(654):
            assert (a in legal) == bool(mask[a])
