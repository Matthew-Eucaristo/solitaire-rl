"""Baseline policies: uniform-random and a deterministic heuristic.

The heuristic's priority order is documented in HEURISTIC.md. Both policies
act on the *observable* game state (they never inspect hidden stock order or
face-down card identities), so they are legal POMDP players — the heuristic's
anti-repeat check reads the env's own seen-signature bookkeeping, not hidden
cards.

Policy protocol: ``policy.act(env) -> action_id``. ``env`` exposes ``.state``,
``.action_masks()`` and ``.legal_ids()``; learned-policy wrappers use the same
protocol plus the current observation.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from solitaire_rl.actions import ACTION_CONCEDE, move_to_action
from solitaire_rl.moves import Move, MoveKind, _can_place_on_tableau, apply_move
from solitaire_rl.state import GameState

if TYPE_CHECKING:
    from solitaire_rl.env import KlondikeEnv


class RandomPolicy:
    """Uniform over legal *game* moves.

    CONCEDE is deliberately excluded: it is a resignation channel for learned
    agents, and sampling it uniformly would make random play degenerate
    (instant give-up every ~30 moves). See DECISIONS.md.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def act(self, env: KlondikeEnv) -> int:
        moves = env._legal()
        if not moves:
            return ACTION_CONCEDE
        return move_to_action(self._rng.choice(moves))


class HeuristicPolicy:
    """Deterministic priority heuristic with anti-repeat filtering.

    Move tiers (first non-empty tier wins; see HEURISTIC.md):
      P1 reveal a face-down card (prefer the column with most face-down)
      P2 safe foundation move (tableau first, then waste)
      P3 waste -> tableau
      P4 productive tableau->tableau (scored; see `_t2t_score`)
      P5 unsafe foundation — only when the stock is empty
      P6 draw / redeal
      P7 any non-repeating move that is left
      P8 CONCEDE — every remaining move revisits a seen position, so the game
         is provably looping: the heuristic resigns instead of drawing dead.

    Any candidate move whose post-move state signature was already seen this
    episode is filtered out before the tiers run — repeating a position can
    never make progress, and this check is exactly how the agent "knows when
    to give up".
    """

    def act(self, env: KlondikeEnv) -> int:
        state = env.state
        assert state is not None
        moves = env._legal()
        if not moves:
            return ACTION_CONCEDE

        fresh = [m for m in moves if not self._repeats(env, m)]
        if not fresh:
            return ACTION_CONCEDE
        moves = fresh

        t2t = [m for m in moves if m.kind == MoveKind.TABLEAU_TO_TABLEAU]
        t2f = [m for m in moves if m.kind == MoveKind.TABLEAU_TO_FOUNDATION]
        w2f = [m for m in moves if m.kind == MoveKind.WASTE_TO_FOUNDATION]
        w2t = [m for m in moves if m.kind == MoveKind.WASTE_TO_TABLEAU]
        draw = [m for m in moves if m.kind == MoveKind.DRAW]

        # ---- P1: moves that flip a face-down card -------------------------
        revealers: list[tuple[int, int, Move]] = []  # (-down, prefer_fdn, move)
        for m in t2f:
            col = state.tableau[m.src]
            if len(col.up) == 1 and col.down:
                revealers.append((-len(col.down), 1, m))
        for m in t2t:
            col = state.tableau[m.src]
            if m.run_start == 0 and col.down:
                revealers.append((-len(col.down), 0, m))
        if revealers:
            revealers.sort(key=lambda t: (t[0], -t[1], t[2].src, t[2].dst, t[2].run_start))
            return move_to_action(revealers[0][2])

        # ---- P2: safe foundation moves ------------------------------------
        safe_t2f = [m for m in t2f if self._is_safe(state, m.src)]
        if safe_t2f:
            return move_to_action(safe_t2f[0])
        if w2f and self._is_safe_waste(state):
            return move_to_action(w2f[0])

        # ---- P3: waste -> tableau -----------------------------------------
        if w2t:
            return move_to_action(w2t[0])

        # ---- P4: productive tableau->tableau -------------------------------
        productive = [m for m in t2t if self._productive_t2t(state, m)]
        if productive:
            best = max(productive, key=lambda m: self._t2t_score(state, m))
            return move_to_action(best)

        # ---- P5: unsafe foundation, only once the stock is empty ----------
        if not state.stock:
            if w2f:
                return move_to_action(w2f[0])
            if t2f:
                return move_to_action(t2f[0])

        # ---- P6: draw / redeal --------------------------------------------
        if draw:
            return move_to_action(draw[0])

        # ---- P7: any remaining non-repeating move --------------------------
        return move_to_action(moves[0])

    # ----------------------------------------------------------------------
    @staticmethod
    def _repeats(env: KlondikeEnv, move: Move) -> bool:
        """True when applying `move` would revisit a seen board signature.

        Only TABLEAU_TO_TABLEAU and DRAW can revisit a state — every other
        kind irreversibly moves a card (waste plays shrink the waste,
        foundation moves grow the foundations). Checking those two kinds
        covers every reachable repeat.
        """
        if move.kind not in (MoveKind.TABLEAU_TO_TABLEAU, MoveKind.DRAW):
            return False
        st = env.state
        assert st is not None
        probe = st.clone()
        apply_move(probe, move)
        return probe.signature() in env._seen

    def _is_safe(self, state: GameState, src_col: int) -> bool:
        card = state.tableau[src_col].up[-1]
        return self._safe_card(state, card)

    def _is_safe_waste(self, state: GameState) -> bool:
        return self._safe_card(state, state.waste[-1])

    @staticmethod
    def _safe_card(state: GameState, card) -> bool:
        """Safe-to-foundation: ranks 1-2 always; otherwise both opposite-color
        foundations must already reach rank-1 (the card can never be needed
        as a holder for a lower card of the opposite color)."""
        if card.rank <= 2:
            return True
        red = card.suit in (1, 2)
        opp_min = min(
            (state.foundation_height(s) for s in range(4) if (s in (1, 2)) != red),
            default=0,
        )
        return opp_min >= card.rank - 1

    @staticmethod
    def _productive_t2t(state: GameState, m: Move) -> bool:
        """A tableau->tableau move counts as productive when it:

        * run_start == 0: reveals a face-down card (a whole-pile move to an
          empty column with no down beneath is a pure shuffle, never useful);
        * run_start > 0: unburies the face-up card below the moved run AND
          that card goes straight to a foundation, can itself move onto a
          third column, or still sits above face-down cards (the covering
          pile got shorter toward a flip).
        """
        src = state.tableau[m.src]
        if m.run_start == 0:
            return bool(src.down)
        new_top = src.up[m.run_start - 1]
        if new_top.rank == state.foundation_height(new_top.suit) + 1:
            return True
        if src.down:
            return True  # shortened the pile covering face-down cards
        for d2, c2 in enumerate(state.tableau):
            if d2 in (m.src, m.dst):
                continue
            if _can_place_on_tableau(new_top, bool(c2.up), c2.top):
                return True
        return False

    @staticmethod
    def _t2t_score(state: GameState, m: Move) -> tuple:
        src = state.tableau[m.src]
        dst = state.tableau[m.dst]
        run_len = len(src.up) - m.run_start
        unburies = 1 if m.run_start > 0 else 0  # frees a face-up card below
        extends = 1 if dst.up else 0  # lands on an existing run
        new_top_founds = 0
        if m.run_start > 0:
            nt = src.up[m.run_start - 1]
            if nt.rank == state.foundation_height(nt.suit) + 1:
                new_top_founds = 1
        return (
            unburies * 40 + new_top_founds * 25 + len(src.down) * 8 + extends * 10 + run_len,
            -m.src,
            -m.dst,
            -m.run_start,
        )


# ----------------------------------------------------------------------
# Learned-policy wrappers (DQN checkpoints / MaskablePPO models)
# ----------------------------------------------------------------------


class MaskedArgmaxPolicy:
    """Argmax over Q-values / policy logits with the action mask applied."""

    def __init__(self, predict_fn) -> None:
        self._predict = predict_fn

    def act(self, env: KlondikeEnv) -> int:
        return self._predict(env)


class EnsemblePolicy:
    """Majority vote over member policies (all sharing the env's obs).

    Voting beats every member when members are comparably strong —
    measured +0.027 win rate on the 1000-deal benchmark (see EVIDENCE.md
    batch-8). Weak members hurt; ensemble only the top checkpoints.
    """

    def __init__(self, members: list) -> None:
        self._members = members

    def act(self, env: KlondikeEnv) -> int:
        votes = [p.act(env) for p in self._members]
        return max(set(votes), key=votes.count)


def load_policy(spec: str, env_kwargs: dict, seed: int = 0):
    """Build a policy from a spec string.

    spec: ``random`` | ``heuristic`` | ``dqn:<path>`` | ``ppo:<path>``
    | ``qrdqn:<path>`` | ``ens:<spec>,<spec>,...`` (majority-vote ensemble)
    """
    if spec == "random":
        return RandomPolicy(seed=seed)
    if spec == "heuristic":
        return HeuristicPolicy()
    if spec.startswith("dqn:"):
        from solitaire_rl.dqn import DQNPolicy

        return DQNPolicy.load(spec[4:], env_kwargs)
    if spec.startswith("ppo:"):
        from solitaire_rl.ppo_wrap import PPOPolicy

        return PPOPolicy.load(spec[4:], env_kwargs)
    if spec.startswith("qrdqn:"):
        from solitaire_rl.qrdqn import QRDQNPolicy

        return QRDQNPolicy.load(spec[6:], env_kwargs)
    if spec.startswith("ens:"):
        members = [load_policy(s, env_kwargs, seed) for s in spec[4:].split(",")]
        return EnsemblePolicy(members)
    raise ValueError(f"unknown policy spec {spec!r}")
