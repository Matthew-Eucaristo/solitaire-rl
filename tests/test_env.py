"""Env API: reset/step contract, rewards, termination, determinism."""

import numpy as np

from solitaire_rl.actions import ACTION_CONCEDE, ACTION_DRAW, N_ACTIONS
from solitaire_rl.cards import Card
from solitaire_rl.env import KlondikeEnv
from solitaire_rl.obs import PERFECT_DIM, POMDP_DIM
from solitaire_rl.policies import RandomPolicy
from solitaire_rl.state import GameState, TableauColumn


def _finish_state() -> GameState:
    """A state one move away from winning: 51 cards founded, K♣ on t0."""
    st = GameState(
        tableau=[TableauColumn([], [Card(13, 0)])] + [TableauColumn([], []) for _ in range(6)],
        stock=[],
        waste=[],
        foundations=[
            [Card(r, 0) for r in range(1, 13)],  # clubs A..Q (12)
            [Card(r, 1) for r in range(1, 14)],  # diamonds
            [Card(r, 2) for r in range(1, 14)],  # hearts
            [Card(r, 3) for r in range(1, 14)],  # spades
        ],
    )
    return st


def test_reset_contract():
    env = KlondikeEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (8 * POMDP_DIM,)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0) and np.all(obs <= 1)
    assert info["action_mask"].shape == (N_ACTIONS,)
    assert info["action_mask"][ACTION_DRAW]


def test_step_returns_gym_tuple():
    env = KlondikeEnv()
    env.reset(seed=0)
    out = env.step(ACTION_DRAW)
    _obs, r, term, trunc, info = out
    assert isinstance(r, float)
    assert not term and not trunc
    assert info["action_mask"].any()


def test_illegal_action_terminates():
    env = KlondikeEnv()
    env.reset(seed=0)
    mask = env.action_masks()
    bad = int(np.flatnonzero(~mask)[0])
    _, r, term, _, info = env.step(bad)
    assert term
    assert info["outcome"] == "illegal_action"
    assert r < 0


def test_concede_always_legal_and_ends():
    env = KlondikeEnv()
    env.reset(seed=0)
    assert env.action_masks()[ACTION_CONCEDE]
    _, _, term, _, info = env.step(ACTION_CONCEDE)
    assert term and info["outcome"] == "concede"


def test_win_on_last_foundation_move():
    env = KlondikeEnv()
    env.reset(seed=0)
    env.state = _finish_state()
    env._legal_cache = None
    env._seen = {env.state.signature()}
    # find the t0->foundation action id: 9 + 0
    _, r, term, _, info = env.step(9)
    assert term
    assert info["outcome"] == "win"
    assert r >= 1.0


def test_max_steps_truncates():
    env = KlondikeEnv(max_steps=3)
    env.reset(seed=0)
    done = False
    for _ in range(5):
        _, _, term, trunc, info = env.step(ACTION_DRAW)
        if term or trunc:
            done = True
            break
    assert done
    assert info["outcome"] in ("truncated", "win", "no_moves", "no_progress_cycle")


def test_cycle_detection_terminates():
    # A stock-only draw-1 game: draw all 24, recycle -> identical signature.
    st = GameState(
        tableau=[TableauColumn([], []) for _ in range(7)],
        stock=[Card(1 + (i % 13), (i // 13) % 4) for i in range(24)],
        waste=[],
        foundations=[[], [], [], []],
    )
    env = KlondikeEnv()
    env.reset(seed=0)
    env.state = st
    env._legal_cache = None
    env._seen = {st.signature()}
    env._steps = 0
    outcome = None
    for _ in range(30):
        _, _, term, trunc, info = env.step(ACTION_DRAW)
        if term or trunc:
            outcome = info["outcome"]
            break
    assert outcome in ("no_progress_cycle", "no_progress_idle", "no_moves")


def test_determinism_replay():
    """Same seed + same action sequence => identical trajectories."""

    def run(actions: list[int] | None = None) -> tuple:
        env = KlondikeEnv()
        obs, _ = env.reset(seed=42)
        pol = RandomPolicy(seed=7)
        traj = [obs.tobytes()]
        rewards = []
        taken: list[int] = []
        while True:
            a = pol.act(env) if actions is None else actions[len(taken)]
            taken.append(a)
            obs, r, t, tr, i = env.step(a)
            traj.append(obs.tobytes())
            rewards.append(r)
            if t or tr:
                return traj, rewards, i["outcome"], taken

    t1, r1, o1, actions = run()
    t2, r2, o2, _ = run(actions)
    assert t1 == t2 and r1 == r2 and o1 == o2


def test_frame_stack_changes_obs_shape():
    env = KlondikeEnv(frame_stack=4)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4 * POMDP_DIM,)
    env2 = KlondikeEnv(frame_stack=1)
    obs2, _ = env2.reset(seed=0)
    assert obs2.shape == (POMDP_DIM,)


def test_perfect_obs_has_stock_order():
    env = KlondikeEnv(obs_variant="perfect", frame_stack=1)
    env.reset(seed=0)
    obs = env._obs()
    assert obs.shape == (PERFECT_DIM,)
    st = env.state
    # first stock card must appear one-hot in the stock-order block
    import solitaire_rl.obs as obs_mod

    base = obs_mod.POMDP_DIM
    top = st.stock[-1]
    block = obs[base : base + 24 * 53].reshape(24, 53)
    assert block[0][top.id] == 1.0 or block[-1][top.id] == 1.0


def test_compact_obs_dims_and_range():
    env = KlondikeEnv(obs_variant="compact", frame_stack=8)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (8 * 132,)
    assert obs.min() >= -1.0 and obs.max() <= 1.0
    env2 = KlondikeEnv(obs_variant="compact_perfect", frame_stack=1)
    obs2, _ = env2.reset(seed=1)
    assert obs2.shape == (198,)


def test_compact_hides_stock_order():
    """Compact POMDP obs must not leak the stock order."""
    env = KlondikeEnv(obs_variant="compact", frame_stack=1)
    env.reset(seed=1)
    st = env.state
    before = env._raw_obs().copy()
    st.stock.reverse()  # scramble hidden order
    after = env._raw_obs()
    np.testing.assert_array_equal(before, after)


def test_pomdp_hides_stock_order():
    """Two states differing only in stock order must produce the same POMDP obs."""
    env = KlondikeEnv(obs_variant="pomdp", frame_stack=1)
    env.reset(seed=0)
    st = env.state
    a = env._raw_obs().copy()
    st.stock = st.stock[::-1]  # permute hidden order
    b = env._raw_obs()
    assert np.array_equal(a, b)


def test_sparse_reward_mode():
    env = KlondikeEnv(reward_mode="sparse")
    env.reset(seed=0)
    _, r, *_ = env.step(ACTION_DRAW)
    assert r == 0.0
