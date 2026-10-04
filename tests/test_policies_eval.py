"""Policies, eval harness, renderer."""

import random

from solitaire_rl.env import KlondikeEnv
from solitaire_rl.evaluator import EpisodeResult, play_episode, run_eval
from solitaire_rl.policies import HeuristicPolicy, RandomPolicy


def test_random_policy_legal():
    env = KlondikeEnv()
    env.reset(seed=0)
    pol = RandomPolicy(seed=1)
    for _ in range(50):
        a = pol.act(env)
        assert env.action_masks()[a]
        _, _, t, tr, _ = env.step(a)
        if t or tr:
            break


def test_heuristic_deterministic():
    """Same deal, two heuristic instances => identical action sequences."""
    seqs = []
    for _ in range(2):
        env = KlondikeEnv()
        env.reset(seed=3)
        pol = HeuristicPolicy()
        seq = []
        for _ in range(60):
            a = pol.act(env)
            seq.append(a)
            _, _, t, tr, _ = env.step(a)
            if t or tr:
                break
        seqs.append(seq)
    assert seqs[0] == seqs[1]


def test_heuristic_only_legal_moves():
    rng = random.Random(0)
    pol = HeuristicPolicy()
    env = KlondikeEnv()
    for _game in range(10):
        env.reset(seed=rng.randrange(1_000_000))
        for _ in range(120):
            a = pol.act(env)
            assert env.action_masks()[a]
            _, _, t, tr, _ = env.step(a)
            if t or tr:
                break


def test_heuristic_beats_random_small():
    """On 30 deals the heuristic must score at least as many wins as random."""
    kw = {"variant": "draw1", "obs_variant": "pomdp"}
    h_wins = r_wins = 0
    for s in range(30):
        if play_episode(HeuristicPolicy(), s, kw).outcome == "win":
            h_wins += 1
        if play_episode(RandomPolicy(seed=0), s, kw).outcome == "win":
            r_wins += 1
    assert h_wins >= r_wins


def test_play_episode_result_fields():
    res = play_episode(HeuristicPolicy(), 0, {"variant": "draw1"})
    assert isinstance(res, EpisodeResult)
    assert res.outcome in (
        "win", "no_moves", "no_progress_cycle", "no_progress_idle",
        "truncated", "concede", "illegal_action",
    )
    assert res.moves > 0


def test_run_eval_aggregates():
    kw = {"variant": "draw1", "obs_variant": "pomdp"}
    summary = run_eval("heuristic", list(range(10)), kw, jobs=1)
    assert summary["n_deals"] == 10
    assert 0.0 <= summary["win_rate"] <= 1.0
    assert sum(summary["outcomes"].values()) == 10
    assert len(summary["results"]) == 10


def test_render_text_contains_cards():
    env = KlondikeEnv()
    env.reset(seed=0)
    txt = env.render()
    assert "STK" in txt and "FDN" in txt
    assert any(ch in txt for ch in "♣♦♥♠")


def test_ensemble_policy_majority():
    """EnsemblePolicy returns the majority vote of its members."""
    from solitaire_rl.policies import EnsemblePolicy

    class Fixed:
        def __init__(self, a):
            self.a = a

        def act(self, env):
            return self.a

    ens = EnsemblePolicy([Fixed(3), Fixed(3), Fixed(7)])
    assert ens.act(None) == 3
    ens2 = EnsemblePolicy([Fixed(1), Fixed(2), Fixed(2)])
    assert ens2.act(None) == 2


def test_load_policy_ens_spec():
    """ens: spec composes members via load_policy (heuristic x3 here —
    identical votes, exercises the plumbing without checkpoint IO)."""
    from solitaire_rl.policies import load_policy

    p = load_policy("ens:heuristic,heuristic,heuristic", {})
    env = KlondikeEnv()
    env.reset(seed=0)
    a = p.act(env)
    assert env.action_masks()[a]
