"""Web API tests: the API must expose the same rules the engine enforces."""

from fastapi.testclient import TestClient

from solitaire_rl.webapi import app

client = TestClient(app)


def _new(seed=42):
    r = client.post("/api/new", json={"seed": seed, "variant": "draw1"})
    assert r.status_code == 200
    return r.json()


def test_new_game_layout():
    st = _new(7)
    assert st["seed"] == 7
    assert st["stock_count"] == 24
    assert st["waste"] == []
    assert all(f == [] for f in st["foundations"])
    # column i: i face-down + 1 face-up
    for i, col in enumerate(st["tableau"]):
        assert col["down"] == i
        assert len(col["up"]) == 1
    assert st["outcome"] is None


def test_deterministic_deal():
    assert _new(123)["tableau"] == _new(123)["tableau"]


def test_legal_moves_present_and_draw_cycles_waste():
    st = _new(42)
    assert any(m["kind"] == 0 for m in st["legal"])  # DRAW
    st = client.post("/api/move", json={"kind": 0}).json()
    assert len(st["waste"]) == 1
    assert st["stock_count"] == 23


def test_illegal_move_rejected():
    _new(42)
    r = client.post("/api/move", json={"kind": 4, "src": 0, "dst": 1, "run_start": 0})
    assert r.status_code in (400, 409)


def test_undo_restores():
    st0 = _new(42)
    client.post("/api/move", json={"kind": 0})
    st = client.post("/api/undo").json()
    assert st["waste"] == [] and st["stock_count"] == 24
    assert st["moves"] == 0 == st0["moves"]


def test_heuristic_hint_and_step_change_state():
    _new(42)
    hint = client.post("/api/agent/hint", json={"policy": "heuristic"})
    assert hint.status_code == 200 and "suggested" in hint.json()
    st = client.post("/api/agent/step", json={"policy": "heuristic"}).json()
    assert st["moves"] == 1


def test_dqn_agent_step():
    _new(42)
    st = client.post("/api/agent/step", json={"policy": "dqn"}).json()
    assert st["moves"] == 1


def test_concede_ends_game():
    st = _new(42)
    st = client.post("/api/concede").json()
    assert st["outcome"] == "concede"
    r = client.post("/api/move", json={"kind": 0})
    assert r.status_code == 409
