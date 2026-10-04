"""FastAPI web app: play Klondike manually in a browser, or watch trained
agents play it — all on the *same* rules engine the tests and evals use.

Run:  `.venv/bin/python -m uvicorn solitaire_rl.webapi:app --port 8080`
      (repo root; the static dir is `<repo>/web`)

Agent play is faithful to eval: the agent's env is reconstructed by replaying
this game's action history into a fresh env with the policy's own obs variant,
so PPO/DQN see exactly the observations they were trained on (frame-stack
included). Hidden information stays hidden — the API never sends stock order
or face-down card identities to the client.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from solitaire_rl.actions import ACTION_CONCEDE, action_to_move, move_to_action
from solitaire_rl.env import KlondikeEnv
from solitaire_rl.moves import Move, MoveKind
from solitaire_rl.state import DRAW1, GameState

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"

# env_kwargs each learned checkpoint was trained/evaluated on.
_AGENT_POLICIES: dict[str, dict[str, Any]] = {
    "heuristic": {"label": "Heuristic (rule-based, 41.4%)", "spec": "heuristic", "env": {}},
    "dqn": {
        "label": "Masked DQN — 250k steps (5.8%)",
        "spec": "dqn:checkpoints/dqn_compact_250k.pt",
        "env": {"obs_variant": "compact"},
    },
    "ppo": {
        "label": "MaskablePPO — 6M steps (11.8%)",
        "spec": "ppo:checkpoints/ppo_compact_6m.zip",
        "env": {"obs_variant": "compact"},
    },
    "ppo_perfect": {
        "label": "PPO perfect-info — 6M steps (17.7%)",
        "spec": "ppo:checkpoints/ppo_perfect_6m.zip",
        "env": {"obs_variant": "compact_perfect"},
    },
    "ppo_amp": {
        "label": "PPO amplified — best single model (35.3%)",
        "spec": "ppo:checkpoints/ppo_amp1_9m_1p6m.zip",
        "env": {"obs_variant": "compact_hint"},
    },
    "ens": {
        "label": "Ensemble vote x3 — best artifact (38.0%)",
        "spec": (
            "ens:ppo:checkpoints/ppo_amp1_9m_1p6m.zip,"
            "ppo:checkpoints/ppo_amp1_9m_2p8m.zip,"
            "ppo:checkpoints/ppo_amp1_12m.zip"
        ),
        "env": {"obs_variant": "compact_hint"},
    },
}


def _card(c) -> str:
    """'AS', '5H', 'TC', 'KD' — rank char + suit letter (ASCII, UI-friendly)."""
    ranks = ".A23456789TJQK"
    suits = "CDHS"
    return f"{ranks[c.rank]}{suits[c.suit]}"


def _move_desc(state: GameState, m: Move) -> dict[str, Any]:
    """A move as UI-friendly JSON. `cards` are the exact cards it moves."""
    d: dict[str, Any] = {
        "kind": int(m.kind),
        "kind_name": m.kind.name,
        "src": m.src,
        "dst": m.dst,
        "run_start": m.run_start,
        "action": move_to_action(m),
    }
    if m.kind == MoveKind.WASTE_TO_FOUNDATION and state.waste:
        d["dst"] = state.waste[-1].suit
        d["cards"] = [_card(state.waste[-1])]
    elif m.kind == MoveKind.WASTE_TO_TABLEAU and state.waste:
        d["cards"] = [_card(state.waste[-1])]
    elif m.kind == MoveKind.TABLEAU_TO_FOUNDATION and m.src >= 0:
        col = state.tableau[m.src]
        if col.up:
            d["dst"] = col.up[-1].suit
            d["cards"] = [_card(col.up[-1])]
    elif m.kind == MoveKind.TABLEAU_TO_TABLEAU and m.src >= 0:
        col = state.tableau[m.src]
        d["cards"] = [_card(c) for c in col.up[m.run_start :]]
    return d


@dataclass
class Session:
    env: KlondikeEnv
    history: list[int]  # applied action ids (manual + agent moves)
    undo: list[KlondikeEnv]  # env deep-copies, one per applied move
    _agents: dict[str, Any] | None = None

    def agent(self, key: str):
        from solitaire_rl.policies import load_policy

        if self._agents is None:
            self._agents = {}
        if key not in self._agents:
            spec = _AGENT_POLICIES[key]
            self._agents[key] = load_policy(spec["spec"], spec["env"])
        return self._agents[key]


SESSION = Session(
    env=KlondikeEnv(variant=DRAW1, obs_variant="compact", train_seeds=False),
    history=[],
    undo=[],
)


def _serialize(sess: Session) -> dict[str, Any]:
    st = sess.env.state
    assert st is not None
    info = sess.env._info()
    moves = sess.env._legal()
    return {
        "seed": info["seed"],
        "variant": "draw3" if st.draw_size == 3 else "draw1",
        "draw_size": st.draw_size,
        "moves": st.moves,
        "redeals": st.redeals,
        "stock_count": len(st.stock),
        "waste": [_card(c) for c in st.waste],
        "foundations": [[_card(c) for c in f] for f in st.foundations],
        "tableau": [
            {"down": len(col.down), "up": [_card(c) for c in col.up]} for col in st.tableau
        ],
        "outcome": info["outcome"],
        "foundations_count": info["foundations"],
        "facedown": info["facedown"],
        "legal": [_move_desc(st, m) for m in moves],
        "can_undo": bool(sess.undo),
    }


class NewGame(BaseModel):
    seed: int | None = None
    variant: str = "draw1"


class MoveReq(BaseModel):
    kind: int
    src: int = -1
    dst: int = -1
    run_start: int = -1


class AgentReq(BaseModel):
    policy: str


app = FastAPI(title="Klondike Solitaire RL")


@app.get("/api/state")
def get_state() -> dict:
    if SESSION.env.state is None:
        SESSION.env.reset(seed=random.randrange(1_000_000))
    return _serialize(SESSION)


@app.post("/api/new")
def new_game(req: NewGame) -> dict:
    variant = req.variant if req.variant in ("draw1", "draw3") else DRAW1
    if SESSION.env.variant != variant or SESSION.env.obs_variant != "compact":
        SESSION.env = KlondikeEnv(variant=variant, obs_variant="compact", train_seeds=False)
    seed = req.seed if req.seed is not None else random.randrange(1_000_000)
    SESSION.env.reset(seed=seed)
    SESSION.history = []
    SESSION.undo = []
    return _serialize(SESSION)


@app.get("/api/legal")
def legal() -> dict:
    st = SESSION.env.state
    if st is None:
        raise HTTPException(409, "no game in progress")
    return {"legal": [_move_desc(st, m) for m in SESSION.env._legal()]}


def _step(sess: Session, action: int) -> dict:
    """Apply one action id through the env; attach a UI description."""
    st = sess.env.state
    assert st is not None
    desc: dict[str, Any] = {"action": action, "concede": action == ACTION_CONCEDE}
    if action != ACTION_CONCEDE:
        desc.update(_move_desc(st, action_to_move(action, st)))
    sess.undo.append(copy.deepcopy(sess.env))
    if len(sess.undo) > 600:
        sess.undo.pop(0)
    _, reward, terminated, truncated, _info = sess.env.step(action)
    sess.history.append(action)
    return {
        "applied": desc,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        **_serialize(sess),
    }


@app.post("/api/move")
def play_move(req: MoveReq) -> dict:
    st = SESSION.env.state
    if st is None or SESSION.env._outcome is not None:
        raise HTTPException(409, "no game in progress — start a new deal")
    try:
        mv = Move(MoveKind(req.kind), src=req.src, dst=req.dst, run_start=req.run_start)
        action = move_to_action(mv)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, f"bad move descriptor: {e}") from e
    if action not in {m["action"] for m in _serialize(SESSION)["legal"]}:
        raise HTTPException(400, "illegal move in current state")
    return _step(SESSION, action)


@app.post("/api/concede")
def concede() -> dict:
    if SESSION.env.state is None or SESSION.env._outcome is not None:
        raise HTTPException(409, "no game in progress")
    return _step(SESSION, ACTION_CONCEDE)


@app.post("/api/undo")
def undo() -> dict:
    if not SESSION.undo:
        raise HTTPException(409, "nothing to undo")
    SESSION.env = SESSION.undo.pop()
    SESSION.history.pop()
    return _serialize(SESSION)


@app.get("/api/agent/policies")
def policies() -> dict:
    return {
        "policies": [
            {"id": k, "label": v["label"], "obs": v["env"].get("obs_variant", "compact")}
            for k, v in _AGENT_POLICIES.items()
        ]
    }


def _agent_env(sess: Session, policy_key: str) -> KlondikeEnv:
    """Rebuild the policy's view by replaying this game's action history into
    a fresh env with the policy's own obs variant — identical to eval."""
    spec = _AGENT_POLICIES[policy_key]
    env = KlondikeEnv(
        variant=sess.env.variant,
        train_seeds=False,
        **spec["env"],
    )
    env.reset(seed=sess.env.deal_seed)
    for a in sess.history:
        env.step(a)
    return env


@app.post("/api/agent/hint")
def agent_hint(req: AgentReq) -> dict:
    """Suggest a move without applying it (heuristic is instant; NN policies
    are also cheap enough)."""
    if SESSION.env.state is None or SESSION.env._outcome is not None:
        raise HTTPException(409, "no game in progress")
    if req.policy not in _AGENT_POLICIES:
        raise HTTPException(404, f"unknown policy {req.policy!r}")
    policy = SESSION.agent(req.policy)
    env = _agent_env(SESSION, req.policy)
    action = policy.act(env)
    st = SESSION.env.state
    desc: dict[str, Any] = {"action": action, "concede": action == ACTION_CONCEDE}
    if action != ACTION_CONCEDE:
        desc.update(_move_desc(st, action_to_move(action, st)))
    return {"suggested": desc}


@app.post("/api/agent/step")
def agent_step(req: AgentReq) -> dict:
    """One greedy/policy move on the live game — watch mode calls this on a
    timer."""
    if SESSION.env.state is None or SESSION.env._outcome is not None:
        raise HTTPException(409, "game over — start a new deal")
    if req.policy not in _AGENT_POLICIES:
        raise HTTPException(404, f"unknown policy {req.policy!r}")
    policy = SESSION.agent(req.policy)
    env = _agent_env(SESSION, req.policy)
    action = policy.act(env)
    return _step(SESSION, action)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
