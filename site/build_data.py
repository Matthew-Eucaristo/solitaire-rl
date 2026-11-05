"""Build-time data for the GitHub Pages site (stdlib only, no model code).

Reads the committed per-deal eval JSONs in results/ and writes deals.js for
the page's 1000-deal field. Every count the page shows is recomputed here and
checked against the numbers README.md / RESULTS.md report, so the build fails
instead of publishing a figure the results do not support.

    python3 -I site/build_data.py OUTDIR [--media mp4|gif]
    python3 -I site/build_data.py --list-recordings
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
N_DEALS = 1000

# Recordings the page shows (recordings/<name>.gif); build.sh converts these only.
RECORDINGS = {
    "ppo_win_seed47": (47, "MaskablePPO, compact POMDP", "win"),
    "heuristic_win_seed326": (326, "Heuristic baseline", "win"),
    "champ_win_seed47": (47, "Single-model champion, compact + hint", "win"),
    "dqn_win_seed47": (47, "Masked DQN, compact", "win"),
    "x_portfolio_win2_seed105_divA": (105, "Portfolio member divA", "win"),
    "sparse_win_seed105_champ_loses": (105, "Sparse-reward lineage", "win"),
    "dqn_concede_seed14": (14, "Masked DQN, compact", "concede"),
    "champ_lose_seed0": (0, "Single-model champion, compact + hint", "no_progress_cycle"),
}

OUTCOME_CODES = {
    "win": "W",
    "no_progress_cycle": "C",
    "no_progress_idle": "I",
    "concede": "K",
    "truncated": "T",
    "no_moves": "N",
}

# (id, file, label, observation, expected wins as README.md / RESULTS.md report)
SINGLE = [
    ("heuristic", "m1_heuristic_draw1", "Heuristic", "rule-based", 414),
    ("dqn", "m2_dqn", "Masked DQN, 250k steps", "compact", 58),
    ("ppo", "m3_ppo_6m", "MaskablePPO, 6M steps", "compact POMDP", 118),
    ("perfect", "m4_ppo_perfect", "MaskablePPO, 6M, perfect info", "compact, perfect info", 177),
    ("random", "m1_random_draw1", "Random legal move", "none", 0),
    ("heuristic3", "m1_heuristic_draw3", "Heuristic, draw-3", "rule-based", 118),
]
EXPECTED_UNIONS = {"canon9": 579, "p26": 625, "p26h": 645}


def load(name: str) -> dict:
    data = json.loads((RESULTS / f"{name}.json").read_text())
    rows = sorted(data["results"], key=lambda r: r["seed"])
    if [r["seed"] for r in rows] != list(range(N_DEALS)):
        raise SystemExit(f"{name}: expected seeds 0..{N_DEALS - 1}")
    wins = sum(r["outcome"] == "win" for r in rows)
    if wins != data["wins"]:
        raise SystemExit(f"{name}: {wins} win rows but wins={data['wins']}")
    data["results"] = rows
    return data


def member_label(spec: str) -> str:
    """'ppo:runs/ppo_amp1_9m/ppo_1600000.zip' -> 'ppo_amp1_9m @ 1.6M'."""
    if ":" not in spec:
        return spec
    path = Path(spec.split(":", 1)[1])
    stem = path.stem
    if path.parent.name == "checkpoints":
        return stem
    step = re.fullmatch(r"ppo_(\d+)", stem)
    if step:
        return f"{path.parent.name} @ {int(step.group(1)) / 1e6:g}M"
    return path.parent.name


def check(label: str, got: int, want: int) -> None:
    if got != want:
        raise SystemExit(f"{label}: results give {got} wins, the docs say {want}")


def build(outdir: Path, media: str) -> None:
    layers = []
    wins_of: dict[str, set[int]] = {}
    for lid, fname, label, obs, want in SINGLE:
        d = load(fname)
        codes = "".join(OUTCOME_CODES.get(r["outcome"], "?") for r in d["results"])
        wins_of[lid] = {r["seed"] for r in d["results"] if r["outcome"] == "win"}
        check(lid, len(wins_of[lid]), want)
        layers.append({
            "id": lid,
            "label": label,
            "obs": obs,
            "variant": d["env"]["variant"],
            "source": f"results/{fname}.json",
            "out": codes,
            "moves": [r["moves"] for r in d["results"]],
        })

    # Declared portfolios: winning member per deal, as attributed by run_eval.
    def portfolio(name: str) -> tuple[list[str], list[int]]:
        d = load(name)
        members = [member_label(m) for m in d["portfolio"]["members"]]
        idx = [r["member_index"] if r["outcome"] == "win" else -1 for r in d["results"]]
        return members, idx

    c9_members, c9_idx = portfolio("portfolio9_canonical")
    p25_members, p25_idx = portfolio("portfolio25_rl")
    qrdqn = load("qrdqn_400k")
    q_wins = {r["seed"] for r in qrdqn["results"] if r["outcome"] == "win"}

    # x26 = the 25-lineage run plus the QRDQN member (RESULTS.md union math).
    p26_members = [*p25_members, "qrdqn_400000"]
    p26_idx = [
        i if i >= 0 else (len(p25_members) if s in q_wins else -1) for s, i in enumerate(p25_idx)
    ]
    p26h_members = [*p26_members, "heuristic"]
    p26h_idx = [
        i if i >= 0 else (len(p26_members) if s in wins_of["heuristic"] else -1)
        for s, i in enumerate(p26_idx)
    ]
    unions = [
        ("canon9", "Portfolio of 9 committed checkpoints", c9_members, c9_idx,
         "results/portfolio9_canonical.json"),
        ("p26", "Portfolio of 26 RL lineages", p26_members, p26_idx,
         "results/portfolio25_rl.json + results/qrdqn_400k.json"),
        ("p26h", "Portfolio of 26 + heuristic", p26h_members, p26h_idx,
         "the above + results/m1_heuristic_draw1.json"),
    ]
    for lid, label, members, idx, source in unions:
        check(lid, sum(i >= 0 for i in idx), EXPECTED_UNIONS[lid])
        layers.append({
            "id": lid,
            "label": label,
            "obs": "compact + hint",
            "variant": "draw1",
            "source": source,
            "members": members,
            "member": idx,
        })

    html = (Path(__file__).parent / "index.html").read_text()
    for name in set(re.findall(r'data-rec="([^"]+)"', html)):
        if name not in RECORDINGS:
            raise SystemExit(f"index.html shows {name}, which RECORDINGS does not list")
    recs: dict[int, list[dict]] = {}
    for name, (seed, label, outcome) in RECORDINGS.items():
        recs.setdefault(seed, []).append({"name": name, "label": label, "outcome": outcome})

    payload = {"n": N_DEALS, "media": media, "layers": layers, "recordings": recs}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "deals.js").write_text(
        "// Generated by site/build_data.py from the per-deal JSONs in results/.\n"
        "window.SRL = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    )
    if media == "gif":  # no ffmpeg: swap each <video> for its GIF
        html = re.sub(
            r'<video[^>]*data-rec="([^"]+)"[^>]*aria-label="([^"]*)"[^>]*>.*?</video>',
            r'<img class="rec" src="rec/\1.gif" alt="\2" width="466" height="380" loading="lazy">',
            html,
            flags=re.S,
        )
        (outdir / "index.html").write_text(html)


def main(argv: list[str]) -> None:
    if argv[:1] == ["--list-recordings"]:
        print("\n".join(RECORDINGS))
        return
    if not argv:
        raise SystemExit(__doc__)
    media = argv[argv.index("--media") + 1] if "--media" in argv else "mp4"
    build(Path(argv[0]), media)


if __name__ == "__main__":
    main(sys.argv[1:])
