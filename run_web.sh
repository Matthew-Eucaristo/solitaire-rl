#!/usr/bin/env bash
# Run the Klondike web app locally: `uv run` syncs deps into .venv on first use.
# Then open http://localhost:8080
set -euo pipefail
cd "$(dirname "$0")"
exec uv run uvicorn solitaire_rl.webapi:app --host 127.0.0.1 --port "${PORT:-8080}"
