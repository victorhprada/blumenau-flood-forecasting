#!/usr/bin/env bash
# Start the FastAPI inference server for the flood forecast model.
# Run from the project root.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate
export PYTHONPATH="vendor/flood-forecasting:."

echo "Starting FastAPI on http://localhost:8000"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
