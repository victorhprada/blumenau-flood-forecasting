#!/usr/bin/env bash
# Start the Streamlit dashboard.
# Expects the FastAPI server to be running on port 8000.
# Run from the project root.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate
export PYTHONPATH="vendor/flood-forecasting:."

echo "Starting Streamlit on http://localhost:8501"
streamlit run app/Home.py --server.port 8501
