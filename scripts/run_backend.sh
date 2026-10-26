#!/usr/bin/env bash
# Run the API locally:  ./scripts/run_backend.sh
cd "$(dirname "$0")/../backend" || exit 1
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
