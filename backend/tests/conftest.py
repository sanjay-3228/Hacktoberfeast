"""Pytest bootstrap: expose the backend modules to the test suite.

Run with ``python -m pytest tests -q`` from ``backend/`` (or any directory;
the path insertion below makes ``import main`` work regardless of cwd).
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
