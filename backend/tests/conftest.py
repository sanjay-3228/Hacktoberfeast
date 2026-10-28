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

# On Windows, socket.socketpair can randomly pick an ephemeral port in an OS
# excluded port range and fail with PermissionError (WinError 10013).
# Retrying with another ephemeral port avoids spurious TestClient setup errors.
if sys.platform == "win32":
    import socket

    _orig_socketpair = getattr(socket, "socketpair", None)
    if _orig_socketpair is not None:
        def _resilient_socketpair(*args, **kwargs):
            for attempt in range(10):
                try:
                    return _orig_socketpair(*args, **kwargs)
                except PermissionError:
                    if attempt == 9:
                        raise

        socket.socketpair = _resilient_socketpair

