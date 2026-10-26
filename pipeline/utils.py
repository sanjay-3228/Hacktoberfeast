import gc
import json
import time
from contextlib import contextmanager
from pathlib import Path


def save_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def free_gpu():
    """Call after finishing with a big model."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@contextmanager
def timed(name: str):
    t0 = time.time()
    print(f"[{name}] started")
    yield
    print(f"[{name}] done in {time.time() - t0:.1f}s")
