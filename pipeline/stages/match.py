"""Stage 6: match each speaker to a face track (or mark off-camera).

Idea: a face whose mouth moves exactly when a speaker talks is that speaker.
We compare each track's mouth-movement signal with each speaker's speech
signal over the whole video (this "votes" across all of the speaker's turns),
then assign one face per speaker greedily by best score. Scores below
ON_CAMERA_THRESHOLD mean the speaker is off camera.

This is the MVP. For better accuracy, replace `_mouth_activity` with an
active-speaker model (LR-ASD / TalkNet) that outputs a per-frame speaking score.
"""
import shutil
from pathlib import Path

import numpy as np

import config


def _speech_vector(messages: list[dict], speaker: str, times: np.ndarray) -> np.ndarray:
    v = np.zeros_like(times)
    for m in messages:
        if m["speaker"] == speaker:
            v[(times >= m["start"]) & (times <= m["end"])] = 1.0
    return v


def _mouth_activity(track: dict, times: np.ndarray):
    """Rolling std of mouth openness. Returns (signal, present_mask)."""
    ts = np.array([s["t"] for s in track["samples"]])
    mv = np.array([s["mouth"] for s in track["samples"]])
    present = np.zeros_like(times, dtype=bool)
    sig = np.zeros_like(times)
    idx = np.searchsorted(times, ts)
    idx = np.clip(idx, 0, len(times) - 1)
    sig[idx] = mv
    present[idx] = True

    win = max(int(0.6 * config.FACE_FPS), 2)
    act = np.zeros_like(sig)
    for i in range(len(sig)):
        lo = max(0, i - win // 2)
        hi = min(len(sig), i + win // 2 + 1)
        seg = sig[lo:hi][present[lo:hi]]
        act[i] = seg.std() if len(seg) > 1 else 0.0
    return act, present


def _corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    a, b = a[mask], b[mask]
    if len(a) < 5 or a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def match_speakers_to_faces(messages: list[dict], tracks: list[dict],
                            duration: float, job_dir: Path) -> dict:
    """Returns {"S1": {"on_camera": True, "track": 0, "score": 0.61, "avatar": "faces/S1.jpg"}, ...}"""
    speakers = sorted({m["speaker"] for m in messages})
    times = np.arange(0, duration, 1.0 / config.FACE_FPS)

    scores: dict[tuple[str, int], float] = {}
    for tr in tracks:
        act, present = _mouth_activity(tr, times)
        for s in speakers:
            scores[(s, tr["track"])] = _corr(act, _speech_vector(messages, s, times), present)

    result = {s: {"on_camera": False, "track": None, "score": 0.0, "avatar": None}
              for s in speakers}
    used_tracks: set[int] = set()
    used_speakers: set[str] = set()
    crops = {tr["track"]: tr["crop"] for tr in tracks}

    # greedy one-to-one assignment, best score first
    for (s, t), sc in sorted(scores.items(), key=lambda kv: -kv[1]):
        if sc < config.ON_CAMERA_THRESHOLD:
            break
        if s in used_speakers or t in used_tracks:
            continue
        dst = f"faces/{s}.jpg"
        shutil.copy(job_dir / crops[t], job_dir / dst)
        result[s] = {"on_camera": True, "track": t, "score": round(sc, 3), "avatar": dst}
        used_speakers.add(s)
        used_tracks.add(t)
    return result
