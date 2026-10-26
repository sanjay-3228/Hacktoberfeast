"""Stage 5: detect and track faces, record mouth movement per track.

Uses MediaPipe FaceMesh (CPU is fine). For each sampled frame we get every
face's bounding box and "mouth openness" (inner-lip distance / face height).
Faces are linked across frames into tracks by nearest centroid.

Output (faces_tracks.json):
[{"track": 0, "samples": [{"t": 1.2, "mouth": 0.03}, ...], "crop": "faces/track_0.jpg"}]
"""
from pathlib import Path

import config
from utils import load_json, save_json

UPPER_LIP, LOWER_LIP = 13, 14  # inner lip landmarks in FaceMesh


def track_faces(video_path: Path, job_dir: Path, out_path: Path) -> list[dict]:
    if out_path.exists():
        return load_json(out_path)

    import cv2
    import mediapipe as mp
    import numpy as np

    faces_dir = job_dir / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(fps / config.FACE_FPS)), 1)

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=6, refine_landmarks=False,
        min_detection_confidence=0.5,
    )

    tracks: list[dict] = []   # {"id", "last_xy", "last_t", "samples", "best_area", "best_crop"}
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            t = frame_idx / fps
            h, w = frame.shape[:2]
            res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            used: set[int] = set()
            for lm in (res.multi_face_landmarks or []):
                xs = np.array([p.x for p in lm.landmark]) * w
                ys = np.array([p.y for p in lm.landmark]) * h
                x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
                face_h = max(y1 - y0, 1.0)
                mouth = abs(lm.landmark[UPPER_LIP].y - lm.landmark[LOWER_LIP].y) * h / face_h
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

                # nearest still-open track
                best, best_d = None, 0.15 * w
                for tr in tracks:
                    if tr["id"] in used or t - tr["last_t"] > config.TRACK_MAX_GAP_SEC:
                        continue
                    d = np.hypot(cx - tr["last_xy"][0], cy - tr["last_xy"][1])
                    if d < best_d:
                        best, best_d = tr, d
                if best is None:
                    best = {"id": len(tracks), "samples": [], "best_area": 0, "best_crop": None}
                    tracks.append(best)
                used.add(best["id"])
                best["last_xy"], best["last_t"] = (cx, cy), t
                best["samples"].append({"t": round(t, 2), "mouth": round(float(mouth), 4)})

                area = (x1 - x0) * face_h
                if area > best["best_area"]:
                    pad = 0.25 * face_h
                    crop = frame[int(max(y0 - pad, 0)):int(min(y1 + pad, h)),
                                 int(max(x0 - pad, 0)):int(min(x1 + pad, w))]
                    if crop.size:
                        best["best_area"], best["best_crop"] = area, crop
        frame_idx += 1
    cap.release()

    out = []
    for tr in tracks:
        if len(tr["samples"]) < config.MIN_TRACK_SAMPLES or tr["best_crop"] is None:
            continue
        crop_path = faces_dir / f"track_{tr['id']}.jpg"
        cv2.imwrite(str(crop_path), cv2.resize(tr["best_crop"], (160, 160)))
        out.append({"track": tr["id"], "samples": tr["samples"],
                    "crop": f"faces/track_{tr['id']}.jpg"})
    save_json(out, out_path)
    return out
