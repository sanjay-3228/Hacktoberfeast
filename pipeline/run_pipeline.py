"""One function turns a video into a job folder.

    from run_pipeline import process_video
    process_video("/kaggle/input/demo-videos/clip1.mp4", "demo1")

CLI:  python run_pipeline.py clip.mp4 demo1 [--no-faces] [--no-llm] [--no-embed]

Every stage saves its output and is skipped if the file already exists, so a
crash never forces a full rerun (delete a file to redo that stage).
"""
import argparse
from pathlib import Path

import config
from stages.diarize import diarize
from stages.faces import track_faces
from stages.index import build_index
from stages.ingest import ingest
from stages.llm import analyze
from stages.match import match_speakers_to_faces
from stages.merge import merge
from stages.result import build_result, validate_result
from stages.transcribe import transcribe
from utils import load_json, save_json, timed


def process_video(video_path: str, job_id: str, use_faces: bool = True,
                  use_llm: bool = True, embed: bool = True,
                  num_speakers: int | None = None) -> Path:
    job_dir = config.JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    with timed("1 ingest"):
        info = ingest(video_path, job_dir)
    with timed("2 transcribe"):
        transcript = transcribe(info["audio"], job_dir / "transcript.json")
    with timed("3 diarize"):
        turns = diarize(info["audio"], job_dir / "diarization.json", num_speakers)
    with timed("4 merge"):
        messages = merge(transcript, turns)

    face_map = {}
    if use_faces:
        with timed("5 faces"):
            tracks = track_faces(info["video"], job_dir, job_dir / "faces_tracks.json")
        with timed("6 match speakers to faces"):
            face_map = match_speakers_to_faces(messages, tracks, info["duration"], job_dir)

    with timed("7 llm analysis"):
        if use_llm:
            analysis = analyze(messages, info["duration"])
        else:
            analysis = {"speaker_names": {}, "title": "", "topic": "",
                        "chapters": [{"title": "Full video", "start": 0.0,
                                      "end": info["duration"], "summary": ""}],
                        "takeaways": []}

    with timed("8 index"):
        chunks_path = job_dir / "chunks.json"
        if not chunks_path.exists():
            save_json(build_index(messages, embed=embed), chunks_path)

    with timed("9 result"):
        result = build_result(job_id, info["duration"], messages, face_map, analysis)
        validate_result(result)
        save_json(result, job_dir / "result.json")

    print(f"\nDone. Job folder: {job_dir.resolve()}")
    return job_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("job_id")
    ap.add_argument("--no-faces", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--num-speakers", type=int, default=None)
    a = ap.parse_args()
    process_video(a.video, a.job_id, not a.no_faces, not a.no_llm,
                  not a.no_embed, a.num_speakers)
