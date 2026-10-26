# Pipeline (runs on Kaggle/Colab GPU)

`process_video(video_path, job_id)` turns one video into `jobs/<job_id>/`
(see `../docs/CONTRACT.md`). Every stage saves its output file and is skipped
if that file already exists.

| # | File | What it does |
|---|------|--------------|
| 1 | `stages/ingest.py` | web-playable `video.mp4` + 16 kHz `audio.wav` (ffmpeg) |
| 2 | `stages/transcribe.py` | faster-whisper, word timestamps |
| 3 | `stages/diarize.py` | pyannote: who spoke when |
| 4 | `stages/merge.py` | words + speakers -> messages (S1, S2, ...) |
| 5 | `stages/faces.py` | MediaPipe face tracks + mouth movement |
| 6 | `stages/match.py` | speaker <-> face matching, off-camera detection |
| 7 | `stages/llm.py` | names, title, topic, chapters, takeaways |
| 8 | `stages/index.py` | search chunks + embeddings |
| 9 | `stages/result.py` | builds and validates `result.json` |

## Run on Kaggle

1. Verify your phone number, enable **GPU** and **Internet** in the notebook settings.
2. Add Secrets: `HF_TOKEN` (Hugging Face read token) and, for the LLM step,
   `LLM_API_KEY` (plus `LLM_BASE_URL` / `LLM_MODEL`, or the Azure ones).
3. On Hugging Face, accept the terms of the pyannote models (while logged in).
4. Upload this project folder (or just `pipeline/`) as a private Kaggle
   Dataset, and your demo videos as another one. Attach both to the notebook.
5. Open `kaggle_notebook.ipynb` (included), run the cells top to bottom.
6. The last cell zips `jobs/`. Download it from the notebook's Output tab and
   unzip into `backend/jobs/`.

## Run locally (tiny test, CPU)

```bash
cd pipeline
pip install -r requirements.txt
WHISPER_MODEL=small HF_TOKEN=... python run_pipeline.py clip.mp4 test1 --no-llm --no-embed
```

## Tests (no models needed)

```bash
python tests/test_logic.py
```

## Tuning (all in `config.py`)

- `ON_CAMERA_THRESHOLD`: raise it if off-camera speakers get a wrong face,
  lower it if on-camera speakers are marked off-camera.
- `MERGE_GAP_SEC`: how close same-speaker messages must be to merge.
- `CHUNK_SECONDS`: search chunk length.

## Known weak spots (test with a messy clip early)

- Overlapping speech confuses diarization. If you know the speaker count,
  pass `num_speakers=2`.
- Mouth-movement matching is the MVP. Upgrade path: replace
  `_mouth_activity` in `stages/match.py` with LR-ASD or TalkNet scores.
- Side-on faces and masks reduce face tracking quality.
- Clips over ~20 minutes need the chunked LLM path (see TODO in `stages/llm.py`).
