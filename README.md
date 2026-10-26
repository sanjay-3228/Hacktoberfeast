# Video Knowledge Extractor

Turns a short video into: speaker-labelled group-chat transcript (including
off-camera speakers), searchable chapters, key takeaways, and source-linked
answers that jump to the exact moment in the video.

## Architecture (Plan A)

```
 [GPU: Kaggle/Colab]                      [Azure: CPU]
 pipeline/  --process_video()-->  jobs/<id>/  --copy-->  backend/ (FastAPI)  <--  frontend/
 (Whisper, pyannote, faces, LLM)  result.json            reads job folders,
                                  chunks.json            search + Q&A + video
                                  video.mp4, faces/
```

Heavy models run once on a free GPU. The Azure server only serves finished
job folders, plus live search and Q&A (light enough for a CPU VM).

## Folders

| Folder       | Owner idea         | What it is |
|--------------|--------------------|------------|
| `pipeline/`  | ML teammate        | Kaggle/Colab code: video -> job folder |
| `backend/`   | Backend teammate   | FastAPI server that serves job folders |
| `frontend/`  | Frontend teammate  | UI (React + Vite + Tailwind suggested) |
| `docs/`      | everyone           | The data contract and API. Read first. |
| `scripts/`   | everyone           | Helper scripts (run, sync to Azure) |

## Read this first

1. `docs/CONTRACT.md`: exact shape of `result.json` and `chunks.json`.
   The pipeline writes it, the backend serves it, the frontend reads it.
   Change it only after telling the team.
2. `docs/API.md`: backend endpoints.
3. `docs/TASKS.md`: task split and build order.

## Quick start (backend with the mock job, no GPU needed)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
# open http://localhost:8000/docs
```

The mock job `backend/jobs/demo/` lets the frontend start before the
pipeline works. Try `http://localhost:8000/jobs/demo/result`.

## Quick start (pipeline)

See `pipeline/README.md` (Kaggle setup + how to run).
