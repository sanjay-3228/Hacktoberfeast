# Backend development guide

The backend is a small FastAPI service. It does not process videos or upload files in Plan A. The pipeline creates a completed job folder, and this service reads that folder to provide the API, search, Q&A, video playback, and reports.

## Prerequisites

- Python 3.11 or newer
- Git
- A completed job folder, or the included `backend/jobs/demo` fixture
- Optional: an OpenAI-compatible API key or Azure OpenAI credentials for LLM-backed Q&A
- Optional: `sentence-transformers` for semantic search

## Local setup on Windows

From the repository root in PowerShell:

```powershell
python -m venv backend\.venv
.\backend\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

If PowerShell blocks activation, use the venv interpreter directly instead:

```powershell
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv backend/.venv
. backend/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

## Configuration

Copy the repository template and edit only the values needed for your local run:

```powershell
Copy-Item .env.example .env
```

Important settings:

- `JOBS_DIR`: directory containing job folders. The default in `.env.example` is `./jobs`; when running from `backend/`, use `JOBS_DIR=./jobs` or an absolute path.
- `CORS_ORIGINS`: comma-separated frontend origins, for example `http://localhost:5173`. Use `*` only for local development.
- `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`: optional OpenAI-compatible Q&A configuration.
- `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`, and `AZURE_OPENAI_API_VERSION`: optional Azure OpenAI configuration.
- `EMBED_MODEL`: sentence-transformers model name used when chunks contain embeddings.

Never commit `.env` or paste API keys into source files. The backend can run without an LLM key; `/ask` then returns the best matching excerpt in development mode.

## Run the API

From the `backend/` directory with the virtual environment active:

```powershell
Set-Location backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Health check: `http://localhost:8000/health`

From the repository root, the helper script also starts the service on macOS/Linux:

```bash
./scripts/run_backend.sh
```

## Smoke test with the demo job

With the server running, check the service and mock job:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/jobs
Invoke-RestMethod http://localhost:8000/jobs/demo/result
Invoke-RestMethod "http://localhost:8000/jobs/demo/search?q=privacy&k=3"
Invoke-WebRequest http://localhost:8000/jobs/demo/report
```

Expected behavior:

- `/health` returns `{"ok": true}`.
- `/jobs` includes the `demo` job.
- `/jobs/demo/result` matches the schema in `docs/CONTRACT.md`.
- Search uses semantic similarity when embeddings and `sentence-transformers` are available; otherwise it falls back to keyword matching.
- `/jobs/demo/ask` accepts JSON such as `{"question":"How is timetable privacy protected?"}`.

To test the POST endpoint from PowerShell:

```powershell
$body = @{ question = "How is timetable privacy protected?" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/jobs/demo/ask -ContentType "application/json" -Body $body
```

## Adding pipeline output

Copy a completed pipeline job into `backend/jobs/<job_id>/`. The folder must contain at least:

```text
backend/jobs/<job_id>/
  result.json
  chunks.json                 # optional for basic listing, required for search/Q&A quality
  video.mp4                   # required for video playback
  faces/S1.jpg                # optional avatar files referenced by result.json
```

Keep `job_id` limited to letters, digits, `_`, and `-`. Do not copy raw videos, audio, model caches, or temporary files into Git. Validate the generated `result.json` against `docs/CONTRACT.md` before copying it to the backend.

For the normal Plan A flow, download the zipped jobs produced by the Kaggle notebook, then extract the job directory under `backend/jobs/`.

## Optional semantic search

The backend works without embeddings. To enable semantic search locally:

```powershell
python -m pip install "sentence-transformers>=3.0"
```

The first semantic-search request may download the model configured by `EMBED_MODEL`, so Internet access is required. If the dependency or model is unavailable, the service should use keyword matching instead.

## Tests and checks

There are currently no backend-specific test files. Before opening a pull request:

1. Start the API with the demo job.
2. Check `/health`, `/jobs`, `/jobs/demo/result`, `/jobs/demo/search`, `/jobs/demo/ask`, and `/jobs/demo/report`.
3. Confirm video and avatar URLs return files when those files exist.
4. Confirm invalid job IDs return HTTP 404 and do not access files outside the job directory.
5. Run the pipeline pure-logic tests separately from `pipeline/`:

```powershell
Set-Location pipeline
python tests\test_logic.py
```

Keep API and data-contract changes synchronized with `docs/API.md`, `docs/CONTRACT.md`, and the mock job.

## Docker

Build and run from `backend/`:

```powershell
Set-Location backend
docker build -t vke-backend .
docker run --rm -p 8000:8000 -v "${PWD}\jobs:/app/jobs" vke-backend
```

Pass production configuration explicitly or through a secret-managed environment file. Do not bake `.env` or API keys into the image.

## Troubleshooting

- **`ModuleNotFoundError`**: activate `backend/.venv` or invoke its Python executable, then reinstall `backend/requirements.txt`.
- **No jobs listed**: confirm `JOBS_DIR` points to the directory containing job subfolders with `result.json`.
- **404 for video or avatar**: check the file exists and that the relative path in `result.json` matches the actual file under the job folder.
- **Search returns weak results**: confirm `chunks.json` exists; install `sentence-transformers` only when embeddings are present and the model can be downloaded.
- **Q&A returns an excerpt instead of an LLM answer**: configure the LLM variables in `.env`; excerpt fallback is expected when no provider is configured.
- **Frontend CORS error**: set `CORS_ORIGINS` to the exact frontend origin, such as `http://localhost:5173`, and restart the API.
- **Port 8000 is busy**: start Uvicorn on another port, for example `--port 8001`, and update the frontend API URL.
