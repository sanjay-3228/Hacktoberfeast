# Synchra — EchoSight Forensics-connected frontend

React + Vite frontend based on the supplied design. The upload flow is connected to the existing FastAPI backend.

## Run locally

Requirements: Node.js 18+ and npm.

1. Start the FastAPI backend on `http://127.0.0.1:8000`.
2. Create a `.env` file in this frontend directory using `.env.example` as a template:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

3. Install and run the frontend:

```bash
npm install
npm run dev
```

Open the local URL shown by Vite (usually `http://localhost:5173`). If the browser reports a CORS error, configure the backend to allow the exact frontend origin, usually `http://localhost:5173`.

## Connected API workflow

- `POST /api/jobs` — uploads one video as multipart form field `video`.
- `GET /api/jobs/{job_id}/status` — frontend polls processing status.
- `GET /jobs/{job_id}/result` — loads generated analysis after processing.
- `GET /jobs/{job_id}/video` — video playback.
- `GET /jobs/{job_id}/search?q=...&k=5` — transcript/chunk search.
- `POST /jobs/{job_id}/ask` — asks a question using JSON `{ "question": "..." }`.
- `GET /jobs/{job_id}/report` — downloads the generated Markdown report.

The frontend expects the backend routes and response shapes to match the existing project. It displays real results returned by the API for uploaded jobs; the built-in example notebooks remain illustrative sample content.

## Deployment

Set `VITE_API_BASE_URL` to the deployed FastAPI base URL in your frontend hosting provider before building. Vite environment variables are embedded into the client build, so this value is not a secret. Keep model/API keys only in the backend environment.

The frontend currently supports one video per notebook upload. Public recording URL import and multi-file batch processing are not connected.

## Build

```bash
npm run build
npm run preview
```
