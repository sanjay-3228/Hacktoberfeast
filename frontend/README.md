hy d# Synchra — AI Research Notebook frontend

A responsive React + Vite frontend for the Video Knowledge Extractor backend. It includes the backend job library, search and sorting controls, drag-and-drop upload, processing status, and a notebook details panel.

## Run locally

Requirements: Node.js 18+ and npm.

```bash
npm install
npm run dev
```

Open the local URL shown by Vite (usually http://localhost:5173).

Build for production:

```bash
npm run build
npm run preview
```

## Backend integration

Set the backend URL with `VITE_API_URL` (defaults to `http://localhost:8000`).
On startup the app loads `GET /jobs`. Creating a notebook with a local recording
sends the first selected file to `POST /jobs`, polls its status, then loads the
completed result and video from the backend. The upload currently starts with
faces and embeddings disabled so local processing is easier to test; enable those
pipeline options later when the model environment is ready.

When the backend is unavailable, the app shows an empty library and a connection
notice. There is no local or hardcoded notebook fallback.

## Stack

- React
- Vite
- Lucide React icons
- CSS with responsive layouts
