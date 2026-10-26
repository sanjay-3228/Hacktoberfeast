# Backend API (FastAPI)

Interactive docs: `http://localhost:8000/docs`

| Method | Path | Returns |
|--------|------|---------|
| GET | `/health` | `{"ok": true}` |
| GET | `/jobs` | list of `{job_id, title, topic, duration}` |
| POST | `/jobs` | multipart upload -> `{"job_id": "...", "status": "processing"}` |
| GET | `/jobs/{id}/status` | `{"status": "processing"|"done"|"error", "progress": 0|100}` |
| GET | `/jobs/{id}/result` | the full `result.json` |
| GET | `/jobs/{id}/video` | the video file (supports seeking) |
| GET | `/jobs/{id}/faces/{file}` | avatar image, e.g. `S1.jpg` |
| GET | `/jobs/{id}/search?q=...&k=5` | `{"results": [{start, end, speaker, text, score}]}` |
| POST | `/jobs/{id}/ask` | body `{"question": "..."}` -> `{"answer": "...", "sources": [{start, end, speaker, text, score}]}` |
| GET | `/jobs/{id}/report` | Markdown report (speakers, chapters, takeaways, transcript) |

## Local processing upload

`POST /jobs` accepts a multipart form with:

- `file`: one video or audio file (`mp4`, `mov`, `mkv`, `avi`, `mp3`, `wav`, or `m4a`)
- `title`: optional display title
- `use_faces`: optional boolean, default `false`
- `use_llm`: optional boolean, default `true`
- `embed`: optional boolean, default `false`
- `num_speakers`: optional integer hint for diarization

The backend saves the upload and invokes `pipeline.process_video` in a background
worker. Poll `/jobs/{id}/status` until it is `done`, then load `/result`. This
local-processing route requires the pipeline dependencies to be installed in the
same Python environment as the backend. A failed pipeline returns `status: error`
with a diagnostic `detail`.

## Frontend notes

- Seek the video with `video.currentTime = source.start`.
- Highlight the current chapter and chat bubble with the `timeupdate` event.
- Errors: 404 `{"detail": "Job not found"}`.
- Answers carry `sources`; render each as a clickable chip like `02:14 Priya`.

## Grounded Q&A and reports

- `/search` retrieves the most relevant timestamped chunks using their stored
  sentence embeddings, with keyword search as a fallback.
- `/ask` passes those chunks to the configured Groq model (or supported
  OpenAI-compatible/Azure fallback), returns a concise answer, and includes
  source excerpts and timestamps for citation and video seeking.
- `/report` generates a detailed conversation analysis from the full transcript
  when Groq is configured, appends the transcript, and caches the Markdown as
  `report.md` in the job folder. Set `GROQ_API_KEY` in the repository `.env`;
  `GROQ_MODEL` is optional and defaults to `openai/gpt-oss-120b`.
