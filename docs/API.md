# Backend API (FastAPI)

Interactive docs: `http://localhost:8000/docs`

| Method | Path | Returns |
|--------|------|---------|
| GET | `/health` | `{"ok": true}` |
| GET | `/jobs` | list of `{job_id, title, topic, duration}` |
| GET | `/jobs/{id}/status` | `{"status": "done", "progress": 100}` or `{"status": "processing", "progress": 0}` |
| GET | `/jobs/{id}/result` | the full `result.json` (missing `job_id`/`title` are filled from the folder name / empty string) |
| GET | `/jobs/{id}/search?q=...&k=3` | `{"results": [{start, end, speaker, text, score}]}` — semantic when embeddings + `sentence-transformers` are available, keyword otherwise; only positive-scoring chunks are returned |
| POST | `/jobs/{id}/ask` | body `{"question": "..."}` → `{"answer": "...", "sources": [{start, end, speaker, text, score}]}` |
| GET | `/jobs/{id}/report` | Markdown report (speakers, chapters, takeaways, transcript) |
| GET | `/jobs/{id}/video` | the video file with `Accept-Ranges`/`206 Partial Content` (browser seeking works) |
| GET | `/jobs/{id}/faces/{file}` | avatar image, e.g. `S1.jpg` (`jpg/jpeg/png/webp` only) |

## Behavior details

- **Search**: keywords match whole words (no stemming). A query with no
  positive-scoring chunk returns an empty list, never a fabricated result.
- **Ask**: always grounded in retrieved chunks and returns them in `sources`
  with timestamps. With an LLM configured the answer cites excerpts like `[1]`;
  without one the answer is clearly labelled
  `(Showing the most relevant transcript excerpt.)` — it never pretends an LLM
  wrote it. If nothing matches: `No relevant transcript excerpts were found
  for this question.`
- **Errors**: `400` invalid query/`k`/body (422 validation is converted to
  400), `404` unknown job/file/invalid job id, `416` unsatisfiable byte range
  (with `Content-Range: bytes */size`), `500` unreadable result/chunks files
  (job-corrupt). Internals are never leaked in error bodies.
- **Safety**: `job_id` must match `[A-Za-z0-9_-]+`; all files resolve under
  `JOBS_DIR/{job_id}`; traversal (`..`, encoded slashes, symlinks out of the
  job folder) returns 404. Faces accept only flat image filenames.
- **CORS**: from `CORS_ORIGINS`. `*` disables credentials; deployed
  environments must list explicit origins.
- **Job listing**: jobs with unreadable/non-object `result.json` or ids that
  violate the pattern are skipped, not served missing.

## Frontend notes

- Seek the video with `video.currentTime = source.start`.
- Highlight the current chapter and chat bubble with the `timeupdate` event.
- Errors: 404 `{"detail": "Job not found"}`.
- Answers carry `sources`; render each as a clickable chip like `02:14 Priya`.

## Tests

```bash
cd backend
python -m pytest tests -q    # 44 tests: endpoints, errors, traversal, ranges, fallbacks
```
