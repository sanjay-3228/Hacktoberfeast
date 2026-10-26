# Backend API (FastAPI)

Interactive docs: `http://localhost:8000/docs`

| Method | Path | Returns |
|--------|------|---------|
| GET | `/health` | `{"ok": true}` |
| GET | `/jobs` | list of `{job_id, title, topic, duration}` |
| GET | `/jobs/{id}/status` | `{"status": "done"}` (always `done` in Plan A) |
| GET | `/jobs/{id}/result` | the full `result.json` |
| GET | `/jobs/{id}/video` | the video file (supports seeking) |
| GET | `/jobs/{id}/faces/{file}` | avatar image, e.g. `S1.jpg` |
| GET | `/jobs/{id}/search?q=...&k=5` | `{"results": [{start, end, speaker, text, score}]}` |
| POST | `/jobs/{id}/ask` | body `{"question": "..."}` -> `{"answer": "...", "sources": [{start, end, speaker, text, score}]}` |
| GET | `/jobs/{id}/report` | Markdown report (speakers, chapters, takeaways, transcript) |

## Frontend notes

- Seek the video with `video.currentTime = source.start`.
- Highlight the current chapter and chat bubble with the `timeupdate` event.
- Errors: 404 `{"detail": "Job not found"}`.
- Answers carry `sources`; render each as a clickable chip like `02:14 Priya`.
