# Tasks and build order

## Build order (each step leaves a working demo)

1. Contract agreed (`docs/CONTRACT.md`), mock job works in the backend.
2. Pipeline: ingest -> transcribe -> diarize -> merge (group chat works).
3. Pipeline: LLM chapters, takeaways, topic, names.
4. Pipeline: chunks + embeddings -> search and Q&A work.
5. Pipeline: face tracking + speaker-to-face matching + off-camera label.
6. Frontend polish, report export, deploy to Azure, rehearse.

## Checklist

### Pipeline (Kaggle/Colab)
- [ ] Kaggle account verified, GPU on, Hugging Face token saved as secret `HF_TOKEN`
- [ ] Accepted terms for the pyannote models on Hugging Face
- [ ] `stages/ingest.py` tested
- [ ] `stages/transcribe.py` tested
- [ ] `stages/diarize.py` tested
- [ ] `stages/merge.py` tested on a 2-speaker clip
- [ ] `stages/llm.py` returns valid JSON (names, topic, chapters, takeaways)
- [ ] `stages/index.py` writes `chunks.json`
- [ ] `stages/faces.py` + `stages/match.py` (tune `ON_CAMERA_THRESHOLD`)
- [ ] 3-5 demo videos processed and downloaded

### Backend (Azure)
- [ ] Runs locally with the mock `demo` job
- [ ] `/ask` works with an LLM key (set in `.env`)
- [ ] Real job folders copied into `backend/jobs/`
- [ ] Deployed on Azure, CORS set to the frontend URL
- [ ] Budget alert set in Azure Cost Management

### Frontend
- [ ] Upload/home screen listing jobs (`GET /jobs`)
- [ ] Workspace: video + chapter list + group-chat transcript
- [ ] Speaker timeline bar (colored segments)
- [ ] Ask box with source chips that seek the video
- [ ] Export report button
- [ ] Deployed (Azure Static Web Apps)

## Demo videos to prepare
- Clean interview, 2 speakers, both on camera
- One speaker off camera (shows the unique feature)
- Lecture/tutorial with a question from the audience
- One messy clip to find failures early
