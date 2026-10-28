# Data contract (do not change keys without telling the team)

All times are **seconds (float) from the start of the video**.

> Note: `job_id` and `title` in `result.json` are **optional**. The pipeline
> may omit them; the backend fills `job_id` from the folder name and `title`
> with `""` when serving, so the frontend always receives both keys.

## Job folder

```
jobs/{job_id}/
  video.mp4          # web-playable (H.264 + AAC)
  audio.wav          # 16 kHz mono (pipeline only, not served)
  transcript.json    # raw Whisper output (pipeline only)
  diarization.json   # raw pyannote output (pipeline only)
  faces_tracks.json  # raw face tracks (pipeline only)
  result.json        # served to the frontend
  chunks.json        # used by search and Q&A
  faces/S1.jpg ...   # avatar crops, one per on-camera speaker
```

`job_id` may contain only letters, digits, `_` and `-`.

## result.json

```json
{
  "job_id": "demo",
  "title": "Optional short title",
  "topic": "2-3 sentence explanation of what the video is about",
  "duration": 120.0,
  "speakers": [
    {
      "id": "S1",
      "name": "Priya",
      "on_camera": true,
      "avatar": "faces/S1.jpg",
      "talk_time_pct": 42
    },
    {
      "id": "S3",
      "name": "Speaker 3",
      "on_camera": false,
      "avatar": null,
      "talk_time_pct": 22
    }
  ],
  "messages": [
    {"speaker": "S1", "start": 10.0, "end": 28.0, "text": "..."}
  ],
  "chapters": [
    {"title": "...", "start": 0.0, "end": 42.0, "summary": "..."}
  ],
  "takeaways": ["...", "..."]
}
```

Rules:
- `speakers[].id` is `S1`, `S2`, ... in order of first appearance.
- `name` falls back to `"Speaker N"` when the transcript does not state a name.
- Off-camera speakers: `on_camera: false`, `avatar: null`.
- `messages` are sorted by `start`. One message = one continuous turn.
- `chapters` are sorted, non-overlapping, and cover the video.
- `avatar` is relative to the job folder. Frontend URL:
  `/jobs/{job_id}/faces/S1.jpg`.

## chunks.json

```json
[
  {
    "id": 0,
    "start": 0.0,
    "end": 42.0,
    "speaker": "S3",
    "text": "...",
    "embedding": [0.01, -0.02, "..."]
  }
]
```

`embedding` may be `null` (the backend then falls back to keyword search).
