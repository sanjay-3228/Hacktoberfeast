# Frontend

Suggested stack: React + Vite + Tailwind. Use the HTML5 `<video>` element
(`currentTime` to seek, `timeupdate` to highlight).

```bash
npm create vite@latest . -- --template react
npm install
npm install -D tailwindcss @tailwindcss/vite
```

Set the API URL in `.env`:  `VITE_API_URL=http://localhost:8000`

Start with the mock job: `GET /jobs/demo/result` (see `../docs/API.md`).

## Screens

**1. Home**: list of processed videos from `GET /jobs` (cards with title,
topic, duration). Optional: upload button (Plan B only).

**2. Workspace** (3 columns on desktop, tabs on mobile)
- Left: chapter list (title + time). Highlight the current chapter. Click to seek.
- Center: video player with a timeline bar showing chapter markers and
  colored segments per speaker. Below: topic card and key-takeaway cards.
- Right: group-chat transcript. WhatsApp-style bubbles with the speaker's
  face avatar (`/jobs/{id}/faces/S1.jpg`), name, timestamp. Off-camera
  speakers get a mic/silhouette icon and an "Off-camera" tag. Click a bubble
  to seek; auto-scroll and highlight the current one.
- Bottom/floating: **Ask AI** box (`POST /jobs/{id}/ask`). Show the answer and
  source chips like `02:14 Priya`; clicking a chip seeks the video.

**Extras**: speaker panel (talk-time %), filter chat by speaker, search box
(`/search`), suggested questions, dark mode, "Export report" button
(`/report`, Markdown download).
