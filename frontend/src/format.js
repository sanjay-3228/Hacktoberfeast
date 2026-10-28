/** Formatting helpers and safe accessors for result.json documents. */

/** Seconds -> "M:SS" (or "H:MM:SS" past an hour). */
export function mmss(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h
    ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
    : `${m}:${String(s).padStart(2, "0")}`;
}

/** Display title for a job/result; never blank. */
export function jobTitle(job) {
  if (!job) return "";
  return job.title || job.topic || `Job ${job.job_id || ""}`.trim();
}

/** A finite number, or `fallback`. */
export function safeNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/** speakers[] as { [id]: {name, on_camera, avatar} } with safe defaults. */
export function speakerMap(result) {
  const map = {};
  const speakers = Array.isArray(result?.speakers) ? result.speakers : [];
  for (const speaker of speakers) {
    if (!speaker || typeof speaker !== "object") continue;
    const id = typeof speaker.id === "string" ? speaker.id : "";
    if (!id) continue;
    map[id] = {
      name:
        (typeof speaker.name === "string" && speaker.name.trim()) || id,
      on_camera: Boolean(speaker.on_camera),
      avatar: speaker.avatar || null,
    };
  }
  return map;
}

/** messages[] sorted by start, text trimmed, valid entries only. */
export function sortedMessages(result) {
  const messages = Array.isArray(result?.messages) ? result.messages : [];
  return messages
    .filter(
      (m) =>
        m &&
        typeof m === "object" &&
        typeof m.text === "string" &&
        m.text.trim() &&
        Number.isFinite(Number(m.start))
    )
    .map((m) => ({
      speaker: typeof m.speaker === "string" ? m.speaker : "",
      start: safeNumber(m.start),
      end: Math.max(safeNumber(m.end), safeNumber(m.start)),
      text: m.text.trim(),
    }))
    .sort((a, b) => a.start - b.start);
}

/** chapters[] sorted by start. */
export function sortedChapters(result) {
  const chapters = Array.isArray(result?.chapters) ? result.chapters : [];
  return chapters
    .filter((c) => c && typeof c === "object" && Number.isFinite(Number(c.start)))
    .map((c) => ({
      title: (typeof c.title === "string" && c.title.trim()) || "Untitled chapter",
      start: safeNumber(c.start),
      end: safeNumber(c.end),
      summary: typeof c.summary === "string" ? c.summary.trim() : "",
    }))
    .sort((a, b) => a.start - b.start);
}

/** takeaways[] as plain strings. */
export function takeaways(result) {
  const items = Array.isArray(result?.takeaways) ? result.takeaways : [];
  return items
    .map((t) => (typeof t === "string" ? t.trim() : String(t ?? "").trim()))
    .filter(Boolean);
}

/** Parse "***bold***" inline markers into runs [{text, bold?}]. */
export function inlineRuns(raw) {
  const text = String(raw ?? "");
  const runs = [];
  let rest = text;
  for (;;) {
    const match = rest.match(/\*\*([^*]+)\*\*/);
    if (!match) break;
    if (match.index > 0) runs.push({ text: rest.slice(0, match.index) });
    runs.push({ text: match[1], bold: true });
    rest = rest.slice(match.index + match[0].length);
  }
  if (rest) runs.push({ text: rest });
  return runs.length ? runs : [{ text: "" }];
}

/** Tiny Markdown reader for the backend's report: h1/h2/h3, bullets, bold. */
export function reportBlocks(markdown) {
  const blocks = [];
  for (const rawLine of String(markdown ?? "").split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    if (line.startsWith("### ")) blocks.push({ kind: "h3", runs: inlineRuns(line.slice(4)) });
    else if (line.startsWith("## ")) blocks.push({ kind: "h2", runs: inlineRuns(line.slice(3)) });
    else if (line.startsWith("# ")) blocks.push({ kind: "h1", runs: inlineRuns(line.slice(2)) });
    else if (/^[-*]\s+/.test(line)) blocks.push({ kind: "li", runs: inlineRuns(line.replace(/^[-*]\s+/, "")) });
    else blocks.push({ kind: "p", runs: inlineRuns(line) });
  }
  return blocks;
}

/** Label used by the backend when no LLM is configured (excerpt fallback). */
export const EXCERPT_FALLBACK_PREFIX =
  "(Showing the most relevant transcript excerpt.)";

/** True when an /ask answer was an excerpt fallback, not an LLM answer. */
export function isExcerptFallback(answer) {
  return String(answer ?? "").startsWith(EXCERPT_FALLBACK_PREFIX);
}
