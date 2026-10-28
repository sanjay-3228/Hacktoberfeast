import { useMemo } from "react";
import { BookOpen, CheckCircle2 } from "lucide-react";
import { mmss, sortedChapters, takeaways } from "../format.js";

/** Chapters (click to seek) and key takeaways from result.json. */
export default function ChaptersPanel({ result, onSeek }) {
  const chapters = useMemo(() => sortedChapters(result), [result]);
  const notes = useMemo(() => takeaways(result), [result]);

  if (!result) {
    return (
      <section className="workstation-panel" aria-busy="true">
        <div className="panel-header">
          <span className="panel-title">
            <BookOpen size={14} /> Chapters &amp; Takeaways
          </span>
        </div>
        <div className="empty-state-message">Loading chapters and notes…</div>
      </section>
    );
  }

  return (
    <section className="workstation-panel">
      <div className="panel-header">
        <span className="panel-title">
          <BookOpen size={14} /> Chapters &amp; Evidence Takeaways
        </span>
      </div>
      <div className="panel-body" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 600, color: "var(--vke-text-muted)", textTransform: "uppercase", marginBottom: 6 }}>
            Timeline Chapters
          </div>
          {chapters.length ? (
            <ul className="chapters-list chapters" data-testid="chapter-list">
              {chapters.map((chapter, index) => (
                <li key={`${chapter.start}-${index}`}>
                  <button
                    type="button"
                    className="chapter-item-btn chapter"
                    onClick={() => onSeek(chapter.start)}
                    title="Seek video to chapter start"
                  >
                    <span className="mono" style={{ color: "var(--vke-accent)", fontSize: 11, minWidth: 42 }}>
                      {mmss(chapter.start)}
                    </span>
                    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                      <strong style={{ fontSize: 12, color: "var(--vke-text-primary)" }}>{chapter.title}</strong>
                      {chapter.summary && (
                        <span style={{ fontSize: 11, color: "var(--vke-text-secondary)" }}>
                          {chapter.summary}
                        </span>
                      )}
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <div className="empty-state-message" style={{ padding: "8px 0" }}>No chapters generated for this video.</div>
          )}
        </div>

        <div style={{ borderTop: "1px solid var(--vke-border-subtle)", paddingTop: 12 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: "var(--vke-text-muted)", textTransform: "uppercase", marginBottom: 6 }}>
            Key Forensic Takeaways
          </div>
          {notes.length ? (
            <ul className="takeaway-list takeaways" data-testid="takeaway-list">
              {notes.map((note, index) => (
                <li key={index} style={{ display: "flex", alignItems: "flex-start", gap: 6 }}>
                  <CheckCircle2 size={13} color="var(--vke-accent)" style={{ flexShrink: 0, marginTop: 2 }} />
                  <span>{note}</span>
                </li>
              ))}
            </ul>
          ) : (
            <div className="empty-state-message" style={{ padding: "8px 0" }}>No structured takeaways recorded.</div>
          )}
        </div>
      </div>
    </section>
  );
}
