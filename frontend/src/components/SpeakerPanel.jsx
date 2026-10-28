import { useMemo } from "react";
import { Users } from "lucide-react";
import Avatar from "./Avatar.jsx";

/** Speaker analytics computed from result.json — only real API values. */
export default function SpeakerPanel({ jobId, result }) {
  const speakers = useMemo(() => {
    const raw = Array.isArray(result?.speakers) ? result.speakers : [];
    return raw.filter((s) => s && typeof s === "object" && s.id);
  }, [result]);

  if (!result) {
    return (
      <section className="workstation-panel" aria-busy="true">
        <div className="panel-header">
          <span className="panel-title">
            <Users size={14} /> Speaker Analytics
          </span>
        </div>
        <div className="empty-state-message">Loading speaker data…</div>
      </section>
    );
  }

  if (!speakers.length) {
    return (
      <section className="workstation-panel">
        <div className="panel-header">
          <span className="panel-title">
            <Users size={14} /> Speaker Analytics
          </span>
        </div>
        <div className="empty-state-message">No speaker records detected for this job.</div>
      </section>
    );
  }

  return (
    <section className="workstation-panel">
      <div className="panel-header">
        <span className="panel-title">
          <Users size={14} /> Speaker Analytics
        </span>
        <span className="panel-badge">{speakers.length} Identified</span>
      </div>
      <div className="panel-body">
        <ul className="speaker-list-container speakers" data-testid="speaker-list">
          {speakers.map((speaker) => {
            const name =
              (typeof speaker.name === "string" && speaker.name.trim()) ||
              String(speaker.id);
            const pct = Number(speaker.talk_time_pct);
            const hasPct = Number.isFinite(pct) && pct > 0;

            return (
              <li key={speaker.id} className="speaker-item speaker-row">
                <Avatar jobId={jobId} avatar={speaker.avatar} name={name} size={32} />
                <div className="speaker-details">
                  <span className="speaker-primary-name speaker-name">{name}</span>
                  <span className="speaker-id-tag bubble-id">Label: {speaker.id}</span>
                </div>
                <span
                  className={`chip-camera chip ${
                    speaker.on_camera ? "chip-on" : "chip-off"
                  }`}
                >
                  {speaker.on_camera ? "On Camera" : "Off Camera"}
                </span>
                <div className="speaker-talk-metric talk-time" title="Share of speech time">
                  <span>{hasPct ? `${Math.round(pct)}%` : "N/A"}</span>
                  {hasPct && (
                    <div className="talk-bar bar" aria-hidden="true">
                      <div
                        className="talk-bar-fill bar-fill"
                        style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
                      />
                    </div>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
        <p style={{ fontSize: 10, color: "var(--vke-text-muted)", marginTop: 10 }}>
          Speaker IDs are acoustic/visual pipeline cluster labels, not biometric identity claims.
        </p>
      </div>
    </section>
  );
}
