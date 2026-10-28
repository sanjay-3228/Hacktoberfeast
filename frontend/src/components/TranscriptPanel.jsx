import { useMemo, useState } from "react";
import { Search, Filter, Copy, Check } from "lucide-react";
import Avatar from "./Avatar.jsx";
import { mmss, sortedMessages, speakerMap } from "../format.js";

/**
 * Dense, synchronized transcript review built from result.json `messages` + `speakers`.
 * Clicking a row seeks the video to the message's start timestamp.
 */
export default function TranscriptPanel({ jobId, result, currentTime, onSeek }) {
  const [filterText, setFilterText] = useState("");
  const [speakerFilter, setSpeakerFilter] = useState("all");
  const [copiedIndex, setCopiedIndex] = useState(null);

  const speakers = useMemo(() => speakerMap(result), [result]);
  const messages = useMemo(() => sortedMessages(result), [result]);

  const uniqueSpeakers = useMemo(() => {
    const set = new Set();
    messages.forEach((m) => {
      if (m.speaker) set.add(m.speaker);
    });
    return Array.from(set);
  }, [messages]);

  const filteredMessages = useMemo(() => {
    return messages.filter((m) => {
      if (speakerFilter !== "all" && m.speaker !== speakerFilter) {
        return false;
      }
      if (filterText.trim()) {
        const q = filterText.toLowerCase();
        return m.text.toLowerCase().includes(q);
      }
      return true;
    });
  }, [messages, speakerFilter, filterText]);

  const handleCopy = (e, text, start, index) => {
    e.stopPropagation();
    navigator.clipboard?.writeText?.(`[${mmss(start)}] ${text}`);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 1800);
  };

  if (!result) {
    return (
      <section className="workstation-panel transcript-panel" aria-busy="true">
        <div className="panel-header">
          <span className="panel-title">Synchronized Transcript</span>
        </div>
        <div className="empty-state-message">Loading transcript data…</div>
      </section>
    );
  }

  if (!messages.length) {
    return (
      <section className="workstation-panel transcript-panel">
        <div className="panel-header">
          <span className="panel-title">Synchronized Transcript</span>
        </div>
        <div className="empty-state-message">This video has no transcript messages.</div>
      </section>
    );
  }

  return (
    <section className="workstation-panel transcript-panel">
      <div className="panel-header">
        <span className="panel-title">
          Synchronized Transcript
          <span className="panel-badge">{messages.length} utterances</span>
        </span>
      </div>

      <div className="transcript-toolbar">
        <div style={{ position: "relative", flex: 1, display: "flex", alignItems: "center" }}>
          <Search size={14} style={{ position: "absolute", left: 8, color: "var(--vke-text-muted)" }} />
          <input
            type="text"
            className="transcript-search-input"
            style={{ paddingLeft: 28 }}
            placeholder="Filter transcript…"
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
          />
        </div>
        {uniqueSpeakers.length > 1 && (
          <select
            className="transcript-speaker-filter"
            value={speakerFilter}
            onChange={(e) => setSpeakerFilter(e.target.value)}
            aria-label="Filter by speaker"
          >
            <option value="all">All Speakers ({uniqueSpeakers.length})</option>
            {uniqueSpeakers.map((spk) => (
              <option key={spk} value={spk}>
                {speakers[spk]?.name || spk}
              </option>
            ))}
          </select>
        )}
      </div>

      <ul className="transcript-scroll-area transcript" data-testid="transcript-list">
        {filteredMessages.map((message, index) => {
          const info = speakers[message.speaker];
          const name = info?.name || message.speaker || "Unknown";
          const active =
            currentTime >= message.start &&
            (currentTime < message.end || message.end <= message.start);
          const endWithin = message.end > message.start ? message.end : message.start;

          return (
            <li key={`${message.start}-${index}`}>
              <div
                role="button"
                tabIndex={0}
                className={`transcript-row bubble${active ? " active" : ""}`}
                onClick={() => onSeek(message.start)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSeek(message.start);
                  }
                }}
                title="Click to seek video to timestamp"
              >
                <div className="transcript-time-col">
                  {mmss(message.start)}
                  {endWithin > message.start && `–${mmss(endWithin)}`}
                </div>

                <div className="transcript-content-col">
                  <div className="transcript-speaker-line">
                    <Avatar
                      jobId={jobId}
                      avatar={info?.avatar}
                      name={name}
                      size={20}
                    />
                    <span>{name}</span>
                    {message.speaker && (
                      <span className="bubble-id">({message.speaker})</span>
                    )}
                    {!info?.on_camera && (
                      <span className="chip-camera chip-off">Off-camera</span>
                    )}
                    <button
                      type="button"
                      className="icon-button"
                      style={{ marginLeft: "auto", opacity: 0.7 }}
                      title="Copy timestamp & text"
                      onClick={(e) => handleCopy(e, message.text, message.start, index)}
                    >
                      {copiedIndex === index ? <Check size={12} color="var(--vke-success)" /> : <Copy size={12} />}
                    </button>
                  </div>
                  <div className="transcript-text bubble-text">{message.text}</div>
                </div>
              </div>
            </li>
          );
        })}
        {filteredMessages.length === 0 && (
          <li className="empty-state-message">No transcript entries match the current filter.</li>
        )}
      </ul>
    </section>
  );
}