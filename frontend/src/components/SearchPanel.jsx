import { useState } from "react";
import { Search, Loader2, Play } from "lucide-react";
import { ApiError, searchJob } from "../api.js";
import { mmss } from "../format.js";

/** Transcript search against GET /jobs/{id}/search (semantic with keyword fallback). */
export default function SearchPanel({ jobId, onSeek }) {
  const [query, setQuery] = useState("");
  const [k, setK] = useState(5);
  const [state, setState] = useState({
    status: "idle", // idle | loading | done | error
    results: [],
    message: "",
  });

  const run = async (event) => {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || state.status === "loading") return;
    setState({ status: "loading", results: [], message: "" });
    try {
      const payload = await searchJob(jobId, trimmed, k);
      const results = Array.isArray(payload?.results) ? payload.results : [];
      setState({
        status: "done",
        results,
        message: results.length ? "" : `No matches found for "${trimmed}".`,
      });
    } catch (err) {
      setState({
        status: "error",
        results: [],
        message: err instanceof ApiError ? err.message : "Search query failed.",
      });
    }
  };

  return (
    <section className="workstation-panel">
      <div className="panel-header">
        <span className="panel-title">
          <Search size={14} /> Semantic Evidence Search
        </span>
      </div>

      <div className="panel-body">
        <form className="search-controls-bar" onSubmit={run}>
          <input
            type="search"
            className="search-input-field"
            value={query}
            placeholder="Search spoken testimony & evidence concepts…"
            aria-label="Search query"
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="search-k-control">
            <span>Top-k</span>
            <input
              type="number"
              className="search-k-input"
              min="1"
              max="50"
              value={k}
              aria-label="Maximum results"
              onChange={(e) => {
                const value = Number(e.target.value);
                setK(Math.min(50, Math.max(1, Math.floor(value) || 1)));
              }}
            />
          </div>
          <button
            type="submit"
            className="btn-sm-primary"
            disabled={!query.trim() || state.status === "loading"}
          >
            {state.status === "loading" ? (
              <>
                <Loader2 size={13} className="spin" /> Searching…
              </>
            ) : (
              "Search"
            )}
          </button>
        </form>

        {state.message && (
          <div
            className={`empty-state-message ${
              state.status === "error" ? "note error-text" : ""
            }`}
            style={{ color: state.status === "error" ? "var(--vke-error)" : undefined }}
          >
            {state.message}
          </div>
        )}

        {state.results.length > 0 && (
          <ul className="evidence-results-list search-results" data-testid="search-results">
            {state.results.map((hit, index) => {
              const startNum = Number(hit?.start);
              const endNum = Number(hit?.end);
              const scoreNum = Number(hit?.score);

              return (
                <li key={index}>
                  <button
                    type="button"
                    className="evidence-result-card search-hit"
                    onClick={() => Number.isFinite(startNum) && onSeek(startNum)}
                    title="Seek video to match timestamp"
                  >
                    <div className="evidence-result-top">
                      <div className="evidence-result-meta">
                        <span className="mono" style={{ color: "var(--vke-accent)", fontSize: 11 }}>
                          {mmss(startNum)}
                          {endNum > startNum && `–${mmss(endNum)}`}
                        </span>
                        {hit?.speaker && (
                          <span className="panel-badge mono">{hit.speaker}</span>
                        )}
                      </div>
                      {Number.isFinite(scoreNum) && (
                        <span className="score-tag hit-score">
                          Score {scoreNum.toFixed(2)}
                        </span>
                      )}
                    </div>
                    <div className="evidence-result-text hit-text">{hit?.text ?? ""}</div>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </section>
  );
}
