import { useEffect, useState } from "react";
import { FileText, Download } from "lucide-react";
import { ApiError, fetchReport } from "../api.js";
import { reportBlocks } from "../format.js";

/** Forensic report from GET /jobs/{id}/report (Markdown), with download. */
export default function ReportPanel({ jobId }) {
  const [state, setState] = useState({
    status: "idle", // idle | loading | done | error
    text: "",
    message: "",
  });

  useEffect(() => {
    if (!jobId) return undefined;
    let cancelled = false;
    setState({ status: "loading", text: "", message: "" });
    fetchReport(jobId)
      .then((text) => {
        if (!cancelled) setState({ status: "done", text, message: "" });
      })
      .catch((err) => {
        if (!cancelled) {
          setState({
            status: "error",
            text: "",
            message: err instanceof ApiError ? err.message : "Report failed to load.",
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  const download = () => {
    if (!state.text) return;
    const blob = new Blob([state.text], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${jobId}-report.md`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="workstation-panel">
      <div className="panel-header">
        <span className="panel-title">
          <FileText size={14} /> Forensic Case Report
        </span>
        {state.status === "done" && state.text && (
          <button
            type="button"
            className="btn-sm-primary link"
            onClick={download}
            title="Export Markdown report"
          >
            <Download size={12} /> Download .md
          </button>
        )}
      </div>

      <div className="panel-body">
        {state.status === "loading" && (
          <div className="empty-state-message">Compiling forensic report…</div>
        )}
        {state.status === "error" && (
          <div className="empty-state-message error-text" style={{ color: "var(--vke-error)" }}>
            {state.message}
          </div>
        )}

        {state.status === "done" && (
          <div className="report report-content" data-testid="report-body">
            {reportBlocks(state.text).map((block, index) => {
              if (block.kind === "h1") {
                return (
                  <h3 key={index} className="report-h1">
                    {block.runs.map((run, i) => (run.bold ? <strong key={i}>{run.text}</strong> : <span key={i}>{run.text}</span>))}
                  </h3>
                );
              }
              if (block.kind === "h2") {
                return (
                  <h4 key={index} className="report-h2">
                    {block.runs.map((run, i) => (run.bold ? <strong key={i}>{run.text}</strong> : <span key={i}>{run.text}</span>))}
                  </h4>
                );
              }
              if (block.kind === "h3") {
                return (
                  <h5 key={index} className="report-h3">
                    {block.runs.map((run, i) => (run.bold ? <strong key={i}>{run.text}</strong> : <span key={i}>{run.text}</span>))}
                  </h5>
                );
              }
              if (block.kind === "li") {
                return (
                  <p key={index} className="report-li" style={{ paddingLeft: 12, display: "flex", gap: 6 }}>
                    <span className="bullet" style={{ color: "var(--vke-accent)" }}>•</span>
                    <span>
                      {block.runs.map((run, i) => (run.bold ? <strong key={i}>{run.text}</strong> : <span key={i}>{run.text}</span>))}
                    </span>
                  </p>
                );
              }
              return (
                <p key={index} className="report-p" style={{ marginBottom: 6 }}>
                  {block.runs.map((run, i) => (run.bold ? <strong key={i}>{run.text}</strong> : <span key={i}>{run.text}</span>))}
                </p>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
