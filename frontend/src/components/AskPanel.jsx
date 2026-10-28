import { useEffect, useState } from "react";
import { MessageSquare, Loader2 } from "lucide-react";
import { ApiError, askJob } from "../api.js";
import { mmss } from "../format.js";

const EXAMPLE_QUESTIONS = [
  "What are the main topics discussed?",
  "What does the speaker say about the core idea?",
  "Summarize the discussion about scheduling.",
  "At what time is privacy mentioned?",
];

/**
 * Grounded Q&A against POST /jobs/{id}/ask with a persisted job-scoped conversation.
 */
export default function AskPanel({ jobId, onSeek }) {
  const [question, setQuestion] = useState("");
  const [conversationId, setConversationId] = useState(() => {
    try {
      return localStorage.getItem(`vke_conversation_${jobId}`) || "";
    } catch {
      return "";
    }
  });
  const [state, setState] = useState({
    status: "idle", // idle | loading | done | error
    answer: "",
    sources: [],
    message: "",
    answerType: "insufficient",
    model: "",
  });

  useEffect(() => {
    let savedId = "";
    try {
      savedId = localStorage.getItem(`vke_conversation_${jobId}`) || "";
    } catch {
      // Storage can be disabled in private/sandboxed browsing contexts.
    }
    setConversationId(savedId);
    setQuestion("");
    setState({ status: "idle", answer: "", sources: [], message: "", answerType: "insufficient", model: "" });
  }, [jobId]);

  const startNewConversation = () => {
    setConversationId("");
    try {
      localStorage.removeItem(`vke_conversation_${jobId}`);
    } catch {
      // The request still starts a new server conversation without storage.
    }
    setQuestion("");
    setState({ status: "idle", answer: "", sources: [], message: "", answerType: "insufficient", model: "" });
  };

  const run = async (event) => {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || state.status === "loading") return;
    setState({
      status: "loading",
      answer: "",
      sources: [],
      message: "",
      answerType: "insufficient",
      model: "",
    });
    try {
      const payload = await askJob(jobId, trimmed, {
        conversation_id: conversationId || undefined,
      });
      if (payload?.conversation_id) {
        setConversationId(payload.conversation_id);
        try {
          localStorage.setItem(`vke_conversation_${jobId}`, payload.conversation_id);
        } catch {
          // The current component state still supports follow-ups this visit.
        }
      }
      setState({
        status: "done",
        answer: String(payload?.answer ?? ""),
        answerType: payload?.answer_type || "transcript_fact",
        sources: Array.isArray(payload?.sources) ? payload.sources : [],
        message: "",
        model: payload?.model || "",
      });
    } catch (err) {
      setState({
        status: "error",
        answer: "",
        sources: [],
        message: err instanceof ApiError ? err.message : "Inquiry failed.",
        model: "",
      });
    }
  };

  return (
    <section className="workstation-panel">
      <div className="panel-header">
        <span className="panel-title">
          <MessageSquare size={14} /> Grounded Evidence Inquiry
        </span>
        <button type="button" className="btn-sm-secondary" onClick={startNewConversation}>
          New conversation
        </button>
      </div>

      <div className="panel-body">
        <form className="search-controls-bar" onSubmit={run}>
          <input
            type="text"
            className="search-input-field"
            value={question}
            placeholder="Ask a question grounded in the recorded evidence…"
            aria-label="Question"
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button
            type="submit"
            className="btn-sm-primary"
            disabled={!question.trim() || state.status === "loading"}
          >
            {state.status === "loading" ? (
              <>
                <Loader2 size={13} className="spin" /> Analyzing…
              </>
            ) : (
              "Ask"
            )}
          </button>
        </form>

        {state.status === "idle" && !question && (
          <div className="example-prompts" style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 10 }}>
            <span style={{ fontSize: 11, color: "var(--vke-text-muted)" }}>Suggested inquiries:</span>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {EXAMPLE_QUESTIONS.map((q) => (
                <button
                  key={q}
                  type="button"
                  className="btn-sm-secondary"
                  style={{ fontSize: 11, textAlign: "left", borderRadius: "var(--vke-radius-sm)" }}
                  onClick={() => setQuestion(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {state.message && (
          <div className="empty-state-message error-text" style={{ color: "var(--vke-error)" }}>
            {state.message}
          </div>
        )}

        {state.status === "done" && (
          <div
            className="answer"
            data-testid="ask-answer"
            style={{
              padding: 12,
              backgroundColor: "var(--vke-bg-input)",
              borderRadius: "var(--vke-radius-sm)",
              border: "1px solid var(--vke-border-subtle)",
              display: "flex",
              flexDirection: "column",
              gap: 10,
              marginTop: 10,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
              {state.answerType === "greeting" ? (
                <span className="chip chip-fallback">Greeting & Guidance</span>
              ) : state.answerType === "insufficient" ? (
                <span className="chip chip-fallback">Insufficient transcript evidence</span>
              ) : state.answerType === "inference" ? (
                <span className="chip chip-fallback">Transcript-based inference</span>
              ) : state.answerType === "conflicting" ? (
                <span className="chip chip-fallback">Conflicting transcript evidence</span>
              ) : (
                <span className="chip">Transcript fact</span>
              )}

              {state.model.includes("gpt-oss-120b") && state.answerType !== "greeting" && (
                <span
                  className="score-tag mono"
                  title="Verified inference using Groq openai/gpt-oss-120b"
                  style={{ fontSize: 10, padding: "2px 6px" }}
                >
                  GPT-OSS 120B · Groq
                </span>
              )}
            </div>

            <p className="answer-text" style={{ fontSize: 12, lineHeight: 1.6, color: "var(--vke-text-primary)", whiteSpace: "pre-wrap" }}>
              {state.answer}
            </p>

            {state.sources.length > 0 && (
              <div style={{ borderTop: "1px solid var(--vke-border-subtle)", paddingTop: 8 }}>
                <span style={{ fontSize: 10, color: "var(--vke-text-muted)", textTransform: "uppercase", display: "block", marginBottom: 6 }}>
                  Source Excerpts
                </span>
                <ul className="answer-sources" style={{ display: "flex", flexDirection: "column", gap: 8, listStyle: "none" }}>
                  {state.sources.map((source, index) => {
                    const sStart = Number(source?.start);
                    const evidenceLabel = source?.evidence_id ? `[${source.evidence_id}]` : `[E${index + 1}]`;
                    return (
                      <li key={index} style={{ padding: "6px 8px", backgroundColor: "var(--vke-bg-panel)", borderRadius: "var(--vke-radius-sm)", border: "1px solid var(--vke-border-subtle)" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                          <button
                            type="button"
                            className="source-chip"
                            title={`Seek video to ${mmss(sStart)}`}
                            style={{
                              fontSize: 10,
                              fontFamily: "var(--vke-font-mono)",
                              padding: "2px 6px",
                              borderRadius: "var(--vke-radius-sm)",
                              backgroundColor: "var(--vke-bg-input)",
                              border: "1px solid var(--vke-border)",
                              color: "var(--vke-accent)",
                              cursor: "pointer",
                              display: "inline-flex",
                              alignItems: "center",
                              gap: 4,
                            }}
                            onClick={() => Number.isFinite(sStart) && onSeek(sStart)}
                          >
                            <strong>{evidenceLabel}</strong>
                            <span>{mmss(sStart)}</span>
                          </button>
                          {source?.speaker && (
                            <span style={{ fontSize: 11, color: "var(--vke-text-muted)" }}>
                              {source.speaker}
                            </span>
                          )}
                        </div>
                        <p className="source-excerpt" style={{ fontSize: 11, lineHeight: 1.45, color: "var(--vke-text-secondary)", margin: 0 }}>
                          {source?.text}
                        </p>
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
