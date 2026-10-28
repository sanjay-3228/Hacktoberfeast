import { useCallback, useEffect, useRef, useState } from "react";
import {
  Shield,
  Film,
  FolderGit2,
  Search,
  FileText,
  UploadCloud,
  CheckCircle2,
  AlertCircle,
  Clock,
  Loader2,
  RefreshCw,
  X,
} from "lucide-react";
import {
  ApiError,
  fetchHealth,
  fetchJobs,
  fetchResult,
  uploadVideo,
  fetchJobStatus,
} from "./api.js";
import { jobTitle, mmss, safeNumber } from "./format.js";
import VideoPanel from "./components/VideoPanel.jsx";
import TranscriptPanel from "./components/TranscriptPanel.jsx";
import SpeakerPanel from "./components/SpeakerPanel.jsx";
import ChaptersPanel from "./components/ChaptersPanel.jsx";
import SearchPanel from "./components/SearchPanel.jsx";
import AskPanel from "./components/AskPanel.jsx";
import ReportPanel from "./components/ReportPanel.jsx";
import OverviewView from "./components/OverviewView.jsx";

/**
 * EchoSight Forensics — Professional Dark Forensic Workstation.
 *
 * Multimodal video analysis, synchronized speaker diarization review,
 * semantic evidence search, and case reporting.
 */
export default function App() {
  const [activeTab, setActiveTab] = useState(() => {
    const hash = typeof window !== "undefined" ? window.location.hash.replace("#", "") : "";
    return hash || "workspace";
  });
  const [backend, setBackend] = useState({ state: "checking", message: "" });
  const [jobs, setJobs] = useState([]);
  const [jobsError, setJobsError] = useState("");
  const [jobId, setJobId] = useState(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const urlJob = params.get("job");
      if (urlJob) return urlJob;
      try {
        return localStorage.getItem("vke_selected_job_id") || "";
      } catch {
        return "";
      }
    }
    return "";
  });
  const [result, setResult] = useState(null);
  const [resultLoading, setResultLoading] = useState(false);
  const [resultError, setResultError] = useState("");
  const [currentTime, setCurrentTime] = useState(0);

  const [uploadState, setUploadState] = useState({
    uploading: false,
    jobId: "",
    progress: 0,
    message: "",
    error: "",
  });
  const [statusDockVisible, setStatusDockVisible] = useState(false);

  const videoRef = useRef(null);
  const fileInputRef = useRef(null);

  const checkBackend = useCallback(async () => {
    setBackend({ state: "checking", message: "" });
    try {
      await fetchHealth();
      setBackend({ state: "online", message: "" });
    } catch (err) {
      setBackend({ state: "offline", message: err.message });
    }
  }, []);

  const loadJobs = useCallback(async () => {
    setJobsError("");
    try {
      const list = await fetchJobs();
      setJobs(Array.isArray(list) ? list : []);
    } catch (err) {
      setJobsError(err instanceof ApiError ? err.message : "Could not list jobs.");
    }
  }, []);

  useEffect(() => {
    checkBackend();
    loadJobs();
  }, [checkBackend, loadJobs]);

  // Resume tracking any in-flight job saved in localStorage on browser reload
  useEffect(() => {
    try {
      const savedJobId = localStorage.getItem("vke_active_job_id");
      if (savedJobId) {
        fetchJobStatus(savedJobId)
          .then((st) => {
            if (st.status === "processing" || st.status === "queued") {
              setUploadState({
                uploading: true,
                jobId: savedJobId,
                progress: st.progress ?? 0,
                message: st.message || (st.status === "queued" ? "Queued for processing" : "Pipeline running..."),
                error: "",
              });
              setStatusDockVisible(true);
            } else if (st.status === "completed") {
              localStorage.removeItem("vke_active_job_id");
              loadJobs();
            } else if (st.status === "failed") {
              localStorage.removeItem("vke_active_job_id");
            }
          })
          .catch(() => {
            localStorage.removeItem("vke_active_job_id");
          });
      }
    } catch {
      // Ignore localStorage exceptions in sandboxed test environments
    }
  }, [loadJobs]);

  // Persist active processing job to localStorage
  useEffect(() => {
    try {
      if (uploadState.uploading && uploadState.jobId) {
        localStorage.setItem("vke_active_job_id", uploadState.jobId);
      } else if (!uploadState.uploading) {
        localStorage.removeItem("vke_active_job_id");
      }
    } catch {
      // Ignore storage access errors
    }
  }, [uploadState.uploading, uploadState.jobId]);

  // Auto-select the first available job once the list loads.
  useEffect(() => {
    if (!jobId && jobs.length > 0) {
      setJobId(jobs[0].job_id);
    }
  }, [jobId, jobs]);

  // Persist selected job ID to localStorage
  useEffect(() => {
    if (jobId) {
      try {
        localStorage.setItem("vke_selected_job_id", jobId);
      } catch {
        // Ignore storage access errors
      }
    }
  }, [jobId]);

  // Load the selected job's result.
  useEffect(() => {
    if (!jobId) {
      setResult(null);
      setResultError("");
      return undefined;
    }
    let cancelled = false;
    setResultLoading(true);
    setResultError("");
    setResult(null);

    fetchResult(jobId)
      .then((data) => {
        if (!cancelled && data && typeof data === "object") {
          setResult(data);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setResultError(err?.message || "Could not load result.");
        }
      })
      .finally(() => {
        if (!cancelled) setResultLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [jobId]);

  // Poll status while an upload is processing.
  useEffect(() => {
    if (!uploadState.uploading || !uploadState.jobId) return undefined;
    const pollId = setInterval(async () => {
      try {
        const st = await fetchJobStatus(uploadState.jobId);
        if (st.status === "completed") {
          clearInterval(pollId);
          setUploadState({
            uploading: false,
            jobId: uploadState.jobId,
            progress: 100,
            message: "Complete",
            error: "",
          });
          setStatusDockVisible(true);
          await loadJobs();
          setJobId(uploadState.jobId);
          setActiveTab("workspace");
        } else if (st.status === "failed") {
          clearInterval(pollId);
          setUploadState({
            uploading: false,
            jobId: uploadState.jobId,
            progress: 0,
            message: "",
            error: st.message || st.error || "Processing failed",
          });
          setStatusDockVisible(true);
        } else {
          setUploadState((prev) => ({
            ...prev,
            progress: st.progress !== undefined ? st.progress : prev.progress,
            message: st.message || (st.status === "queued" ? "Queued for processing" : "Pipeline processing..."),
          }));
        }
      } catch (err) {
        // Transient network issue during status poll
      }
    }, 2000);
    return () => clearInterval(pollId);
  }, [uploadState.uploading, uploadState.jobId, loadJobs]);

  const pollCurrentStatus = async () => {
    if (!uploadState.jobId) return;
    try {
      const st = await fetchJobStatus(uploadState.jobId);
      if (st.status === "completed") {
        setUploadState({
          uploading: false,
          jobId: uploadState.jobId,
          progress: 100,
          message: "Complete",
          error: "",
        });
        await loadJobs();
        setJobId(uploadState.jobId);
      } else if (st.status === "failed") {
        setUploadState({
          uploading: false,
          jobId: uploadState.jobId,
          progress: 0,
          message: "",
          error: st.message || st.error || "Processing failed",
        });
      } else {
        setUploadState((prev) => ({
          ...prev,
          progress: st.progress ?? prev.progress,
          message: st.message || prev.message,
        }));
      }
    } catch {
      // Ignore transient errors on manual probe
    }
  };

  const handleUploadFile = async (file, language = "auto") => {
    if (!file) return;
    setUploadState({
      uploading: true,
      jobId: "",
      progress: 0,
      message: "Uploading video...",
      error: "",
    });
    setStatusDockVisible(true);

    try {
      const resp = await uploadVideo(file, language);
      setUploadState((prev) => ({
        ...prev,
        jobId: resp.job_id,
        message: "Pipeline initialized...",
      }));
    } catch (err) {
      setUploadState({
        uploading: false,
        jobId: "",
        progress: 0,
        message: "",
        error: err.message || "Upload failed",
      });
    }
  };

  const handleTopUpload = (event) => {
    const file = event.target.files?.[0];
    if (file) handleUploadFile(file);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const seek = useCallback((seconds) => {
    const video = videoRef.current;
    const target = Number(seconds);
    if (!video || !Number.isFinite(target)) return;
    video.currentTime = Math.max(0, target);
    video.play?.().catch(() => {});
  }, []);

  const selectedJob = jobs.find((j) => j?.job_id === jobId);
  const headlineTitle = selectedJob
    ? jobTitle(selectedJob)
    : result
      ? jobTitle({ ...result, job_id: jobId })
      : "";

  return (
    <div className="app-container app">
      {/* Persistent Left Sidebar Navigation */}
      <aside className="app-sidebar">
        <div className="sidebar-header">
          <div className="brand-icon-wrapper">
            <Shield size={16} />
          </div>
          <div className="brand-text">
            <span className="brand-title brand">EchoSight Forensics</span>
            <span className="brand-subtitle">VKE Workstation</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-section-title">Navigation</div>
          <button
            type="button"
            className={`nav-button ${activeTab === "workspace" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("workspace");
              try { window.location.hash = "workspace"; } catch {}
            }}
          >
            <Film size={15} />
            <span>Workspace</span>
            {result && <span className="nav-badge mono">{jobId}</span>}
          </button>

          <button
            type="button"
            className={`nav-button ${activeTab === "overview" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("overview");
              try { window.location.hash = "overview"; } catch {}
            }}
          >
            <FolderGit2 size={15} />
            <span>Jobs &amp; Ingest</span>
            <span className="nav-badge mono">{jobs.length}</span>
          </button>

          <button
            type="button"
            className={`nav-button ${activeTab === "search" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("search");
              try { window.location.hash = "search"; } catch {}
            }}
          >
            <Search size={15} />
            <span>Evidence Search</span>
          </button>

          <button
            type="button"
            className={`nav-button ${activeTab === "report" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("report");
              try { window.location.hash = "report"; } catch {}
            }}
          >
            <FileText size={15} />
            <span>Case Report</span>
          </button>
        </nav>

        <div className="sidebar-footer">
          {jobId && (
            <div className="active-job-indicator">
              <span className="active-job-label">Active Case</span>
              <span className="active-job-id" title={headlineTitle || jobId}>
                {headlineTitle || jobId}
              </span>
            </div>
          )}

          <div
            className={`backend-status-pill backend backend-${backend.state}`}
            data-testid="backend-status"
          >
            <span className="status-dot dot" aria-hidden="true" />
            <span>
              Backend:{" "}
              {backend.state === "online"
                ? "Online"
                : backend.state === "offline"
                  ? "Offline"
                  : "Checking…"}
            </span>
            {backend.state === "offline" && (
              <button type="button" className="status-retry-btn link" onClick={checkBackend}>
                Retry
              </button>
            )}
          </div>
        </div>
      </aside>

      {/* Main Forensic Content Area */}
      <div className="app-main">
        {/* Top Header Bar */}
        <header className="app-topbar topbar">
          <div className="topbar-left">
            <div className="topbar-breadcrumbs">
              <span>EchoSight</span>
              <span>/</span>
              <span className="breadcrumb-active">
                {activeTab === "workspace"
                  ? "Workspace"
                  : activeTab === "overview"
                    ? "Evidence Ingestion"
                    : activeTab === "search"
                      ? "Search"
                      : "Case Report"}
              </span>
            </div>

            <div className="topbar-job-selector">
              <label htmlFor="job-select" style={{ fontSize: 11, color: "var(--vke-text-muted)" }}>
                Job
              </label>
              <select
                id="job-select"
                aria-label="Job"
                className="job-select-input"
                value={jobId}
                onChange={(e) => {
                  setJobId(e.target.value);
                  setActiveTab("workspace");
                }}
                disabled={jobs.length === 0}
              >
                <option value="">
                  {jobs.length === 0
                    ? jobsError
                      ? "Unavailable"
                      : "No jobs available"
                    : "Select a job…"}
                </option>
                {jobs.map((j) => (
                  <option key={j.job_id} value={j.job_id}>
                    {j.job_id}
                    {jobTitle(j) ? ` — ${jobTitle(j)}` : ""}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="topbar-right">
            <input
              type="file"
              ref={fileInputRef}
              accept="video/*,.mp4,.mov,.webm,.avi,.mkv"
              style={{ display: "none" }}
              onChange={handleTopUpload}
              disabled={uploadState.uploading}
            />
            <button
              type="button"
              className="btn-sm-primary upload-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploadState.uploading}
            >
              {uploadState.uploading ? (
                <>
                  <Loader2 size={13} className="spin" />
                  <span>Processing ({uploadState.progress > 0 ? `${uploadState.progress}%` : "Running"})</span>
                </>
              ) : (
                <>
                  <UploadCloud size={13} />
                  <span>Upload Video</span>
                </>
              )}
            </button>
          </div>
        </header>

        {/* System Error Banner */}
        {backend.state === "offline" && (
          <div className="app-banner banner-error banner" role="alert">
            <AlertCircle size={14} />
            <span>Backend not reachable — {backend.message}</span>
          </div>
        )}

        {/* View Routing Area */}
        <div className="view-viewport">
          {/* Overview / Ingestion Tab */}
          {activeTab === "overview" && (
            <OverviewView
              jobs={jobs}
              jobsError={jobsError}
              selectedJobId={jobId}
              onSelectJob={(id) => {
                setJobId(id);
                setActiveTab("workspace");
              }}
              uploadState={uploadState}
              onUploadFile={handleUploadFile}
            />
          )}

          {/* Standalone Evidence Search Tab */}
          {activeTab === "search" && (
            <div style={{ maxWidth: 900, margin: "0 auto" }}>
              <SearchPanel jobId={jobId} onSeek={seek} />
            </div>
          )}

          {/* Standalone Case Report & Inquiry Tab */}
          {activeTab === "report" && (
            <div style={{ maxWidth: 900, margin: "0 auto", display: "flex", flexDirection: "column", gap: 16 }}>
              <ReportPanel jobId={jobId} />
              <AskPanel jobId={jobId} onSeek={seek} />
            </div>
          )}

          {/* Workspace Tab (Default when Job Selected) */}
          {activeTab === "workspace" && (
            <>
              {!jobId && !resultLoading && (
                <div className="workstation-panel card card-info" style={{ maxWidth: 800, margin: "40px auto", padding: 24 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
                    <FolderGit2 size={24} color="var(--vke-accent)" />
                    <h3 style={{ fontSize: 14, fontWeight: 600 }}>No Case Selected</h3>
                  </div>
                  <p className="empty-state-message empty-text" style={{ textAlign: "left", padding: 0 }}>
                    {jobsError
                      ? `Could not load the job list: ${jobsError}`
                      : "No completed jobs found. Run the pipeline to produce a job folder under backend/jobs/, or upload a video using the button above."}
                  </p>
                </div>
              )}

              {resultLoading && (
                <div className="empty-state-message" style={{ padding: "60px 0" }}>
                  <Loader2 size={24} className="spin" style={{ margin: "0 auto 12px", display: "block", color: "var(--vke-accent)" }} />
                  <span>Loading forensic case result…</span>
                </div>
              )}

              {resultError && (
                <div className="workstation-panel" style={{ maxWidth: 600, margin: "40px auto", padding: 20, borderColor: "var(--vke-error-border)" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--vke-error)", marginBottom: 8 }}>
                    <AlertCircle size={16} />
                    <span style={{ fontWeight: 600 }}>Result Load Error</span>
                  </div>
                  <p className="error-text" style={{ fontSize: 12, color: "var(--vke-text-secondary)" }}>
                    {resultError}
                  </p>
                </div>
              )}

              {jobId && result && (
                <main className="workspace-layout grid">
                  {/* Left Column: Video + Speakers + Chapters */}
                  <div className="workspace-left-pane col">
                    <VideoPanel
                      jobId={jobId}
                      videoRef={videoRef}
                      onTimeUpdate={setCurrentTime}
                    />

                    {/* Metadata Summary Strip */}
                    <div className="evidence-meta-strip">
                      <span className="meta-item">
                        <Clock size={12} color="var(--vke-accent)" />
                        <span className="mono">{result.duration ? mmss(safeNumber(result.duration)) : "–"}</span>
                      </span>
                      <span>·</span>
                      <span className="meta-item">
                        <span>{Array.isArray(result.speakers) ? result.speakers.length : 0} Speakers</span>
                      </span>
                      <span>·</span>
                      <span className="meta-item">
                        <span>{Array.isArray(result.messages) ? result.messages.length : 0} Utterances</span>
                      </span>
                    </div>

                    <SpeakerPanel jobId={jobId} result={result} />
                    <ChaptersPanel result={result} onSeek={seek} />
                  </div>

                  {/* Right Column: Synchronized Transcript + Inquiry + Search + Report */}
                  <div className="workspace-right-pane col">
                    <TranscriptPanel
                      jobId={jobId}
                      result={result}
                      currentTime={currentTime}
                      onSeek={seek}
                    />
                    <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 16 }}>
                      <SearchPanel jobId={jobId} onSeek={seek} />
                      <AskPanel jobId={jobId} onSeek={seek} />
                      <ReportPanel jobId={jobId} />
                    </div>
                  </div>
                </main>
              )}
            </>
          )}
        </div>
      </div>

      {/* Non-Blocking Status Dock (Floating in bottom-right, never covers screen) */}
      {statusDockVisible && uploadState.jobId && (
        <aside className="status-dock" role="status" aria-live="polite">
          <div className="status-dock-header">
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
              {uploadState.uploading ? (
                <Loader2 size={13} className="spin" color="var(--vke-accent)" />
              ) : uploadState.error ? (
                <AlertCircle size={13} color="var(--vke-error)" />
              ) : (
                <CheckCircle2 size={13} color="var(--vke-success)" />
              )}
              <span>Case: <strong className="mono">{uploadState.jobId}</strong></span>
            </span>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              {uploadState.uploading && (
                <button
                  type="button"
                  className="icon-button"
                  title="Refresh status now"
                  onClick={pollCurrentStatus}
                >
                  <RefreshCw size={12} />
                </button>
              )}
              <button
                type="button"
                className="icon-button"
                title="Dismiss status dialog (does not cancel processing)"
                onClick={() => setStatusDockVisible(false)}
              >
                <X size={13} />
              </button>
            </div>
          </div>
          <div className="status-dock-body">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontWeight: 500, color: "var(--vke-text-primary)" }}>
                {uploadState.uploading
                  ? uploadState.message || "Pipeline processing in progress..."
                  : uploadState.error
                  ? "Processing Failed"
                  : "Analysis Completed"}
              </span>
              <span className="mono" style={{ color: "var(--vke-text-muted)", fontSize: 10 }}>
                {uploadState.progress > 0 ? `${uploadState.progress}%` : "Running"}
              </span>
            </div>

            {uploadState.uploading && (
              uploadState.progress > 0 ? (
                <div className="progress-track">
                  <div className="progress-bar-fill" style={{ width: `${uploadState.progress}%` }} />
                </div>
              ) : (
                <div className="progress-track-indeterminate">
                  <div className="progress-bar-indeterminate" />
                </div>
              )
            )}

            {uploadState.error && (
              <p style={{ color: "var(--vke-error)", fontSize: 11, margin: 0 }}>
                {uploadState.error}
              </p>
            )}

            {!uploadState.uploading && !uploadState.error && (
              <button
                type="button"
                className="btn-sm-primary"
                style={{ width: "100%", justifyContent: "center", marginTop: 4 }}
                onClick={() => {
                  setJobId(uploadState.jobId);
                  setActiveTab("workspace");
                  setStatusDockVisible(false);
                }}
              >
                Open Case in Workspace
              </button>
            )}
          </div>
        </aside>
      )}
    </div>
  );
}
