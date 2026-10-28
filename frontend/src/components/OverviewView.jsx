import { useState, useRef } from "react";
import {
  UploadCloud,
  Film,
  Clock,
  ArrowRight,
  FileCheck,
  AlertCircle,
  Loader2,
  CheckCircle2,
  Languages,
} from "lucide-react";
import { mmss, jobTitle, safeNumber } from "../format.js";

export const SUPPORTED_LANGUAGES = [
  { code: "auto", label: "Auto-detect (Recommended)" },
  { code: "ta", label: "Tamil (தமிழ்)" },
  { code: "en", label: "English" },
  { code: "hi", label: "Hindi (हिन्दी)" },
  { code: "te", label: "Telugu (తెలుగు)" },
  { code: "kn", label: "Kannada (ಕನ್ನಡ)" },
  { code: "ml", label: "Malayalam (മലയാളം)" },
  { code: "mr", label: "Marathi (मराठी)" },
  { code: "bn", label: "Bengali (বাংলা)" },
  { code: "es", label: "Spanish (Español)" },
  { code: "fr", label: "French (Français)" },
  { code: "de", label: "German (Deutsch)" },
];

export default function OverviewView({
  jobs,
  jobsError,
  selectedJobId,
  onSelectJob,
  uploadState,
  onUploadFile,
}) {
  const [selectedLanguage, setSelectedLanguage] = useState("auto");
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      onUploadFile(files[0], selectedLanguage);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onUploadFile(file, selectedLanguage);
    }
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="overview-grid">
      {/* Upload Zone */}
      <div
        className={`upload-dropzone ${isDragging ? "drag-active" : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !uploadState.uploading && fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          accept="video/*,.mp4,.mov,.webm,.avi,.mkv"
          style={{ display: "none" }}
          onChange={handleFileChange}
          disabled={uploadState.uploading}
        />
        <div className="upload-icon-container">
          {uploadState.uploading ? (
            <Loader2 size={22} className="spin" />
          ) : (
            <UploadCloud size={22} />
          )}
        </div>
        <div className="upload-dropzone-title">
          {uploadState.uploading
            ? "Video Processing In Progress"
            : "Upload Video for Forensic Analysis"}
        </div>
        <div className="upload-dropzone-subtitle">
          {uploadState.uploading
            ? "The multimodal pipeline is extracting audio, diarizing speakers, and indexing transcript"
            : "Drag & drop video file or click to browse (MP4, MOV, WEBM, MKV, AVI)"}
        </div>
        <div
          className="upload-language-select-row"
          onClick={(e) => e.stopPropagation()}
          style={{
            marginTop: 14,
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            backgroundColor: "var(--vke-bg-surface)",
            padding: "6px 12px",
            borderRadius: "var(--vke-radius-md)",
            border: "1px solid var(--vke-border)",
          }}
        >
          <Languages size={14} color="var(--vke-accent)" />
          <label
            htmlFor="transcription-lang-select"
            style={{ fontSize: 12, fontWeight: 500, color: "var(--vke-text-secondary)" }}
          >
            Spoken language:
          </label>
          <select
            id="transcription-lang-select"
            className="job-select-input"
            value={selectedLanguage}
            onChange={(e) => setSelectedLanguage(e.target.value)}
            disabled={uploadState.uploading}
            style={{
              padding: "3px 8px",
              fontSize: 12,
              width: "auto",
              cursor: uploadState.uploading ? "not-allowed" : "pointer",
            }}
          >
            {SUPPORTED_LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>
                {l.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Upload & Processing Progress Banner if Active */}
      {(uploadState.uploading || uploadState.error || uploadState.message === "Complete") && (
        <div className="upload-progress-card">
          <div className="upload-progress-header">
            <span style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 500 }}>
              {uploadState.uploading ? (
                <>
                  <Loader2 size={13} className="spin" color="var(--vke-accent)" />
                  <span>Processing Job: <strong className="mono">{uploadState.jobId || "initializing…"}</strong></span>
                </>
              ) : uploadState.error ? (
                <>
                  <AlertCircle size={13} color="var(--vke-error)" />
                  <span style={{ color: "var(--vke-error)" }}>Processing Error</span>
                </>
              ) : (
                <>
                  <CheckCircle2 size={13} color="var(--vke-success)" />
                  <span style={{ color: "var(--vke-success)" }}>Processing Completed</span>
                </>
              )}
            </span>
            <span className="mono" style={{ color: "var(--vke-text-muted)" }}>
              {uploadState.progress > 0 ? `${uploadState.progress}%` : "Running"}
            </span>
          </div>

          {uploadState.uploading && uploadState.progress === 0 ? (
            <div className="progress-track-indeterminate">
              <div className="progress-bar-indeterminate" />
            </div>
          ) : (
            <div className="progress-track">
              <div
                className="progress-bar-fill"
                style={{
                  width: `${uploadState.progress}%`,
                  backgroundColor: uploadState.error ? "var(--vke-error)" : undefined,
                }}
              />
            </div>
          )}

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, color: "var(--vke-text-muted)" }}>
            <span>{uploadState.message || "Awaiting task execution…"}</span>
            {uploadState.error && (
              <span style={{ color: "var(--vke-error)" }}>{uploadState.error}</span>
            )}
            {!uploadState.uploading && !uploadState.error && uploadState.jobId && (
              <button
                type="button"
                className="btn-sm-primary"
                style={{ padding: "3px 10px", fontSize: 11 }}
                onClick={() => onSelectJob(uploadState.jobId)}
              >
                Open in Workspace
              </button>
            )}
          </div>
        </div>
      )}

      {/* Jobs Catalog */}
      <div className="workstation-panel">
        <div className="panel-header">
          <span className="panel-title">
            <Film size={14} /> Evidence Library &amp; Processed Recordings
          </span>
          <span className="panel-badge">{jobs.length} Available</span>
        </div>

        {jobsError ? (
          <div className="empty-state-message error-text" style={{ color: "var(--vke-error)" }}>
            Could not load jobs: {jobsError}
          </div>
        ) : jobs.length === 0 ? (
          <div className="empty-state-message">
            No completed jobs found. Drop a video above or place files in <code>backend/jobs/</code> to extract knowledge.
          </div>
        ) : (
          <div className="jobs-table-container">
            <table className="jobs-table">
              <thead>
                <tr>
                  <th style={{ width: 140 }}>Job ID</th>
                  <th>Title / Subject Matter</th>
                  <th style={{ width: 90 }}>Duration</th>
                  <th style={{ width: 110 }}>Status</th>
                  <th style={{ width: 130, textAlign: "right" }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => {
                  const isCurrent = job.job_id === selectedJobId;
                  const titleStr = jobTitle(job);
                  const durSec = safeNumber(job.duration);

                  return (
                    <tr key={job.job_id} className={isCurrent ? "selected-row" : ""}>
                      <td>
                        <span className="mono" style={{ fontWeight: 600, color: "var(--vke-accent)" }}>
                          {job.job_id}
                        </span>
                      </td>
                      <td>
                        <div style={{ display: "flex", flexDirection: "column" }}>
                          <span style={{ fontWeight: 500, color: "var(--vke-text-primary)" }}>
                            {titleStr}
                          </span>
                          {job.topic && job.topic !== titleStr && (
                            <span style={{ fontSize: 11, color: "var(--vke-text-muted)" }}>
                              {job.topic}
                            </span>
                          )}
                        </div>
                      </td>
                      <td>
                        <span className="mono" style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                          <Clock size={11} color="var(--vke-text-muted)" />
                          {durSec > 0 ? mmss(durSec) : "–"}
                        </span>
                      </td>
                      <td>
                        <span className="status-badge completed">
                          <CheckCircle2 size={11} /> Completed
                        </span>
                      </td>
                      <td style={{ textAlign: "right" }}>
                        <button
                          type="button"
                          className="btn-sm-primary"
                          style={{ padding: "3px 8px", fontSize: 11 }}
                          onClick={() => onSelectJob(job.job_id)}
                        >
                          <span>Analyze</span>
                          <ArrowRight size={11} />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

