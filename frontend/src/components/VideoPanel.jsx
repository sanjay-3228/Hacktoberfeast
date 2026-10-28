import { useState } from "react";
import { Film, AlertTriangle } from "lucide-react";
import { videoUrl } from "../api.js";

/**
 * HTML5 video player using the backend's HTTP-Range route
 * (GET /jobs/{id}/video). Seeking is built into the element because the
 * endpoint answers 206 Partial Content for byte ranges.
 */
export default function VideoPanel({ jobId, videoRef, onTimeUpdate }) {
  const [failed, setFailed] = useState(false);

  if (!jobId) return null;

  if (failed) {
    return (
      <section className="workstation-panel" data-testid="video-missing">
        <div className="panel-header">
          <span className="panel-title">
            <Film size={14} /> Video Player
          </span>
        </div>
        <div className="video-fallback-box">
          <AlertTriangle size={24} color="var(--vke-warning)" style={{ marginBottom: 8 }} />
          <p style={{ fontWeight: 500, color: "var(--vke-text-primary)", marginBottom: 4 }}>
            Video track unavailable
          </p>
          <p style={{ fontSize: 11, color: "var(--vke-text-muted)" }}>
            The job folder contains no playable <code>video.mp4</code>.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="workstation-panel">
      <div className="panel-header">
        <span className="panel-title">
          <Film size={14} /> Video Player
        </span>
        <span className="panel-badge mono">{jobId}</span>
      </div>
      <div className="video-wrapper">
        <video
          ref={videoRef}
          className="workstation-video video"
          controls
          preload="metadata"
          src={videoUrl(jobId)}
          onTimeUpdate={(event) => onTimeUpdate(event.currentTarget.currentTime)}
          onError={() => setFailed(true)}
        >
          Your browser does not support HTML5 video streaming.
        </video>
      </div>
    </section>
  );
}
