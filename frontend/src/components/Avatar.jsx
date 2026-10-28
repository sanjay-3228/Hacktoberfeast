import { useState } from "react";
import { avatarUrl } from "../api.js";

/**
 * Speaker avatar. Shows the backend-served crop when available; otherwise a
 * neutral initial-letter placeholder (identity is NOT biometrically verified
 * by this app).
 */
export default function Avatar({ jobId, avatar, name, size = 34 }) {
  const [failed, setFailed] = useState(false);
  const src = avatar && !failed ? avatarUrl(jobId, avatar) : "";
  if (src) {
    return (
      <img
        className="avatar"
        src={src}
        alt={`${name || "Speaker"} avatar crop`}
        width={size}
        height={size}
        onError={() => setFailed(true)}
      />
    );
  }
  return (
    <span
      className="avatar avatar-fallback"
      style={{ width: size, height: size, fontSize: size * 0.45 }}
      role="img"
      aria-label={`${name || "Unknown speaker"} placeholder avatar`}
    >
      {(name || "?").slice(0, 1).toUpperCase()}
    </span>
  );
}
