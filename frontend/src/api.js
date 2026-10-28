/** Shared API client for the EchoSight Forensics backend.
 *
 * Every request goes through `request()`, which normalizes errors into
 * `ApiError` and parses JSON (or text) responses. Nothing here is mock data;
 * failures propagate to the UI as visible messages.
 */
const RAW_BASE =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_BASE_URL) ||
  "http://127.0.0.1:8000";

/** Backend origin, without a trailing slash. */
export const API_BASE = String(RAW_BASE).replace(/\/+$/, "");

export class ApiError extends Error {
  /** @param {string} message @param {number} status HTTP status (0 = network failure) */
  constructor(message, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Perform a request against the backend and normalize the response.
 * @param {string} path Path starting with "/".
 * @param {RequestInit} [options] fetch options.
 * @param {"json"|"text"} [kind] How to parse a successful response.
 */
async function request(path, options = {}, kind = "json") {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch {
    throw new ApiError(
      `Cannot reach the backend at ${API_BASE}. Is it running?`
    );
  }

  if (!response.ok) {
    let detail = "";
    try {
      const body = await response.json();
      detail = typeof body?.detail === "string" ? body.detail : "";
      if (!detail) detail = JSON.stringify(body);
    } catch {
      detail = await response.text().catch(() => "");
    }
    throw new ApiError(
      detail || `Request failed with HTTP ${response.status}`,
      response.status
    );
  }

  if (kind === "text") {
    return response.text();
  }

  try {
    const data = await response.json();
    if (data == null) throw new Error("empty JSON body");
    return data;
  } catch {
    throw new ApiError("Backend returned invalid JSON", response.status);
  }
}

/** @returns {Promise<{ok: boolean}>} */
export const fetchHealth = () => request("/health");

/** @returns {Promise<Array<{job_id: string, title: string, topic: string, duration: number}>>} */
export const fetchJobs = () => request("/jobs");

/** @returns {Promise<object>} the job's full result document. */
export const fetchResult = (jobId) =>
  request(`/jobs/${encodeURIComponent(jobId)}/result`);

/**
 * Semantic (or keyword-fallback) transcript search.
 * @returns {Promise<{results: Array<{start,end,speaker,text,score}>}>}
 */
export const searchJob = (jobId, query, k = 5) =>
  request(
    `/jobs/${encodeURIComponent(jobId)}/search?q=${encodeURIComponent(
      query
    )}&k=${encodeURIComponent(k)}`
  );

/**
 * Ask a grounded question about the video.
 * @returns {Promise<{answer: string, answer_type: string, conversation_id: string, sources: Array<{segment_id,start,end,speaker_id,speaker,text,score}>}>}
 */
export const askJob = (jobId, question, options = {}) =>
  request(
    `/jobs/${encodeURIComponent(jobId)}/ask`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, ...options }),
    }
  );

/** @returns {Promise<string>} the Markdown report body. */
export const fetchReport = (jobId) =>
  request(`/jobs/${encodeURIComponent(jobId)}/report`, {}, "text");

/** Range-capable (seekable) video stream URL. */
export const videoUrl = (jobId) =>
  `${API_BASE}/jobs/${encodeURIComponent(jobId)}/video`;

/**
 * Build the absolute avatar URL from a result.json avatar value such as
 * "faces/S1.jpg". Only the file name is used; the backend route serves flat
 * files from the job's faces/ directory.
 */
export function avatarUrl(jobId, avatar) {
  if (!avatar) return "";
  const file = String(avatar).split(/[\\/]/).pop();
  if (!file) return "";
  return `${API_BASE}/jobs/${encodeURIComponent(
    jobId
  )}/faces/${encodeURIComponent(file)}`;
}

/**
 * Upload a video file to /api/jobs and begin pipeline processing.
 * @param {File} file
 * @param {string} [language="auto"] Whisper language code or "auto"
 * @returns {Promise<{job_id: string, status: string}>}
 */
export async function uploadVideo(file, language = "auto") {
  const form = new FormData();
  form.append("video", file, file.name);
  if (language && language !== "auto") {
    form.append("language", language);
  }
  return request("/api/jobs", {
    method: "POST",
    body: form,
  });
}

/**
 * Reprocess an existing job with optional language override.
 * Backs up original transcript and result to .original.json.
 * @param {string} jobId
 * @param {string} [language="auto"]
 * @returns {Promise<{status: string}>}
 */
export async function reprocessJob(jobId, language = "auto") {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ language }),
  });
}

/**
 * Fetch current processing status and progress for a job.
 * @param {string} jobId
 * @returns {Promise<{status: string, progress: number, message: string, error?: string}>}
 */
export const fetchJobStatus = (jobId) =>
  request(`/api/jobs/${encodeURIComponent(jobId)}/status`);

