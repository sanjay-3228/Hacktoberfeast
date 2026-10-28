/** Unit tests for the API client and formatting helpers. */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  askJob,
  ApiError,
  avatarUrl,
  fetchJobs,
  fetchReport,
  searchJob,
  videoUrl,
  uploadVideo,
  reprocessJob,
  fetchJobStatus,
} from "./api.js";
import {
  inlineRuns,
  isExcerptFallback,
  mmss,
  reportBlocks,
  sortedMessages,
  speakerMap,
  takeaways,
} from "./format.js";

const jsonResponse = (body, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

beforeEach(() => {
  vi.unstubAllGlobals();
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("api client", () => {
  it("exposes API_BASE without a trailing slash", () => {
    expect(typeof videoUrl("demo")).toBe("string");
    expect(videoUrl("demo").startsWith("http")).toBe(true);
    expect(videoUrl("demo/x y?").endsWith("%2Fx%20y%3F/video")).toBe(true);
    expect(videoUrl("demo").includes("//jobs/")).toBe(false);
  });

  it("builds avatar URLs from path-ish values using only the file name", () => {
    expect(avatarUrl("demo", "faces/S1.jpg")).toMatch(/\/jobs\/demo\/faces\/S1\.jpg$/);
    expect(avatarUrl("demo", "faces\\S5.webp")).toMatch(/\/faces\/S5\.webp$/);
    expect(avatarUrl("demo", null)).toBe("");
    expect(avatarUrl("demo", "")).toBe("");
  });

  it("parses JSON responses and passes through data", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse([{ job_id: "demo" }])));
    await expect(fetchJobs()).resolves.toEqual([{ job_id: "demo" }]);
  });

  it("converts backend error details into ApiError", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ detail: "Job not found" }, 404)));
    const error = await fetchJobs().catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(404);
    expect(error.message).toBe("Job not found");
  });

  it("reports network failures with the API base hint", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("failed to fetch"); }));
    const error = await fetchJobs().catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.message).toMatch(/Cannot reach the backend/);
  });

  it("reports invalid JSON as an error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("<html>", { status: 200 }))
    );
    const error = await fetchJobs().catch((e) => e);
    expect(error.message).toMatch(/invalid JSON/i);
  });

  it("sends the ask payload as {question} JSON", async () => {
    const stub = vi.fn(async () => jsonResponse({ answer: "a", sources: [] }));
    vi.stubGlobal("fetch", stub);
    await askJob("demo", "What was decided?");
    const [url, init] = stub.mock.calls[0];
    expect(url).toContain("/jobs/demo/ask");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ question: "What was decided?" });
  });

  it("encodes query params for search", async () => {
    const stub = vi.fn(async () => jsonResponse({ results: [] }));
    vi.stubGlobal("fetch", stub);
    await searchJob("demo", "reliability & time", 5);
    const [url] = stub.mock.calls[0];
    expect(url).toContain("/search?q=reliability%20%26%20time&k=5");
  });

  it("returns the report as text", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("# Report", { status: 200 }))
    );
    await expect(fetchReport("demo")).resolves.toBe("# Report");
  });

  it("uploads video files via FormData to /api/jobs", async () => {
    const stub = vi.fn(async () => jsonResponse({ job_id: "test1234", status: "queued" }));
    vi.stubGlobal("fetch", stub);
    const fakeFile = new File(["test data"], "sample.mp4", { type: "video/mp4" });
    const result = await uploadVideo(fakeFile);
    expect(result.job_id).toBe("test1234");
    expect(stub.mock.calls[0][0]).toMatch(/\/api\/jobs$/);
    expect(stub.mock.calls[0][1].method).toBe("POST");
  });

  it("fetches job status from /api/jobs/:id/status", async () => {
    const stub = vi.fn(async () => jsonResponse({ status: "processing", progress: 45 }));
    vi.stubGlobal("fetch", stub);
    const result = await fetchJobStatus("test1234");
    expect(result.status).toBe("processing");
    expect(result.progress).toBe(45);
    expect(stub.mock.calls[0][0]).toMatch(/\/api\/jobs\/test1234\/status$/);
  });

  it("posts uploadVideo with language and video FormData", async () => {
    let capturedBody = null;
    const stub = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ job_id: "lang123", status: "queued" }),
    });
    vi.stubGlobal("fetch", stub);
    const file = new File(["dummy content"], "tamil_clip.mp4", { type: "video/mp4" });
    const res = await uploadVideo(file, "ta");
    expect(res.job_id).toBe("lang123");
    expect(stub).toHaveBeenCalledTimes(1);
    const callArgs = stub.mock.calls[0];
    expect(callArgs[0]).toMatch(/\/api\/jobs$/);
    const body = callArgs[1].body;
    expect(body instanceof FormData).toBe(true);
    expect(body.get("language")).toBe("ta");
    expect(body.get("video").name).toBe("tamil_clip.mp4");
  });

  it("posts reprocessJob with language override", async () => {
    const stub = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: "queued" }),
    });
    vi.stubGlobal("fetch", stub);
    const res = await reprocessJob("job999", "ta");
    expect(res.status).toBe("queued");
    expect(stub).toHaveBeenCalledTimes(1);
    const callArgs = stub.mock.calls[0];
    expect(callArgs[0]).toMatch(/\/api\/jobs\/job999\/reset$/);
    expect(JSON.parse(callArgs[1].body)).toEqual({ language: "ta" });
  });
});

describe("format helpers", () => {
  it("formats mm:ss and h:mm:ss", () => {
    expect(mmss(65)).toBe("1:05");
    expect(mmss(0)).toBe("0:00");
    expect(mmss(3700)).toBe("1:01:40");
    expect(mmss(Number.NaN)).toBe("0:00");
  });

  it("maps speakers with safe fallbacks", () => {
    const map = speakerMap({
      speakers: [
        { id: "S1", name: "Priya", on_camera: true, avatar: "faces/S1.jpg" },
        { id: "S3", name: "", on_camera: false, avatar: null },
        null,
        { name: "NoId" },
      ],
    });
    expect(Object.keys(map)).toEqual(["S1", "S3"]);
    expect(map.S3.name).toBe("S3"); // name falls back to the id
    expect(map.S1.on_camera).toBe(true);
  });

  it("sorts messages and drops invalid ones", () => {
    const messages = sortedMessages({
      messages: [
        { speaker: "S2", start: 10, end: 12, text: "second" },
        { speaker: "S1", start: 0, end: 5, text: "first" },
        "junk",
        { speaker: "S3", start: "bad", text: "no times" },
        { speaker: "S1", start: 7, text: "   " },
      ],
    });
    expect(messages.map((m) => m.text)).toEqual(["first", "second"]);
  });

  it("collects takeaways as strings", () => {
    expect(takeaways({ takeaways: [" a ", 42, null, "b"] })).toEqual(["a", "42", "b"]);
  });

  it("parses markdown report blocks", () => {
    const blocks = reportBlocks("# Title\n\n## Topic\n\n- **Priya** speaks\nplain line");
    expect(blocks.map((b) => b.kind)).toEqual(["h1", "h2", "li", "p"]);
    expect(blocks[2].runs[0].bold).toBe(true);
    expect(blocks[2].runs[0].text).toBe("Priya");
  });

  it("handles inline bold markers", () => {
    expect(inlineRuns("**[00:12] S1:** hello")).toEqual([
      { text: "[00:12] S1:", bold: true },
      { text: " hello" },
    ]);
  });

  it("detects the backend's excerpt-fallback answer", () => {
    expect(isExcerptFallback("(Showing the most relevant transcript excerpt.) [0:42] ...")).toBe(true);
    expect(isExcerptFallback("They used a reliability score [1].")).toBe(false);
    expect(isExcerptFallback(undefined)).toBe(false);
  });
});
