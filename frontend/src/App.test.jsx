/** UI behavior tests for the dashboard (jsdom + testing-library). */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App.jsx";

const testResult = vi.hoisted(() => ({
  job_id: "demo",
  title: "Study-group app interview",
  topic: "Students build a study-group app.",
  duration: 120.0,
  speakers: [
    { id: "S1", name: "Priya", on_camera: true, avatar: "faces/S1.jpg", talk_time_pct: 54 },
    { id: "S3", name: "Speaker 3", on_camera: false, avatar: null, talk_time_pct: 22 },
  ],
  messages: [
    { speaker: "S1", start: 0.0, end: 42.0, text: "The core idea is to match students." },
    { speaker: "S3", start: 42.0, end: 85.0, text: "We use a reliability score." },
  ],
  chapters: [{ title: "Introduction", start: 0.0, end: 120.0, summary: "Overview." }],
  takeaways: ["Match students by shared course."],
}));

vi.mock("./api.js", () => ({
  API_BASE: "http://test",
  ApiError: class ApiError extends Error {},
  fetchHealth: vi.fn(),
  fetchJobs: vi.fn(),
  fetchResult: vi.fn(),
  searchJob: vi.fn(async () => ({ results: [] })),
  askJob: vi.fn(async () => ({ answer: "a", sources: [] })),
  fetchReport: vi.fn(async () => "# Report\n\n## Topic\n\n- takeaway"),
  videoUrl: vi.fn((id) => `http://test/jobs/${id}/video`),
  avatarUrl: vi.fn((id, avatar) =>
    avatar ? `http://test/jobs/${id}/faces/${avatar.split("/").pop()}` : ""
  ),
  uploadVideo: vi.fn(async () => ({ job_id: "test-upload", status: "queued" })),
  fetchJobStatus: vi.fn(async () => ({ status: "completed", progress: 100 })),
}));

import { fetchHealth, fetchJobs, fetchResult } from "./api.js";

beforeEach(() => {
  vi.clearAllMocks();
  fetchHealth.mockResolvedValue({ ok: true });
  fetchJobs.mockResolvedValue([
    { job_id: "demo", title: "Study-group app interview", topic: "", duration: 120 },
  ]);
  fetchResult.mockResolvedValue(testResult);
});

describe("App", () => {
  it("shows the brand and an online backend status", async () => {
    render(<App />);
    expect(screen.getByText("EchoSight Forensics")).toBeInTheDocument();
    expect(await screen.findByText(/Backend:\s*Online/)).toBeInTheDocument();
  });

  it("shows an offline state and job error when the backend fails", async () => {
    fetchJobs.mockRejectedValue(new Error("Cannot reach the backend at http://test. Is it running?"));
    render(<App />);
    await screen.findByText(/Could not load the job list/);
    // /jobs failing does not break the health probe: stay Online.
    expect(screen.getByTestId("backend-status").textContent).toMatch(/Backend:\s*Online/);
    expect(screen.getByRole("combobox", { name: /job/i })).toBeDisabled();
  });

  it("auto-selects the first job and renders its transcript, speakers and chapters", async () => {
    render(<App />);
    await userEvent.selectOptions(await screen.findByRole("combobox"), "demo");
    expect(await screen.findByTestId("transcript-list")).toBeInTheDocument();
    expect(screen.getByText(/core idea is to match students/i)).toBeInTheDocument();
    expect(screen.getAllByText("Speaker 3").length).toBeGreaterThan(0);
    expect((await screen.findAllByText("Priya")).length).toBeGreaterThan(0);
    expect(screen.getByTestId("chapter-list")).toBeInTheDocument();
    expect(screen.getByTestId("takeaway-list")).toBeInTheDocument();
    expect(screen.getByTestId("report-body")).toBeInTheDocument();
    expect(fetchResult).toHaveBeenCalledWith("demo");
  });

  it("renders a result load error when the endpoint fails", async () => {
    fetchResult.mockRejectedValue({ message: "Job not found" });
    render(<App />);
    await userEvent.selectOptions(await screen.findByRole("combobox"), "demo");
    expect(await screen.findByText(/Job not found/)).toBeInTheDocument();
  });
});
