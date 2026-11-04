import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CorpusStagingPanel } from "./CorpusStagingPanel";

const api = vi.hoisted(() => ({ getCorpusStagingStatus: vi.fn(), getCorpusStagingTemplate: vi.fn(), submitCorpusStagingAttempt: vi.fn(), validateCorpusStagingAttempt: vi.fn(), approveCorpusStagingAttempt: vi.fn() }));
vi.mock("../../api/admin", () => api);

const reviewed = { attempt_id: "attempt-1", revision: 3, state: "awaiting_review", completed_stage: "review", preview: [{ entry_index: 0, section_path: ["Chapter 1"], page_start: 1, page_end: 2, match_status: "matched", correction_codes: [] }], correction_codes: [], parent_count: 0, child_count: 0, chunk_count: 0, staged_corpus_version_id: null, error: null };

describe("CorpusStagingPanel", () => {
  beforeEach(() => { vi.clearAllMocks(); window.localStorage.clear(); api.getCorpusStagingStatus.mockResolvedValue(reviewed); });
  afterEach(() => vi.restoreAllMocks());

  it("uploads paired files, reviews, and approves the exact revision", async () => {
    const user = userEvent.setup();
    api.submitCorpusStagingAttempt.mockResolvedValue({ ...reviewed, state: "validating", preview: null });
    api.validateCorpusStagingAttempt.mockResolvedValue(reviewed);
    api.approveCorpusStagingAttempt.mockResolvedValue({ ...reviewed, revision: 4, state: "queued", preview: null });
    render(<CorpusStagingPanel token="admin-token" />);
    await user.upload(screen.getByLabelText("Source PDF"), new File(["pdf"], "source.pdf", { type: "application/pdf" }));
    await user.upload(screen.getByLabelText("Structure JSON"), new File(["{}"], "structure.json", { type: "application/json" }));
    await user.click(screen.getByRole("button", { name: "Upload and preview" }));
    expect(await screen.findByText("Chapter 1")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Approve and create chunks" }));
    expect(api.approveCorpusStagingAttempt).toHaveBeenCalledWith("admin-token", "attempt-1", 3);
    expect(window.localStorage.getItem("dmbok-compass.corpus-staging-attempt")).toBe("attempt-1");
  });

  it("keeps approval disabled for unmatched previews and explains corrections", async () => {
    const user = userEvent.setup();
    api.submitCorpusStagingAttempt.mockResolvedValue({ ...reviewed, state: "validating", preview: null });
    api.validateCorpusStagingAttempt.mockResolvedValue({ ...reviewed, preview: [{ ...reviewed.preview[0], match_status: "unmatched", correction_codes: ["unmatched_passage"] }] });
    render(<CorpusStagingPanel token="admin-token" />);
    await user.upload(screen.getByLabelText("Source PDF"), new File(["pdf"], "source.pdf", { type: "application/pdf" }));
    await user.upload(screen.getByLabelText("Structure JSON"), new File(["{}"], "structure.json", { type: "application/json" }));
    await user.click(screen.getByRole("button", { name: "Upload and preview" }));
    expect(await screen.findByText(/no matching section/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Approve and create chunks" })).toBeDisabled();
    expect(api.approveCorpusStagingAttempt).not.toHaveBeenCalled();
  });

  it("resumes status by saved attempt ID and provides keyboard accessible controls", async () => {
    window.localStorage.setItem("dmbok-compass.corpus-staging-attempt", "saved-id");
    api.getCorpusStagingStatus.mockResolvedValue({ ...reviewed, state: "staged", staged_corpus_version_id: "candidate-1", parent_count: 2, chunk_count: 4, preview: null });
    render(<CorpusStagingPanel token="admin-token" />);
    await waitFor(() => expect(api.getCorpusStagingStatus).toHaveBeenCalledWith("admin-token", "saved-id"));
    expect(await screen.findByText(/Staged version candidate-1; 2 parents and 4 chunks/)).toBeInTheDocument();
    expect(screen.getByLabelText("Source PDF")).toBeInTheDocument();
    expect(screen.getByLabelText("Structure JSON")).toBeInTheDocument();
  });

  it("keeps the template download URL available through the click and revokes it later", async () => {
    api.getCorpusStagingTemplate.mockResolvedValue({ schema_version: "structure-template-v1", document_title: "DMBOK", instructions: [], entries: [] });
    const createUrl = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:template");
    const revokeUrl = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
      expect(this.href).toBe("blob:template");
      expect(revokeUrl).not.toHaveBeenCalled();
    });
    vi.useFakeTimers();
    try {
      render(<CorpusStagingPanel token="admin-token" />);
      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "Download structure template" }));
        await Promise.resolve(); await Promise.resolve();
      });
      expect(createUrl).toHaveBeenCalledOnce();
      expect(revokeUrl).not.toHaveBeenCalled();
      await act(async () => { vi.advanceTimersByTime(1000); });
      expect(revokeUrl).toHaveBeenCalledWith("blob:template");
    } finally { vi.useRealTimers(); }
  });

  it("focuses an accessible error summary and links back to the upload controls", async () => {
    const user = userEvent.setup();
    api.getCorpusStagingTemplate.mockRejectedValue(new Error("offline"));
    render(<CorpusStagingPanel token="admin-token" />);
    await user.click(screen.getByRole("button", { name: "Download structure template" }));
    const summary = await screen.findByRole("alert");
    await waitFor(() => expect(summary).toHaveFocus());
    expect(summary).toHaveAttribute("tabindex", "-1");
    expect(within(summary).getByRole("link", { name: "Review staging controls" })).toHaveAttribute("href", "#staging-pdf");
  });

  it("shows a stale revision safely and refreshes the reviewed attempt", async () => {
    const user = userEvent.setup();
    api.getCorpusStagingStatus.mockResolvedValue({ ...reviewed, revision: 4 });
    api.approveCorpusStagingAttempt.mockRejectedValue({ status: 409 });
    api.submitCorpusStagingAttempt.mockResolvedValue({ ...reviewed, state: "validating", preview: null });
    api.validateCorpusStagingAttempt.mockResolvedValue(reviewed);
    render(<CorpusStagingPanel token="admin-token" />);
    await user.upload(screen.getByLabelText("Source PDF"), new File(["pdf"], "source.pdf", { type: "application/pdf" }));
    await user.upload(screen.getByLabelText("Structure JSON"), new File(["{}"], "structure.json", { type: "application/json" }));
    await user.click(screen.getByRole("button", { name: "Upload and preview" }));
    await user.click(await screen.findByRole("button", { name: "Approve and create chunks" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/reviewed attempt changed/i);
    expect(screen.getByRole("link", { name: "Review staging controls" })).toHaveAttribute("href", "#staging-refresh");
    await waitFor(() => expect(api.getCorpusStagingStatus).toHaveBeenCalledWith("admin-token", "attempt-1"));
    expect(await screen.findByText(/Status: awaiting review/)).toBeInTheDocument();
  });

  it("explains a failed attempt and its correction code", async () => {
    window.localStorage.setItem("dmbok-compass.corpus-staging-attempt", "failed-id");
    api.getCorpusStagingStatus.mockResolvedValue({ ...reviewed, attempt_id: "failed-id", state: "failed", completed_stage: "validation", preview: null, correction_codes: ["invalid_structure"], error: { code: "invalid_structure", message: "Correct the structure file and submit a new attempt.", retryable: false, status: 422 } });
    render(<CorpusStagingPanel token="admin-token" />);
    expect(await screen.findByText("Correct the structure file and submit a new attempt.")).toBeInTheDocument();
    expect(screen.getByText(/structure file is invalid/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Approve and create chunks" })).toBeDisabled();
  });

  it("keeps semantic controls, labels and keyboard focus usable at a narrow viewport", async () => {
    const user = userEvent.setup();
    api.getCorpusStagingTemplate.mockResolvedValue({ schema_version: "structure-template-v1", document_title: "DMBOK", instructions: [], entries: [] });
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:keyboard-template");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 390 });
    render(<CorpusStagingPanel token="admin-token" />);
    const panel = screen.getByRole("region", { name: "Stage a corpus" });
    expect(panel.querySelector("form")).toHaveClass("staging-form");
    expect(screen.getByLabelText("Source PDF")).toHaveAttribute("accept", "application/pdf,.pdf");
    expect(screen.getByLabelText("Structure JSON")).toHaveAttribute("accept", "application/json,.json");
    await user.tab();
    expect(screen.getByRole("button", { name: "Download structure template" })).toHaveFocus();
    await user.keyboard("{Enter}");
    await waitFor(() => expect(api.getCorpusStagingTemplate).toHaveBeenCalledWith("admin-token"));
    await user.tab();
    expect(screen.getByLabelText("Source PDF")).toHaveFocus();
  });
});
