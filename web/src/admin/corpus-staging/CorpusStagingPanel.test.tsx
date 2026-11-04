import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CorpusStagingPanel } from "./CorpusStagingPanel";

const api = vi.hoisted(() => ({ getCorpusStagingStatus: vi.fn(), getCorpusStagingTemplate: vi.fn(), submitCorpusStagingAttempt: vi.fn(), validateCorpusStagingAttempt: vi.fn(), approveCorpusStagingAttempt: vi.fn() }));
vi.mock("../../api/admin", () => api);

const reviewed = { attempt_id: "attempt-1", revision: 3, state: "awaiting_review", completed_stage: "review", preview: [{ entry_index: 0, section_path: ["Chapter 1"], page_start: 1, page_end: 2, match_status: "matched", correction_codes: [] }], correction_codes: [], parent_count: 0, child_count: 0, chunk_count: 0, staged_corpus_version_id: null, error: null };

describe("CorpusStagingPanel", () => {
  beforeEach(() => { vi.clearAllMocks(); window.localStorage.clear(); api.getCorpusStagingStatus.mockResolvedValue(reviewed); });

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
});
