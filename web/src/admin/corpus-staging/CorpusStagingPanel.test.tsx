import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CorpusStagingPanel } from "./CorpusStagingPanel";

const api = vi.hoisted(() => ({
  getCorpusStagingStatus: vi.fn(), submitPdfOnlyCorpusStagingAttempt: vi.fn(), startCorpusStagingParse: vi.fn(),
  proposeCorpusStagingStructure: vi.fn(), correctCorpusStagingStructure: vi.fn(), approveCorpusStagingAttempt: vi.fn(),
}));
vi.mock("../../api/admin", () => api);

const entry = { entry_index: 0, section_path: ["Chapter 1"], page_start: 1, page_end: 2, match_status: "matched", correction_codes: [], block_ids: ["block-1"], excerpt: "Source text excerpt" };
const reviewed = { attempt_id: "attempt-1", revision: 3, state: "awaiting_review", completed_stage: "review", preview: [entry], correction_codes: [], parent_count: 0, child_count: 0, chunk_count: 0, staged_corpus_version_id: null, hosted_parsing_state: "awaiting_review", parsing_output_sha256: "a".repeat(64), error: null };

describe("CorpusStagingPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks(); window.localStorage.clear();
    api.getCorpusStagingStatus.mockResolvedValue(reviewed);
  });

  it("opens and remembers an existing staging attempt", async () => {
    const user = userEvent.setup();
    api.getCorpusStagingStatus.mockResolvedValue({ ...reviewed, attempt_id: "saved-attempt" });
    render(<CorpusStagingPanel token="admin-token" />);

    await user.type(screen.getByLabelText("Open an existing staging attempt"), " saved-attempt ");
    await user.click(screen.getByRole("button", { name: "Open attempt" }));

    expect(await screen.findByText(/Attempt ID:/)).toBeInTheDocument();
    expect(screen.getByText("saved-attempt")).toBeInTheDocument();
    expect(api.getCorpusStagingStatus).toHaveBeenCalledWith("admin-token", "saved-attempt");
    expect(window.localStorage.getItem("dmbok-compass.corpus-staging-attempt")).toBe("saved-attempt");
  });

  it("proposes an outline for a recovered parse before staging review", async () => {
    const user = userEvent.setup();
    api.getCorpusStagingStatus.mockResolvedValue({ ...reviewed, attempt_id: "saved-attempt", state: "validating", preview: null });
    api.proposeCorpusStagingStructure.mockResolvedValue({ ...reviewed, attempt_id: "saved-attempt" });
    render(<CorpusStagingPanel token="admin-token" />);

    await user.type(screen.getByLabelText("Open an existing staging attempt"), "saved-attempt");
    await user.click(screen.getByRole("button", { name: "Open attempt" }));

    expect(await screen.findByText("Match: matched")).toBeInTheDocument();
    expect(api.proposeCorpusStagingStructure).toHaveBeenCalledWith("admin-token", "saved-attempt");
  });

  it("requires a unique physical page selection before submitting", async () => {
    const user = userEvent.setup();
    render(<CorpusStagingPanel token="admin-token" />);
    await user.upload(screen.getByLabelText("Source PDF"), new File(["pdf"], "source.pdf", { type: "application/pdf" }));
    await user.clear(screen.getByLabelText("Physical pages to send for parsing"));
    await user.type(screen.getByLabelText("Physical pages to send for parsing"), "21, 21");
    await user.click(screen.getByRole("button", { name: "Upload PDF and parse" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/unique physical page numbers/i);
    expect(api.submitPdfOnlyCorpusStagingAttempt).not.toHaveBeenCalled();
  });
  afterEach(() => vi.restoreAllMocks());

  it("uploads only a PDF, proposes an outline, saves corrections, and approves the exact revision", async () => {
    const user = userEvent.setup();
    api.submitPdfOnlyCorpusStagingAttempt.mockResolvedValue({ ...reviewed, state: "validating", hosted_parsing_state: null, preview: null });
    api.startCorpusStagingParse.mockResolvedValue({ ...reviewed, state: "validating", hosted_parsing_state: "awaiting_review", preview: null });
    api.proposeCorpusStagingStructure.mockResolvedValue(reviewed);
    api.correctCorpusStagingStructure.mockResolvedValue({ ...reviewed, revision: 4, preview: [{ ...entry, section_path: ["Chapter 1", "Overview"] }] });
    api.approveCorpusStagingAttempt.mockResolvedValue({ ...reviewed, revision: 5, state: "queued", preview: null, hosted_parsing_state: "awaiting_review" });
    render(<CorpusStagingPanel token="admin-token" />);

    expect(screen.queryByLabelText(/structure json/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /structure template/i })).not.toBeInTheDocument();
    await user.upload(screen.getByLabelText("Source PDF"), new File(["pdf"], "source.pdf", { type: "application/pdf" }));
    await user.click(screen.getByRole("button", { name: "Upload PDF and parse" }));
    const submissionId = api.submitPdfOnlyCorpusStagingAttempt.mock.calls[0][3];
    expect(api.submitPdfOnlyCorpusStagingAttempt).toHaveBeenCalledWith("admin-token", expect.any(File), [21, 25, 58, 78, 105], expect.any(String));
    expect(window.localStorage.getItem("dmbok-compass.corpus-staging-submission")).not.toBe(submissionId);
    expect(await screen.findByLabelText("Section path for pages 1 to 2")).toHaveValue("Chapter 1");
    expect(api.proposeCorpusStagingStructure).toHaveBeenCalledWith("admin-token", "attempt-1");

    await user.clear(screen.getByLabelText("Section path for pages 1 to 2"));
    await user.type(screen.getByLabelText("Section path for pages 1 to 2"), "Chapter 1 / Overview");
    await user.click(screen.getByRole("button", { name: "Save outline corrections" }));
    await waitFor(() => expect(api.correctCorpusStagingStructure).toHaveBeenCalledWith("admin-token", "attempt-1", 3, [{ section_path: ["Chapter 1", "Overview"], block_ids: ["block-1"] }]));
    await user.click(screen.getByRole("button", { name: "Approve outline and create chunks" }));
    expect(api.approveCorpusStagingAttempt).toHaveBeenCalledWith("admin-token", "attempt-1", 4);
    expect(window.localStorage.getItem("dmbok-compass.corpus-staging-attempt")).toBe("attempt-1");
  });

  it("reuses the submission id after an uncertain upload across reloads", async () => {
    const user = userEvent.setup();
    api.submitPdfOnlyCorpusStagingAttempt.mockRejectedValueOnce(new Error("Upload response was lost"));
    const first = render(<CorpusStagingPanel token="admin-token" />);
    await user.upload(screen.getByLabelText("Source PDF"), new File(["pdf"], "source.pdf", { type: "application/pdf" }));
    await user.click(screen.getByRole("button", { name: "Upload PDF and parse" }));
    await screen.findByRole("alert");
    const submissionId = api.submitPdfOnlyCorpusStagingAttempt.mock.calls[0][3];
    first.unmount();

    api.submitPdfOnlyCorpusStagingAttempt.mockResolvedValue({ ...reviewed, state: "validating", hosted_parsing_state: null, preview: null });
    api.startCorpusStagingParse.mockResolvedValue({ ...reviewed, state: "validating", hosted_parsing_state: "awaiting_budget", preview: null });
    render(<CorpusStagingPanel token="admin-token" />);
    await user.upload(screen.getByLabelText("Source PDF"), new File(["pdf"], "source.pdf", { type: "application/pdf" }));
    await user.click(screen.getByRole("button", { name: "Upload PDF and parse" }));
    await waitFor(() => expect(api.submitPdfOnlyCorpusStagingAttempt).toHaveBeenCalledTimes(2));
    expect(api.submitPdfOnlyCorpusStagingAttempt.mock.calls[1][3]).toBe(submissionId);
  });

  it("reuses the same attempt after an uncertain parse start", async () => {
    const user = userEvent.setup();
    api.submitPdfOnlyCorpusStagingAttempt.mockResolvedValue({ ...reviewed, state: "validating", hosted_parsing_state: null, preview: null });
    api.startCorpusStagingParse.mockRejectedValueOnce(new Error("Parse response was lost"));
    const first = render(<CorpusStagingPanel token="admin-token" />);
    await user.upload(screen.getByLabelText("Source PDF"), new File(["pdf"], "source.pdf", { type: "application/pdf" }));
    await user.click(screen.getByRole("button", { name: "Upload PDF and parse" }));
    await screen.findByRole("alert");
    const submissionId = api.submitPdfOnlyCorpusStagingAttempt.mock.calls[0][3];
    expect(window.localStorage.getItem("dmbok-compass.corpus-staging-submission")).toBe(submissionId);
    first.unmount();

    api.startCorpusStagingParse.mockResolvedValue({ ...reviewed, state: "validating", hosted_parsing_state: "awaiting_budget", preview: null });
    render(<CorpusStagingPanel token="admin-token" />);
    await user.upload(screen.getByLabelText("Source PDF"), new File(["pdf"], "source.pdf", { type: "application/pdf" }));
    await user.click(screen.getByRole("button", { name: "Upload PDF and parse" }));
    await waitFor(() => expect(api.submitPdfOnlyCorpusStagingAttempt).toHaveBeenCalledTimes(2));
    expect(api.submitPdfOnlyCorpusStagingAttempt.mock.calls[1][3]).toBe(submissionId);
  });

  it("keeps incomplete matches editable and never implies approval", async () => {
    const user = userEvent.setup();
    const unmatched = { ...reviewed, preview: [{ ...entry, match_status: "unmatched", correction_codes: ["unmatched_passage"] }] };
    api.submitPdfOnlyCorpusStagingAttempt.mockResolvedValue({ ...unmatched, state: "validating", hosted_parsing_state: null, preview: null });
    api.startCorpusStagingParse.mockResolvedValue({ ...unmatched, state: "validating", hosted_parsing_state: "awaiting_review", preview: null });
    api.proposeCorpusStagingStructure.mockResolvedValue(unmatched);
    api.correctCorpusStagingStructure.mockResolvedValue({ ...reviewed, revision: 4, preview: [{ ...entry, section_path: ["Chapter 1", "Recovered section"] }] });
    api.approveCorpusStagingAttempt.mockResolvedValue({ ...reviewed, revision: 5, state: "queued", preview: null });
    render(<CorpusStagingPanel token="admin-token" />);
    await user.upload(screen.getByLabelText("Source PDF"), new File(["pdf"], "source.pdf", { type: "application/pdf" }));
    await user.click(screen.getByRole("button", { name: "Upload PDF and parse" }));
    expect(await screen.findByText("Match: unmatched")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Approve outline and create chunks" })).toBeDisabled();
    expect(api.approveCorpusStagingAttempt).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Save outline corrections" })).toBeEnabled();
    await user.clear(screen.getByLabelText("Section path for pages 1 to 2"));
    await user.type(screen.getByLabelText("Section path for pages 1 to 2"), "Chapter 1 / Recovered section");
    await user.click(screen.getByRole("button", { name: "Save outline corrections" }));
    await waitFor(() => expect(api.correctCorpusStagingStructure).toHaveBeenCalledWith("admin-token", "attempt-1", 3, [{ section_path: ["Chapter 1", "Recovered section"], block_ids: ["block-1"] }]));
    expect(await screen.findByText("Match: matched")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Approve outline and create chunks" }));
    expect(api.approveCorpusStagingAttempt).toHaveBeenCalledWith("admin-token", "attempt-1", 4);
  });

  it("keeps awaiting-budget attempts actionable without treating them as reviewed", async () => {
    const user = userEvent.setup();
    api.submitPdfOnlyCorpusStagingAttempt.mockResolvedValue({ ...reviewed, state: "validating", hosted_parsing_state: null, preview: null });
    api.startCorpusStagingParse
      .mockResolvedValueOnce({ ...reviewed, state: "validating", hosted_parsing_state: "awaiting_budget", preview: null })
      .mockResolvedValueOnce({ ...reviewed, state: "validating", hosted_parsing_state: "awaiting_review", preview: null });
    api.proposeCorpusStagingStructure.mockResolvedValue(reviewed);
    render(<CorpusStagingPanel token="admin-token" />);
    await user.upload(screen.getByLabelText("Source PDF"), new File(["pdf"], "source.pdf", { type: "application/pdf" }));
    await user.click(screen.getByRole("button", { name: "Upload PDF and parse" }));
    expect(await screen.findByText(/Hosted parsing is waiting for available budget/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry parsing" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Approve outline and create chunks" })).toBeDisabled();
    expect(api.proposeCorpusStagingStructure).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Retry parsing" }));
    expect(await screen.findByLabelText("Section path for pages 1 to 2")).toHaveValue("Chapter 1");
    expect(api.proposeCorpusStagingStructure).toHaveBeenCalledWith("admin-token", "attempt-1");
  });

  it("shows and polls ingestion after approval while the hosted parse state stays awaiting review", async () => {
    const user = userEvent.setup();
    window.localStorage.setItem("dmbok-compass.corpus-staging-attempt", "approved-id");
    api.getCorpusStagingStatus
      .mockResolvedValueOnce(reviewed)
      .mockResolvedValueOnce({ ...reviewed, state: "running", hosted_parsing_state: "awaiting_review", preview: null });
    api.approveCorpusStagingAttempt.mockResolvedValue({ ...reviewed, state: "queued", hosted_parsing_state: "awaiting_review", preview: null });
    render(<CorpusStagingPanel token="admin-token" />);
    await waitFor(() => expect(api.getCorpusStagingStatus).toHaveBeenCalledTimes(1));
    await user.click(await screen.findByRole("button", { name: "Approve outline and create chunks" }));
    expect(await screen.findByText(/Status: queued/)).toBeInTheDocument();
    expect(await screen.findByText(/Status: running/, {}, { timeout: 3500 })).toBeInTheDocument();
    expect(api.getCorpusStagingStatus).toHaveBeenCalledTimes(2);
  });

  it("resumes an existing parse job without requesting a new upload", async () => {
    const user = userEvent.setup();
    const parsing = { ...reviewed, state: "validating", preview: null, hosted_parsing_state: "parsing" };
    window.localStorage.setItem("dmbok-compass.corpus-staging-attempt", parsing.attempt_id);
    api.getCorpusStagingStatus.mockResolvedValueOnce(parsing).mockResolvedValue(parsing);
    api.startCorpusStagingParse.mockResolvedValue(parsing);
    render(<CorpusStagingPanel token="admin-token" />);

    await user.click(await screen.findByRole("button", { name: "Resume parsing" }));

    expect(api.startCorpusStagingParse).toHaveBeenCalledWith("admin-token", parsing.attempt_id);
    expect(api.submitPdfOnlyCorpusStagingAttempt).not.toHaveBeenCalled();
    expect(await screen.findByText(/resumed the saved parse job/i)).toBeInTheDocument();
  });

  it("resumes the saved attempt, handles stale revisions, and keeps refresh available", async () => {
    const user = userEvent.setup();
    window.localStorage.setItem("dmbok-compass.corpus-staging-attempt", "saved-id");
    api.getCorpusStagingStatus.mockResolvedValue(reviewed);
    api.approveCorpusStagingAttempt.mockRejectedValue({ status: 409, code: "stale_staging_attempt" });
    render(<CorpusStagingPanel token="admin-token" />);
    await waitFor(() => expect(api.getCorpusStagingStatus).toHaveBeenCalledWith("admin-token", "saved-id"));
    await user.click(await screen.findByRole("button", { name: "Approve outline and create chunks" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/revision is stale/i);
    expect(screen.getByRole("button", { name: "Refresh status" })).toBeEnabled();
    expect(screen.getByRole("link", { name: "Review staging controls" })).toHaveAttribute("href", "#staging-refresh");
  });

  it("does not offer retry for a terminally failed parse and keeps refresh available", async () => {
    const failed = { ...reviewed, state: "failed", hosted_parsing_state: "failed", preview: null };
    window.localStorage.setItem("dmbok-compass.corpus-staging-attempt", failed.attempt_id);
    api.getCorpusStagingStatus.mockResolvedValue(failed);
    render(<CorpusStagingPanel token="admin-token" />);

    expect(await screen.findByText(/upload the PDF to start a new attempt/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Retry parsing" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Refresh status" })).toBeEnabled();
  });

  it("reports a terminal parse conflict without calling it a stale review", async () => {
    const user = userEvent.setup();
    const parsing = { ...reviewed, state: "validating", preview: null, hosted_parsing_state: "parsing" };
    window.localStorage.setItem("dmbok-compass.corpus-staging-attempt", parsing.attempt_id);
    api.getCorpusStagingStatus.mockResolvedValue(parsing);
    api.startCorpusStagingParse.mockRejectedValue({ status: 409, code: "staging_attempt_not_parseable" });
    render(<CorpusStagingPanel token="admin-token" />);

    await user.click(await screen.findByRole("button", { name: "Resume parsing" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/cannot be resumed/i);
    expect(screen.getByRole("alert")).not.toHaveTextContent(/revision is stale/i);
    expect(screen.getByRole("button", { name: "Refresh status" })).toBeEnabled();
  });

  it("renders bounded source excerpts and keyboard-usable controls at a narrow viewport", async () => {
    const user = userEvent.setup();
    const excerpt = "e".repeat(500);
    api.getCorpusStagingStatus.mockResolvedValue({ ...reviewed, preview: [{ ...entry, excerpt }] });
    window.localStorage.setItem("dmbok-compass.corpus-staging-attempt", "excerpt-id");
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 390 });
    render(<CorpusStagingPanel token="admin-token" />);
    const region = screen.getByRole("region", { name: "Stage a corpus" });
    expect(within(region).getByLabelText("Source PDF")).toHaveAttribute("accept", "application/pdf,.pdf");
    expect((await within(region).findByText(excerpt)).textContent).toHaveLength(500);
    await user.upload(screen.getByLabelText("Source PDF"), new File(["pdf"], "source.pdf", { type: "application/pdf" }));
    screen.getByLabelText("Source PDF").focus();
    expect(screen.getByLabelText("Source PDF")).toHaveFocus();
    await user.tab();
    expect(screen.getByLabelText("Physical pages to send for parsing")).toHaveFocus();
    await user.tab();
    expect(screen.getByRole("button", { name: "Upload PDF and parse" })).toHaveFocus();
  });

  it("states that staging does not change the active corpus", async () => {
    window.localStorage.setItem("dmbok-compass.corpus-staging-attempt", "staged-id");
    api.getCorpusStagingStatus.mockResolvedValue({ ...reviewed, state: "staged", staged_corpus_version_id: "candidate-1", parent_count: 2, chunk_count: 4, preview: null });
    render(<CorpusStagingPanel token="admin-token" />);
    expect(await screen.findByText(/The active corpus is unchanged/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Approve outline and create chunks" })).toBeDisabled();
  });
});
