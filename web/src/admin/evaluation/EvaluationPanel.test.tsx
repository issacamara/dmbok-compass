import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EvaluationPanel } from "./EvaluationPanel";

const api = vi.hoisted(() => ({
  launchEvaluation: vi.fn(),
  getEvaluationRun: vi.fn(),
  importEvaluationDataset: vi.fn(),
  getActiveEvaluationGeneration: vi.fn(),
  reviewEvaluationItem: vi.fn(),
  recordReleaseDecision: vi.fn(),
}));

vi.mock("./api", () => api);

const completedRun = {
  run_id: "eval-1", dataset_version_id: "dataset-v1", corpus_version_id: "corpus-v1",
  configuration_version_id: "config-v1", model_version_id: "model-v1", status: "completed",
  selected_item_ids: ["item-1"], item_results: [], error: null,
  metrics: [{ metric_name: "retrieval_success", numerator: 1, denominator: 1, percentage: 100, threshold: 90, passed: true }],
};

describe("EvaluationPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.launchEvaluation.mockResolvedValue(completedRun);
    api.getActiveEvaluationGeneration.mockResolvedValue({
      generation_id: "generation-v1", item_count: 2, approved_gold_count: 0, status: "active",
      item_reviews: [
        { generation_id: "generation-v1", item_id: "item-1", review_status: "candidate" },
        { generation_id: "generation-v1", item_id: "item-2", review_status: "candidate" },
      ],
    });
    api.recordReleaseDecision.mockResolvedValue({ ...completedRun, decision: "approved" });
  });

  it("launches the full evaluation and displays version-bound gate evidence", async () => {
    const user = userEvent.setup();
    render(<EvaluationPanel token="admin-token" />);

    await user.click(screen.getByRole("button", { name: "Run full evaluation" }));
    await waitFor(() => expect(api.launchEvaluation).toHaveBeenCalledWith("admin-token", {
      corpus_version_id: "corpus-v1",
      configuration_version_id: "config-v1", model_version_id: "model-v1",
    }));
    expect(await screen.findByText("Top-5 retrieval success")).toBeInTheDocument();
    expect(screen.getByText("100% · threshold 90% · Pass")).toBeInTheDocument();
    expect(screen.getByText("dataset-v1")).toBeInTheDocument();
  });

  it("sends only selected IDs for a subset run", async () => {
    const user = userEvent.setup();
    render(<EvaluationPanel token="admin-token" />);
    await user.click(screen.getByLabelText("Selected subset"));
    await user.click(await screen.findByRole("checkbox", { name: /item-1/ }));
    await user.click(screen.getByRole("checkbox", { name: /item-2/ }));
    await user.click(screen.getByRole("button", { name: "Run subset evaluation" }));

    await waitFor(() => expect(api.launchEvaluation).toHaveBeenCalledWith("admin-token", expect.objectContaining({ item_ids: ["item-1", "item-2"] })));
  });

  it("imports mixed records, refreshes the active generation, and reviews a candidate", async () => {
    const user = userEvent.setup();
    api.importEvaluationDataset.mockResolvedValue({
      status: "accepted", generation_id: "generation-v2", submitted_count: 2, imported_count: 1,
      skipped_count: 1, validation_errors: [{ index: 1, code: "missing_question", message: "Record requires a question." }],
    });
    render(<EvaluationPanel token="admin-token" />);
    const file = new File([JSON.stringify([{ question: "What is governance?" }])], "evaluation.json", { type: "application/json" });
    await user.upload(screen.getByLabelText("Evaluation JSON"), file);
    await waitFor(() => expect(api.importEvaluationDataset).toHaveBeenCalledWith("admin-token", [{ question: "What is governance?" }]));
    expect(await screen.findByRole("status")).toHaveTextContent("Imported 1 of 2 records; 1 skipped.");
    await user.click(screen.getAllByRole("button", { name: "Approve gold" })[0]);
    await waitFor(() => expect(api.reviewEvaluationItem).toHaveBeenCalledWith("admin-token", "generation-v1", "item-1", "approved"));
  });

  it("requires explicit confirmation before recording a sponsor decision", async () => {
    const user = userEvent.setup();
    render(<EvaluationPanel token="admin-token" />);
    await user.click(screen.getByRole("button", { name: "Run full evaluation" }));
    await screen.findByText("Sponsor decision");
    await user.type(screen.getByLabelText("Gate report IDs"), "report-1");
    await user.type(screen.getByLabelText("Rationale"), "All release gates passed.");
    const submit = screen.getByRole("button", { name: "Record sponsor decision" });
    expect(submit).toBeDisabled();
    await user.click(screen.getByLabelText(/I confirm this sponsor decision/));
    await user.click(submit);

    await waitFor(() => expect(api.recordReleaseDecision).toHaveBeenCalledWith("admin-token", expect.objectContaining({
      release_id: "release-eval-1", decision: "approved", gate_report_ids: ["report-1"],
    })));
    expect(await screen.findByRole("status")).toHaveTextContent("Release decision recorded: approved.");
  });
});
