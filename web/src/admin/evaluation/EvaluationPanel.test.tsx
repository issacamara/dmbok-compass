import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EvaluationPanel } from "./EvaluationPanel";

const api = vi.hoisted(() => ({
  launchEvaluation: vi.fn(),
  getEvaluationRun: vi.fn(),
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
    await user.type(screen.getByLabelText("Item IDs"), "item-1, item-2");
    await user.click(screen.getByRole("button", { name: "Run subset evaluation" }));

    await waitFor(() => expect(api.launchEvaluation).toHaveBeenCalledWith("admin-token", expect.objectContaining({ item_ids: ["item-1", "item-2"] })));
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
