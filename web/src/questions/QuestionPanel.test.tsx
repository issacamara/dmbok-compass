import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AnswerResponse } from "../api/contracts";
import { QuestionPanel } from "./QuestionPanel";

const quota = {
  user_used: 1,
  user_limit: 100,
  global_used: 4,
  global_limit: 100,
  resets_at: "2030-01-01T00:00:00Z",
};

function response(outcome: AnswerResponse["outcome"]): AnswerResponse {
  return {
    outcome,
    answer_text: outcome === "refusal" ? "I could not find enough relevant evidence." : `${outcome} answer`,
    synthesis: outcome === "answer",
    citations: outcome === "refusal" ? [] : [{ citation_id: "cite-1", page: 69, section: "Data Governance", excerpt: "Governance evidence." }],
    trace: {
      retrieved_passages: [{ chunk_id: "chunk-1", page: 69, section: "Data Governance", excerpt: "Governance evidence.", relevance_score: .91 }],
      selected_model: "test-model",
      timings_ms: { retrieval_ms: 10, total_ms: 20 },
    },
    quota,
  };
}

function mockApi(answer: AnswerResponse) {
  vi.stubGlobal("fetch", vi.fn().mockImplementation((path: string) => Promise.resolve({
    ok: true,
    json: async () => path === "/api/quota" ? quota : answer,
  })));
}

describe("QuestionPanel", () => {
  beforeEach(() => vi.restoreAllMocks());

  it.each([
    ["answer", "Strong evidence"],
    ["qualified", "Qualified answer · partial evidence"],
    ["refusal", "Evidence-based refusal"],
  ] as const)("renders the %s outcome with its evidence label", async (outcome, label) => {
    mockApi(response(outcome));
    const user = userEvent.setup();
    render(<QuestionPanel token="firebase-token" onSignOut={vi.fn()} />);

    await user.type(await screen.findByLabelText("Ask a DMBOK question"), "What is data governance?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText(label)).toBeInTheDocument();
    expect(screen.getByText(response(outcome).answer_text!)).toBeInTheDocument();
    if (outcome === "refusal") expect(screen.queryByLabelText("Citations")).not.toBeInTheDocument();
  });

  it("clears the answer and question from the active session", async () => {
    mockApi(response("answer"));
    const user = userEvent.setup();
    render(<QuestionPanel token="firebase-token" onSignOut={vi.fn()} />);
    const input = await screen.findByLabelText("Ask a DMBOK question");
    await user.type(input, "A temporary question");
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("answer answer")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(screen.queryByText("answer answer")).not.toBeInTheDocument();
    expect(input).toHaveValue("");
    await waitFor(() => expect(input).toHaveFocus());
  });

  it("shows which provider routes failed after retrieval", async () => {
    const failed = response("refusal");
    failed.answer_text = undefined;
    failed.error = { code: "answer_provider_unavailable", message: "The model provider is unavailable.", retryable: true, status: 503 };
    failed.trace.selected_model = "fallback";
    failed.trace.model_attempts = [
      { model: "primary", outcome: "unavailable", status_code: 404 },
      { model: "fallback", outcome: "unavailable", status_code: 503 },
    ];
    mockApi(failed);
    const user = userEvent.setup();
    render(<QuestionPanel token="firebase-token" onSignOut={vi.fn()} />);
    await user.type(await screen.findByLabelText("Ask a DMBOK question"), "List the dimensions");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("primary · unavailable (404)")).toBeInTheDocument();
    expect(screen.getByText("fallback · unavailable (503)")).toBeInTheDocument();
    expect(screen.getByText("The model provider is unavailable.")).toBeInTheDocument();
    expect(screen.getByText("Answer unavailable")).toBeInTheDocument();
    expect(screen.queryByText("Evidence-based refusal")).not.toBeInTheDocument();
  });

  it("exposes the composer guidance and answer controls to keyboard and assistive technology", async () => {
    mockApi(response("answer"));
    const user = userEvent.setup();
    render(<QuestionPanel token="firebase-token" onSignOut={vi.fn()} />);

    const input = await screen.findByLabelText("Ask a DMBOK question");
    expect(input).toHaveAttribute("aria-describedby", "question-help");
    expect(screen.getByText("Questions are answered from the approved DMBOK corpus.")).toHaveClass("visually-hidden");

    await user.type(input, "What is data governance?");
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Strong evidence")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Clear" })).toBeEnabled();
  });

  it("does not persist the question or trace in browser storage", async () => {
    mockApi(response("answer"));
    const user = userEvent.setup();
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    render(<QuestionPanel token="firebase-token" onSignOut={vi.fn()} />);
    await user.type(await screen.findByLabelText("Ask a DMBOK question"), "No saved question");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByText("answer answer");

    expect(setItem).not.toHaveBeenCalled();
  });
});
