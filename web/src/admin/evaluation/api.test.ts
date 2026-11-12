import { beforeEach, describe, expect, it, vi } from "vitest";
import { addEvaluationDatasetRow, assessRunItem, attestBatchReview, decideIndividualReview, getActiveDataset, getEvaluationDatasetRevision, getReviewSummary, getRunItemAssessmentContext, launchActiveEvaluation, removeEvaluationDatasetRow } from "./api";

describe("batch evidence review API", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("loads the run-bound browser-safe review projection", async () => {
    const response = { summary: {}, sample: [], exceptions: [] };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify(response), { status: 200 }));
    await getReviewSummary("admin-token", "run/1");
    expect(fetchMock).toHaveBeenCalledWith("/api/admin/evaluations/run%2F1/review-summary", expect.objectContaining({ headers: { Authorization: "Bearer admin-token" } }));
  });

  it("sends only the defined batch decision and explicit exception decisions", async () => {
    const payload = { decision: "attested" as const, exception_decisions: [{ item_id: "service-provided-item", decision: "inapplicable" as const, reason: "Outside scope." }], semantic_assessments: [{ item_id: "sample-item", assessment_id: "assessment-id" }], refusal_confirmation_item_ids: ["refusal-item"], reason: null };
    const response = { status: "attested", decision: "attested", exception_decisions: payload.exception_decisions, decision_metadata: { decision: "attested", actor: "admin", decided_at: "2026-11-05T00:00:00Z", decision_version: "v1" } };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify(response), { status: 200 }));
    await attestBatchReview("admin-token", "run-1", payload);
    expect(fetchMock).toHaveBeenCalledWith("/api/admin/evaluations/run-1/batch-attestation", expect.objectContaining({ method: "POST", body: JSON.stringify(payload), headers: { Authorization: "Bearer admin-token", "Content-Type": "application/json" } }));
  });

  it("preserves safe authorization and stale-conflict statuses for the panel", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ detail: { code: "stale_batch_review", message: "The batch review snapshot is stale." } }), { status: 409 }));
    await expect(getReviewSummary("token", "run-1")).rejects.toMatchObject({ status: 409, message: "The batch review snapshot is stale." });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("loads the active dataset and launches full or selected server-resolved subsets", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify({}), { status: 200 }));
    await getActiveDataset("token");
    await launchActiveEvaluation("token");
    await launchActiveEvaluation("token", ["selected-id"]);
    expect(fetchMock).toHaveBeenNthCalledWith(1, "/api/admin/evaluation-datasets/active", expect.anything());
    expect(fetchMock).toHaveBeenNthCalledWith(2, "/api/evaluations", expect.objectContaining({ method: "POST", body: JSON.stringify({ item_ids: null }) }));
    expect(fetchMock).toHaveBeenNthCalledWith(3, "/api/evaluations", expect.objectContaining({ method: "POST", body: JSON.stringify({ item_ids: ["selected-id"] }) }));
  });

  it("loads the versioned rows and sends revision-bound optional edits", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify({ revision: 3, dataset_sha256: "a".repeat(64), rows: [] }), { status: 200 }));
    await getEvaluationDatasetRevision("token");
    await addEvaluationDatasetRow("token", { expected_revision: 3, operation_key: "operation-1", question: "What is data quality?", category: "definitions", expected_outcome: "answer", expected_answer_or_rubric: "Explain it.", supporting_passage: "Matching source text.", scope_rationale: null });
    await removeEvaluationDatasetRow("token", "item/1", { expected_revision: 3, operation_key: "operation-2" });
    expect(fetchMock).toHaveBeenNthCalledWith(1, "/api/admin/evaluation-datasets/revision", expect.anything());
    expect(fetchMock).toHaveBeenNthCalledWith(2, "/api/admin/evaluation-datasets/rows", expect.objectContaining({ method: "POST", body: expect.stringContaining('"expected_revision":3') }));
    expect(fetchMock).toHaveBeenNthCalledWith(3, "/api/admin/evaluation-datasets/rows/item%2F1/remove", expect.objectContaining({ method: "POST", body: JSON.stringify({ expected_revision: 3, operation_key: "operation-2" }) }));
  });

  it("sends an individual decision without candidate or chunk selectors", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({}), { status: 200 }));
    await decideIndividualReview("token", "run-1", "item-1", "reject");
    expect(fetchMock).toHaveBeenCalledWith("/api/admin/evaluations/run-1/items/item-1/review", expect.objectContaining({ method: "PUT", body: JSON.stringify({ decision: "reject" }) }));
  });

  it("loads and saves run-bound exact-output assessment context", async () => {
    const context = { run_id: "run-1", item_id: "item-1", generation_id: "generation-1", expected_refusal: true, output_text: "A refusal.", attempt_id: "server-attempt", candidate_id: "server-candidate", scorer_version: "scorer-v1", output_digest: "b".repeat(64), assessment: null };
    const response = { ...context, assessment: { assessment_id: "assessment-1", claim_support: true, citation_support: true, rubric_passed: true, refusal_judgment: true, reviewed_at: "2026-11-05T00:00:00Z" } };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => new Response(JSON.stringify(init?.method === "PUT" ? response : context), { status: 200 }));
    await getRunItemAssessmentContext("token", "run/1", "item/1");
    await assessRunItem("token", "run/1", "item/1", { generation_id: "generation-1", attempt_id: "server-attempt", candidate_id: "server-candidate", scorer_version: "scorer-v1", output_digest: "b".repeat(64), claim_support: true, citation_support: true, rubric_passed: true, refusal_judgment: true });
    expect(fetchMock).toHaveBeenNthCalledWith(1, "/api/admin/evaluations/run%2F1/items/item%2F1/assessment", expect.anything());
    expect(fetchMock).toHaveBeenNthCalledWith(2, "/api/admin/evaluations/run%2F1/items/item%2F1/assessment", expect.objectContaining({ method: "PUT", body: expect.stringContaining('"attempt_id":"server-attempt"') }));
  });
});
