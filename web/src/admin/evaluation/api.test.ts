import { beforeEach, describe, expect, it, vi } from "vitest";
import { attestBatchReview, getReviewSummary } from "./api";

describe("batch evidence review API", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("loads the run-bound browser-safe review projection", async () => {
    const response = { summary: {}, sample: [], exceptions: [] };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify(response), { status: 200 }));
    await getReviewSummary("admin-token", "run/1");
    expect(fetchMock).toHaveBeenCalledWith("/api/admin/evaluations/run%2F1/review-summary", expect.objectContaining({ headers: { Authorization: "Bearer admin-token" } }));
  });

  it("sends only the defined batch decision and explicit exception decisions", async () => {
    const payload = { decision: "attested" as const, exception_decisions: [{ item_id: "service-provided-item", decision: "inapplicable" as const, reason: "Outside scope." }], reason: null };
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
});
