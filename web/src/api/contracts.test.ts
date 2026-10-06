import { describe, expect, it } from "vitest";
import fixture from "../../../documents/contracts/api-contract-fixture.json";
import { API_ROUTES, TELEMETRY_FIELDS, type AnswerResponse } from "./contracts";

describe("frozen API contracts", () => {
  it("keeps shared enum and telemetry values aligned with the fixture", () => {
    expect(["answer", "qualified", "refusal"]).toEqual(fixture.outcomes);
    expect(TELEMETRY_FIELDS).toEqual(fixture.telemetryFields);
    expect(fixture.embeddingDimensions).toBe(768);
  });

  it("keeps logical route names stable", () => {
    expect(API_ROUTES.answer).toBe("/api/questions");
    expect(API_ROUTES.currentUser).toBe("/api/me");
  });

  it("models the answer response without durable interaction ownership", () => {
    const response: AnswerResponse = {
      outcome: "qualified",
      answer_text: "Limited answer",
      synthesis: false,
      citations: [],
      trace: { retrieved_passages: [], selected_model: "test", timings_ms: {} },
      quota: {
        user_used: 1,
        user_limit: 100,
        global_used: 1,
        global_limit: 100,
        resets_at: new Date().toISOString(),
      },
    };
    expect(response.trace.retrieved_passages).toHaveLength(0);
    expect(fixture.ephemeralInteractionFields).toContain("question");
  });
});
