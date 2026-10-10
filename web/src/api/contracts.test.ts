import { describe, expect, it } from "vitest";
import fixture from "../../../documents/contracts/api-contract-fixture.json";
import {
  API_ROUTES,
  MAX_APPROVED_GOLD_ITEMS,
  MAX_EVALUATION_IMPORT_BYTES,
  MAX_EVALUATION_IMPORT_ITEMS,
  MIN_APPROVED_GOLD_ITEMS,
  TELEMETRY_FIELDS,
  type AnswerResponse,
  type EvaluationImportResponse,
  type EvaluationReportEligibility,
} from "./contracts";

describe("frozen API contracts", () => {
  it("keeps shared enum and telemetry values aligned with the fixture", () => {
    expect(["answer", "qualified", "refusal"]).toEqual(fixture.outcomes);
    expect(TELEMETRY_FIELDS).toEqual(fixture.telemetryFields);
    expect(fixture.embeddingDimensions).toBe(768);
  });

  it("keeps logical route names stable", () => {
    expect(API_ROUTES.answer).toBe("/api/questions");
    expect(API_ROUTES.registration).toBe("/api/registration");
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

  it("freezes import bounds, categories, and error shape", () => {
    expect(MAX_EVALUATION_IMPORT_BYTES).toBe(fixture.evaluation.maxImportBytes);
    expect(MAX_EVALUATION_IMPORT_ITEMS).toBe(fixture.evaluation.maxImportItems);
    expect(MIN_APPROVED_GOLD_ITEMS).toBe(fixture.evaluation.minApprovedGoldItems);
    expect(MAX_APPROVED_GOLD_ITEMS).toBe(fixture.evaluation.maxApprovedGoldItems);

    const response: EvaluationImportResponse = {
      status: "accepted",
      generation_id: "generation-2",
      submitted_count: 2,
      imported_count: 1,
      skipped_count: 1,
      validation_errors: [{ index: 1, code: "invalid_category", message: "Unsupported category." }],
    };
    expect(response.validation_errors[0]?.index).toBe(1);
    expect(fixture.evaluation.categories).toContain("scenarios");
  });

  it("models replacement visibility and the 30–50 gold gate", () => {
    const eligibility: EvaluationReportEligibility = {
      generation_id: "generation-2",
      eligibility: "release_evidence",
      approved_gold_count: 30,
      is_current_generation: true,
      is_superseded: false,
    };
    expect(eligibility.eligibility).toBe("release_evidence");
  });
});
