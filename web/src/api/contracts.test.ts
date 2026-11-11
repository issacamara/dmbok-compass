import { describe, expect, it } from "vitest";
import fixture from "../../../documents/contracts/api-contract-fixture.json";
import architectureFixture from "../../../backend/tests/fixtures/architecture-v21.json";
import stagingFixture from "../../../documents/contracts/corpus-staging-contract-fixture.json";
import structureTemplateFixture from "../../../backend/app/templates/dmbok-structure-template-v1.json";
import {
  API_ROUTES,
  ADMIN_ROUTES,
  MAX_APPROVED_GOLD_ITEMS,
  MAX_EVALUATION_IMPORT_BYTES,
  MAX_EVALUATION_IMPORT_ITEMS,
  MIN_APPROVED_GOLD_ITEMS,
  TELEMETRY_FIELDS,
  type AnswerResponse,
  type EvaluationImportResponse,
  type EvaluationReportEligibility,
  type AnswerOutcome,
  type CandidateIdentity,
  type EvaluationCase,
  type RunItemResult,
  type RuntimeSettings,
  OPERATIONAL_ERROR_STATUS,
  type RetrievalResult,
  type BatchAttestationRequest,
  type BatchAttestationResult,
  type ExceptionRecord,
  type RepresentativeSampleEntry,
  type ReviewSummary,
  type CorpusStagingAttemptStatus,
  type StructureTemplate,
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

  it("freezes versioned, browser-safe corpus staging contracts", () => {
    const template: StructureTemplate = {
      schema_version: "structure-template-v1",
      document_title: "DAMA-DMBOK, Second Edition",
      instructions: ["Replace the example with the supplied PDF hierarchy."],
      entries: [{ section_path: ["Chapter 1", "1.1 Introduction"], page_start: 1, page_end: 3 }],
    };
    const attempt: CorpusStagingAttemptStatus = {
      attempt_id: "attempt-valid",
      revision: 2,
      state: "awaiting_review",
      completed_stage: "review",
      preview: [{
        entry_index: 0,
        section_path: ["Chapter 1", "1.1 Introduction"],
        page_start: 1,
        page_end: 3,
        match_status: "matched",
        correction_codes: [],
        block_ids: ["block-1"],
        excerpt: "Bounded source excerpt.",
      }],
      correction_codes: [],
      parent_count: 0,
      child_count: 0,
      chunk_count: 0,
      staged_corpus_version_id: null,
      hosted_parsing_state: "awaiting_review",
      parsing_output_sha256: "a".repeat(64),
      error: null,
    };
    expect(ADMIN_ROUTES.corpusStagingTemplate).toBe(API_ROUTES.corpusStagingTemplate);
    expect(template.schema_version).toBe(structureTemplateFixture.schema_version);
    expect(attempt.state).toBe(stagingFixture.attempts.valid.state);
    expect(stagingFixture.attempts.invalid.state).toBe("failed");
    expect(stagingFixture.attempts.stale.status).toBe(409);
    expect(stagingFixture.attempts.terminal.state).toBe("staged");
    expect(Object.keys(template)).not.toEqual(expect.arrayContaining(stagingFixture.forbiddenBrowserFields));
    expect(Object.keys(attempt)).not.toEqual(expect.arrayContaining(stagingFixture.forbiddenBrowserFields));
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

  it("keeps v2.1 candidate and retrieval identity aligned", () => {
    const candidate: CandidateIdentity = {
      artifact_id: "backend@2.1.0",
      installed_lock_sha256: "a".repeat(64),
      corpus_version_id: "corpus-1",
      configuration_version_id: "config-1",
      scorer_version: "scorer-1",
    };
    const retrieval: RetrievalResult = {
      candidate,
      ordered_chunk_ids: ["chunk-1", "chunk-2"],
      scores: [0.91, 0.72],
      score_semantics: "cosine_similarity",
      strategy: "dense",
      stage_status: "succeeded",
      search_ready: true,
    };
    expect(retrieval.ordered_chunk_ids).toHaveLength(retrieval.scores.length);
    expect(OPERATIONAL_ERROR_STATUS).toEqual({ invalid_output: 502, unavailable: 503, deadline: 504 });
  });

  it("models evaluation cases, content-free results, and reference-only settings", () => {
    const settings: RuntimeSettings = {
      project: "dmbok-compass-dev",
      location: "europe-west1",
      provider: "openrouter",
      primary_model: "primary-model",
      secret_version: "openrouter-key-v3",
      lexical_artifact_id: "lexical-bm25-v1",
    };
    const evaluationCase: EvaluationCase = {
      case_id: "case-1",
      question: "What is the retention policy?",
      expected_outcome: "answer",
      relevant_chunk_ids: ["chunk-1"],
      expected_claim_ids: ["claim-1"],
    };
    const result: RunItemResult = {
      case_id: evaluationCase.case_id,
      candidate: {
        artifact_id: "backend@2.1.0",
        installed_lock_sha256: "a".repeat(64),
        corpus_version_id: "corpus-1",
        configuration_version_id: "config-1",
        scorer_version: "scorer-1",
      },
      retrieval_success: true,
      grounded: true,
      citation_correct: true,
      answer_quality: true,
      refusal_correct: false,
      response_time_ms: 125,
      stage_status: "succeeded",
    };
    expect(settings.secret_version).toBe("openrouter-key-v3");
    expect(result.case_id).toBe(evaluationCase.case_id);
    expect(architectureFixture.compatibility.accepted).toEqual(["2.1"]);
    expect(architectureFixture.compatibility.rejected).toEqual(["2.0", "3.0"]);
  });

  it("keeps completed evidence refusal separate from operational failure", () => {
    const outcome: AnswerOutcome = {
      outcome: "evidence_refusal",
      answer_text: "No evidence.",
      claims: [],
      evidence: [],
      trace: { retrieved_passages: [], selected_model: "test", timings_ms: {} },
      quota: {
        user_used: 1,
        user_limit: 100,
        global_used: 1,
        global_limit: 100,
        resets_at: new Date().toISOString(),
      },
      error: null,
    };
    expect(outcome.error).toBeNull();
    expect(architectureFixture.operationalErrorStatus.evidence_refusal).toBe(200);
  });

  it("freezes browser-safe batch review payloads", () => {
    const statuses = fixture.evaluation.batchReviewStatuses;
    expect(statuses).toEqual([
      "pending", "eligible", "exception", "attested", "rejected", "stale", "inapplicable",
    ]);
    const summary: ReviewSummary = {
      status: "eligible",
      eligible_count: 1,
      exception_count: 1,
      sampled_count: 1,
      attested_count: 0,
      decision: null,
    };
    const sample: RepresentativeSampleEntry = {
      item_id: "item-1",
      question: "What is a data domain?",
      rubric: "Defines the concept accurately.",
      support_excerpt: "A bounded source excerpt.",
      page: 7,
      section: "Glossary",
      status: "eligible",
      decision: null,
    };
    const exception: ExceptionRecord = {
      item_id: "item-2",
      question: "What is a data domain?",
      rubric: "Defines the concept accurately.",
      support_excerpt: null,
      page: null,
      section: null,
      status: "exception",
      reason: "Ambiguous support.",
      decision: null,
    };
    expect(summary.status).toBe(statuses[1]);
    expect(sample.support_excerpt.length).toBeLessThanOrEqual(fixture.evaluation.supportExcerptMaxChars);
    expect(Object.keys(exception)).not.toEqual(expect.arrayContaining(fixture.evaluation.browserReviewTechnicalFields));
  });

  it("freezes attestation commands without technical identity controls", () => {
    const request: BatchAttestationRequest = {
      decision: "attested",
      exception_decisions: [{ item_id: "item-2", decision: "rejected", reason: "Wrong support." }],
      reason: null,
    };
    const result: BatchAttestationResult = {
      status: "attested",
      decision: request.decision,
      exception_decisions: request.exception_decisions,
      decision_metadata: {
        decision: "attested",
        actor: "admin-1",
        decided_at: new Date().toISOString(),
        decision_version: "batch-review-v1",
      },
    };
    expect(result.exception_decisions[0]?.item_id).toBe("item-2");
    expect(Object.keys(request)).not.toEqual(expect.arrayContaining(fixture.evaluation.browserReviewTechnicalFields));
  });
});
