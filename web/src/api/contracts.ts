export const API_ROUTES = {
  registration: "/api/registration",
  registrationStatus: "/api/registration/status",
  currentUser: "/api/me",
  answer: "/api/questions",
  quota: "/api/quota",
  evaluationRuns: "/api/evaluations",
  evaluationImport: "/api/admin/evaluation-datasets/import",
  activeEvaluationGeneration: "/api/admin/evaluation-datasets/active",
  releaseDecisions: "/api/releases",
  evaluationDatasets: "/api/admin/evaluation-datasets",
  evaluationAssessments: "/api/admin/evaluations",
  evaluationSupport: "/api/admin/evaluations",
} as const;

export const ADMIN_ROUTES = {
  users: "/api/admin/users",
  configuration: "/api/admin/configuration",
  aggregateMetrics: "/api/admin/metrics",
  releaseDecisions: "/api/releases",
} as const;

export type Outcome = "answer" | "qualified" | "refusal";
export type ApprovalState = "pending" | "approved" | "rejected" | "deactivated";
export type QuestionCategory = "definitions" | "explanations" | "comparisons" | "study" | "scenarios";
export type ReviewStatus = "candidate" | "approved" | "rejected";
export type RunEligibility = "exploratory" | "release_evidence";
export type SupportMatchStatus = "eligible" | "missing" | "ambiguous" | "invalid" | "inapplicable" | "review_approved" | "review_rejected";
export type BatchReviewStatus = "pending" | "eligible" | "exception" | "attested" | "rejected" | "stale" | "inapplicable";
export type ExceptionDecisionKind = "attested" | "rejected" | "inapplicable";
export type BatchAttestationDecision = "attested" | "rejected";
export type ImportStatus = "accepted" | "rejected";
export type AnswerOutcomeKind = "answer" | "qualified" | "evidence_refusal" | "operational_error";
export type StageStatus = "pending" | "running" | "succeeded" | "failed" | "unavailable";
export type RetrievalStrategy = "dense" | "hybrid" | "reranked";
export type RetrievalScoreSemantics = "cosine_similarity" | "distance" | "calibrated";

export const OPERATIONAL_ERROR_STATUS = {
  invalid_output: 502,
  unavailable: 503,
  deadline: 504,
} as const;

export const MAX_EVALUATION_IMPORT_BYTES = 1_048_576;
export const MAX_EVALUATION_IMPORT_ITEMS = 500;
export const MIN_APPROVED_GOLD_ITEMS = 30;
export const MAX_APPROVED_GOLD_ITEMS = 50;

export interface CandidateIdentity {
  artifact_id: string;
  installed_lock_sha256: string;
  corpus_version_id: string;
  configuration_version_id: string;
  scorer_version: string;
}

export interface RuntimeSettings {
  project: string;
  location: string;
  provider: "openrouter" | "vertex";
  primary_model: string;
  fallback_model?: string | null;
  secret_version?: string | null;
  lexical_artifact_id?: string | null;
}

export interface RetrievalResult {
  candidate: CandidateIdentity;
  ordered_chunk_ids: string[];
  scores: number[];
  score_semantics: RetrievalScoreSemantics;
  strategy: RetrievalStrategy;
  stage_status: StageStatus;
  search_ready: boolean;
}

export interface EvaluationCase {
  case_id: string;
  question: string;
  expected_outcome: "answer" | "qualified" | "evidence_refusal";
  relevant_chunk_ids: string[];
  expected_claim_ids: string[];
}

export interface RunItemResult {
  case_id: string;
  candidate: CandidateIdentity;
  retrieval_success: boolean;
  grounded: boolean;
  citation_correct: boolean;
  answer_quality: boolean;
  refusal_correct: boolean;
  response_time_ms: number;
  stage_status: StageStatus;
}

export interface CitationEvidence {
  citation_id: string;
  claim_id: string;
  chunk_id: string;
  page: number;
  section: string;
  excerpt: string;
}

export interface AnswerOutcome {
  outcome: AnswerOutcomeKind;
  answer_text?: string | null;
  claims: string[];
  evidence: CitationEvidence[];
  trace: RetrievalTrace;
  quota: QuotaStatus;
  error?: ApiError | null;
}

export interface QuotaPolicy {
  per_user_daily_limit: number;
  global_daily_limit: number;
}

export interface UserProfile {
  user_id: string;
  email: string;
  username: string;
  approval_state: ApprovalState;
  role: "user" | "admin";
  email_verified: boolean;
}

export interface ApiError {
  code: string;
  message: string;
  retryable: boolean;
  status: number;
}

export interface QuotaStatus {
  user_used: number;
  user_limit: number;
  global_used: number;
  global_limit: number;
  resets_at: string;
}

export interface Citation {
  citation_id: string;
  page: number;
  section: string;
}

export interface RetrievedPassage {
  chunk_id: string;
  page: number;
  section: string;
  excerpt: string;
  relevance_score?: number;
}

export interface RetrievalTrace {
  retrieved_passages: RetrievedPassage[];
  selected_model: string;
  timings_ms: Record<string, number>;
  model_attempts?: { model: string; outcome: string; status_code?: number | null }[];
  model_output?: string;
}

export interface AnswerResponse {
  outcome: Outcome;
  answer_text?: string;
  synthesis: boolean;
  citations: Citation[];
  trace: RetrievalTrace;
  quota: QuotaStatus;
  error?: ApiError;
}

export interface DocumentChunk {
  chunk_id: string;
  corpus_version_id: string;
  page: number;
  section: string;
  chunk_ordinal: number;
  content_hash: string;
  text: string;
  embedding: number[];
}

export interface AggregateMetric {
  metric_name: string;
  numerator: number;
  denominator: number;
  percentage: number;
}

export interface EvaluationMetric extends AggregateMetric {
  threshold: number;
  passed: boolean;
}

export interface EvaluationRun {
  run_id: string;
  dataset_version_id: string;
  corpus_version_id: string;
  status: "queued" | "running" | "completed" | "failed";
  metrics: EvaluationMetric[];
  item_results: EvaluationItemResult[];
  selected_item_ids: string[];
  configuration_version_id: string;
  model_version_id: string;
  error: string | null;
  report_eligibility?: EvaluationReportEligibility | null;
  candidate_snapshot?: { [key: string]: unknown } | null;
}

export interface VerifiedChunkChoice {
  chunk_id: string;
  corpus_version_id: string;
  content_hash: string;
  page: number;
  section: string;
  source_provenance_sha256: string;
}

export interface RunBoundSupportEvidence {
  schema_version: "run-bound-support-v1";
  run_id: string;
  item_id: string;
  candidate_snapshot_sha256: string;
  corpus_version_id: string;
  source_provenance_sha256: string;
  match_status: SupportMatchStatus;
  candidate_chunk_ids: string[];
  verified_chunks: VerifiedChunkChoice[];
  selected_chunk: VerifiedChunkChoice | null;
  reason: string | null;
  expected_refusal: boolean;
  reviewer_actor?: string | null;
  reviewed_at?: string | null;
  binding_version: string;
}

export interface SupportStatusResponse {
  evidence: RunBoundSupportEvidence;
  candidate_chunks: VerifiedChunkChoice[];
}

export interface ReviewDecisionMetadata {
  decision: BatchReviewStatus;
  actor: string;
  decided_at: string;
  decision_version: string;
}

export interface ReviewSummary {
  status: BatchReviewStatus;
  eligible_count: number;
  exception_count: number;
  sampled_count: number;
  attested_count: number;
  decision?: ReviewDecisionMetadata | null;
}

export interface RepresentativeSampleEntry {
  item_id: string;
  question: string;
  rubric: string;
  support_excerpt: string;
  page: number;
  section: string;
  status: BatchReviewStatus;
  decision?: ReviewDecisionMetadata | null;
}

export interface ExceptionRecord {
  item_id: string;
  question: string;
  rubric: string;
  support_excerpt?: string | null;
  page?: number | null;
  section?: string | null;
  status: "exception" | "stale" | "inapplicable";
  reason: string;
  decision?: ReviewDecisionMetadata | null;
}

export interface ExplicitExceptionDecision {
  item_id: string;
  decision: ExceptionDecisionKind;
  reason?: string | null;
}

export interface BatchAttestationRequest {
  decision: BatchAttestationDecision;
  exception_decisions: ExplicitExceptionDecision[];
  reason?: string | null;
}

export interface BatchAttestationResult {
  status: BatchReviewStatus;
  decision: BatchAttestationDecision;
  exception_decisions: ExplicitExceptionDecision[];
  decision_metadata: ReviewDecisionMetadata;
}

export interface DurableReviewSummary {
  generation_id: string;
  run_id: string;
  candidate_snapshot_sha256: string;
  corpus_version_id: string;
  status: BatchReviewStatus;
  eligible_count: number;
  exception_count: number;
  sampled_item_ids: string[];
  actor?: string | null;
  decided_at?: string | null;
  decision_version: string;
}

export interface DurableRepresentativeSampleEntry {
  generation_id: string;
  run_id: string;
  candidate_snapshot_sha256: string;
  corpus_version_id: string;
  sample_identity: string;
  item_id: string;
  candidate_chunk_ids: string[];
  content_hashes: string[];
  provenance_digests: string[];
  status: BatchReviewStatus;
  decision_version: string;
}

export interface DurableExceptionRecord {
  generation_id: string;
  run_id: string;
  candidate_snapshot_sha256: string;
  corpus_version_id: string;
  sample_identity: string;
  item_id: string;
  candidate_chunk_ids: string[];
  content_hashes: string[];
  provenance_digests: string[];
  status: "exception" | "stale" | "inapplicable";
  reason: string;
  decision_version: string;
}

export interface DurableExceptionDecision {
  generation_id: string;
  run_id: string;
  candidate_snapshot_sha256: string;
  corpus_version_id: string;
  sample_identity: string;
  item_id: string;
  decision: ExceptionDecisionKind;
  actor: string;
  decided_at: string;
  decision_version: string;
  reason?: string | null;
}

export interface DurableBatchAttestation {
  generation_id: string;
  run_id: string;
  candidate_snapshot_sha256: string;
  corpus_version_id: string;
  sampled_item_ids: string[];
  decision: BatchAttestationDecision;
  actor: string;
  decided_at: string;
  decision_version: string;
  exception_decisions: DurableExceptionDecision[];
}

export type AssessmentStatus = "pending" | "reviewed" | "inapplicable";

export interface EvaluationItemOutput {
  response_sha256: string;
  outcome: AnswerOutcomeKind;
  answer_text?: string | null;
  claims: string[];
  evidence: CitationEvidence[];
}

export interface RunItemAssessment {
  assessment_status: AssessmentStatus;
  response_sha256: string;
  scorer_version: string;
  claim_support: boolean | null;
  claim_citation_support: boolean | null;
  rubric_passed: boolean | null;
  expected_refusal: boolean | null;
}

export interface ImportValidationError {
  index: number;
  code: "invalid_record" | "missing_question" | "missing_expected_answer_or_rubric" | "missing_supporting_passage" | "invalid_category";
  message: string;
}

export interface EvaluationImportResponse {
  status: ImportStatus;
  generation_id: string | null;
  submitted_count: number;
  imported_count: number;
  skipped_count: number;
  validation_errors: ImportValidationError[];
}

export interface ActiveEvaluationGeneration {
  generation_id: string;
  item_count: number;
  approved_gold_count: number;
  status: "active";
  item_reviews: ItemReview[];
}

export interface ItemReview {
  generation_id: string;
  item_id: string;
  review_status: ReviewStatus;
}

export interface EvaluationRunSelection {
  item_ids?: string[] | null;
}

export interface EvaluationReportEligibility {
  generation_id: string;
  eligibility: RunEligibility;
  approved_gold_count: number;
  is_current_generation: boolean;
  is_superseded: boolean;
}

export interface EvaluationItemResult {
  item_id: string;
  retrieval_success: boolean;
  grounded: boolean;
  citation_correct: boolean;
  answer_quality: boolean;
  refusal_correct: boolean;
  response_time_ms: number;
  assessment_status?: AssessmentStatus;
  output?: EvaluationItemOutput | null;
  assessment?: RunItemAssessment | null;
}

export interface ReleaseDecision {
  release_id: string;
  evaluation_run_id: string;
  dataset_version_id: string;
  corpus_version_id: string;
  configuration_version_id: string;
  provider_version_id: string;
  model_version_id: string;
  scorer_version_id: string;
  gate_report_ids: string[];
  decision: "pending" | "approved" | "rejected";
  rationale: string;
  exception_approved: boolean;
  exception_rationale: string | null;
}

export const TELEMETRY_FIELDS = [
  "request_count",
  "outcome",
  "duration_ms",
  "provider_model",
  "token_estimate",
  "quota_utilization",
  "corpus_version",
  "configuration_version",
  "error_class",
  "model_attempts",
] as const;

export type TelemetryField = (typeof TELEMETRY_FIELDS)[number];
