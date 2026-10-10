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
export type ImportStatus = "accepted" | "rejected";

export const MAX_EVALUATION_IMPORT_BYTES = 1_048_576;
export const MAX_EVALUATION_IMPORT_ITEMS = 500;
export const MIN_APPROVED_GOLD_ITEMS = 30;
export const MAX_APPROVED_GOLD_ITEMS = 50;

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
  excerpt: string;
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
