export const API_ROUTES = {
  registration: "/api/registration",
  registrationStatus: "/api/registration/status",
  currentUser: "/api/me",
  answer: "/api/questions",
  quota: "/api/quota",
  evaluationRuns: "/api/evaluations",
  releaseDecisions: "/api/releases",
} as const;

export const ADMIN_ROUTES = {
  users: "/api/admin/users",
  configuration: "/api/admin/configuration",
  aggregateMetrics: "/api/admin/metrics",
} as const;

export type Outcome = "answer" | "qualified" | "refusal";
export type ApprovalState = "pending" | "approved" | "rejected" | "deactivated";

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
  decision: "pending" | "approved" | "rejected";
  rationale: string;
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
] as const;

export type TelemetryField = (typeof TELEMETRY_FIELDS)[number];
