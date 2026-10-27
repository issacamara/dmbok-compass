import {
  API_ROUTES,
  type ActiveEvaluationGeneration,
  type CandidateBindingInput,
  type CandidateBindingResult,
  type EvaluationImportResponse,
  type EvaluationRun,
  type ItemReview,
  type ReleaseDecision,
  type RunItemAssessment,
  type ReviewStatus,
  type SupportStatusResponse,
} from "../../api/contracts";

export class EvaluationApiError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}

async function request<T>(path: string, token: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...init, headers: { Authorization: `Bearer ${token}`, ...(init?.body ? { "Content-Type": "application/json" } : {}), ...init?.headers } });
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: { message?: string } | string } | null;
    const detail = typeof body?.detail === "string" ? body.detail : body?.detail?.message;
    throw new EvaluationApiError(response.status, detail ?? "The evaluation service could not complete the request.");
  }
  return response.json() as Promise<T>;
}

export type EvaluationLaunch = {
  item_ids?: string[];
};

export function launchEvaluation(token: string, payload: EvaluationLaunch): Promise<EvaluationRun> {
  return request<EvaluationRun>(API_ROUTES.evaluationRuns, token, { method: "POST", body: JSON.stringify(payload) });
}

export function getEvaluationRun(token: string, runId: string): Promise<EvaluationRun> {
  return request<EvaluationRun>(`${API_ROUTES.evaluationRuns}/${encodeURIComponent(runId)}`, token);
}

export function importEvaluationDataset(token: string, source: unknown): Promise<EvaluationImportResponse> {
  return request<EvaluationImportResponse>(API_ROUTES.evaluationImport, token, {
    method: "POST",
    body: JSON.stringify(source),
  });
}

export function getActiveEvaluationGeneration(token: string): Promise<ActiveEvaluationGeneration> {
  return request<ActiveEvaluationGeneration>(API_ROUTES.activeEvaluationGeneration, token);
}

export function reviewEvaluationItem(
  token: string,
  generationId: string,
  itemId: string,
  reviewStatus: Exclude<ReviewStatus, "candidate">,
): Promise<ItemReview> {
  return request<ItemReview>(
    `${API_ROUTES.evaluationDatasets}/${encodeURIComponent(generationId)}/items/${encodeURIComponent(itemId)}/review`,
    token,
    { method: "PATCH", body: JSON.stringify({ review_status: reviewStatus }) },
  );
}

export function bindEvaluationItem(
  token: string,
  generationId: string,
  itemId: string,
  payload: CandidateBindingInput,
): Promise<CandidateBindingResult> {
  return request<CandidateBindingResult>(
    `${API_ROUTES.evaluationDatasets}/${encodeURIComponent(generationId)}/items/${encodeURIComponent(itemId)}/binding`,
    token,
    { method: "PUT", body: JSON.stringify(payload) },
  );
}

export function getSupportStatus(token: string, runId: string, itemId: string): Promise<SupportStatusResponse> {
  return request<SupportStatusResponse>(
    `${API_ROUTES.evaluationSupport}/${encodeURIComponent(runId)}/items/${encodeURIComponent(itemId)}/support`,
    token,
  );
}

export type SupportReviewInput = { decision: "approve" | "reject"; selected_chunk_id?: string };

export function reviewSupport(
  token: string,
  runId: string,
  itemId: string,
  payload: SupportReviewInput,
): Promise<SupportStatusResponse> {
  return request<SupportStatusResponse>(
    `${API_ROUTES.evaluationSupport}/${encodeURIComponent(runId)}/items/${encodeURIComponent(itemId)}/support`,
    token,
    { method: "PUT", body: JSON.stringify(payload) },
  );
}

export type AssessmentInput = Omit<RunItemAssessment, "assessment_status">;

export function assessEvaluationItem(
  token: string,
  runId: string,
  itemId: string,
  payload: AssessmentInput,
): Promise<RunItemAssessment> {
  return request<RunItemAssessment>(
    `${API_ROUTES.evaluationAssessments}/${encodeURIComponent(runId)}/items/${encodeURIComponent(itemId)}/assessment`,
    token,
    { method: "PUT", body: JSON.stringify(payload) },
  );
}

export type ReleaseDecisionInput = Omit<ReleaseDecision, "decision"> & { decision: "approved" | "rejected" };

export function recordReleaseDecision(token: string, payload: ReleaseDecisionInput): Promise<ReleaseDecision> {
  return request<ReleaseDecision>(API_ROUTES.releaseDecisions, token, { method: "POST", body: JSON.stringify(payload) });
}

export function publishRelease(token: string, releaseId: string): Promise<unknown> {
  return request<unknown>(`${API_ROUTES.releaseDecisions}/${encodeURIComponent(releaseId)}/publish`, token, { method: "POST" });
}
