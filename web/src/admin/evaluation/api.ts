import { API_ROUTES, type EvaluationImportResponse, type EvaluationRun } from "../../api/contracts";

export class EvaluationApiError extends Error { constructor(readonly status: number, message: string) { super(message); } }

async function request<T>(path: string, token: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...init, headers: { Authorization: `Bearer ${token}`, ...(init?.body ? { "Content-Type": "application/json" } : {}), ...init?.headers } });
  if (!response.ok) { const body = await response.json().catch(() => null) as { detail?: { message?: string } | string } | null; const detail = typeof body?.detail === "string" ? body.detail : body?.detail?.message; throw new EvaluationApiError(response.status, detail ?? "The evaluation service could not complete the request."); }
  return response.json() as Promise<T>;
}

export type EvaluationUploadRunResponse = EvaluationImportResponse & { run_id: string | null };
export function uploadAndRunEvaluation(token: string, source: unknown): Promise<EvaluationUploadRunResponse> { return request<EvaluationUploadRunResponse>(API_ROUTES.evaluationImport, token, { method: "POST", body: JSON.stringify(source) }); }
export function getEvaluationRun(token: string, runId: string): Promise<EvaluationRun> { return request<EvaluationRun>(`${API_ROUTES.evaluationRuns}/${encodeURIComponent(runId)}`, token); }
