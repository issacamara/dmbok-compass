import { API_ROUTES, type EvaluationRun, type ReleaseDecision } from "../../api/contracts";

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
  corpus_version_id: string; configuration_version_id: string; model_version_id: string; item_ids?: string[];
};

export function launchEvaluation(token: string, payload: EvaluationLaunch): Promise<EvaluationRun> {
  return request<EvaluationRun>(API_ROUTES.evaluationRuns, token, { method: "POST", body: JSON.stringify(payload) });
}

export function getEvaluationRun(token: string, runId: string): Promise<EvaluationRun> {
  return request<EvaluationRun>(`${API_ROUTES.evaluationRuns}/${encodeURIComponent(runId)}`, token);
}

export type ReleaseDecisionInput = Omit<ReleaseDecision, "decision"> & { decision: "approved" | "rejected" };

export function recordReleaseDecision(token: string, payload: ReleaseDecisionInput): Promise<ReleaseDecision> {
  return request<ReleaseDecision>(API_ROUTES.releaseDecisions, token, { method: "POST", body: JSON.stringify(payload) });
}
