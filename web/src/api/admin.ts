import { ADMIN_ROUTES, type AggregateMetric, type ApprovalState, type CorpusStagingAttemptStatus, type CorpusStagingStructureCorrectionEntry, type QuotaPolicy, type StructureTemplate, type UserProfile } from "./contracts";

export class AdminApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

async function adminRequest<T>(path: string, token: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(init?.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string | { message?: string } } | null;
    const detail = body?.detail;
    throw new AdminApiError(response.status, typeof detail === "string" ? detail : detail?.message ?? "The administrator service could not complete the request.");
  }
  return response.json() as Promise<T>;
}

export function getCorpusStagingTemplate(token: string): Promise<StructureTemplate> {
  return adminRequest<StructureTemplate>(ADMIN_ROUTES.corpusStagingTemplate, token);
}

export async function submitPdfOnlyCorpusStagingAttempt(token: string, sourcePdf: File): Promise<CorpusStagingAttemptStatus> {
  const body = new FormData();
  body.append("source_pdf", sourcePdf);
  return adminRequest<CorpusStagingAttemptStatus>(`${ADMIN_ROUTES.corpusStagingAttempts}/pdf`, token, { method: "POST", body });
}

export function startCorpusStagingParse(token: string, attemptId: string): Promise<CorpusStagingAttemptStatus> {
  return adminRequest<CorpusStagingAttemptStatus>(`${ADMIN_ROUTES.corpusStagingAttempts}/${encodeURIComponent(attemptId)}/parse`, token, { method: "POST" });
}

export function proposeCorpusStagingStructure(token: string, attemptId: string): Promise<CorpusStagingAttemptStatus> {
  return adminRequest<CorpusStagingAttemptStatus>(`${ADMIN_ROUTES.corpusStagingAttempts}/${encodeURIComponent(attemptId)}/structure/propose`, token, { method: "POST" });
}

export function correctCorpusStagingStructure(token: string, attemptId: string, expectedRevision: number, entries: CorpusStagingStructureCorrectionEntry[]): Promise<CorpusStagingAttemptStatus> {
  return adminRequest<CorpusStagingAttemptStatus>(`${ADMIN_ROUTES.corpusStagingAttempts}/${encodeURIComponent(attemptId)}/structure`, token, {
    method: "PUT",
    body: JSON.stringify({ expected_revision: expectedRevision, entries }),
  });
}

export async function submitCorpusStagingAttempt(token: string, sourcePdf: File, structureJson: File): Promise<CorpusStagingAttemptStatus> {
  const body = new FormData();
  body.append("source_pdf", sourcePdf);
  body.append("structure_json", structureJson);
  return adminRequest<CorpusStagingAttemptStatus>(ADMIN_ROUTES.corpusStagingAttempts, token, { method: "POST", body });
}

export function validateCorpusStagingAttempt(token: string, attemptId: string): Promise<CorpusStagingAttemptStatus> {
  return adminRequest<CorpusStagingAttemptStatus>(`${ADMIN_ROUTES.corpusStagingAttempts}/${encodeURIComponent(attemptId)}/validate`, token, { method: "POST" });
}

export function approveCorpusStagingAttempt(token: string, attemptId: string, expectedRevision: number): Promise<CorpusStagingAttemptStatus> {
  const query = new URLSearchParams({ expected_revision: String(expectedRevision) });
  return adminRequest<CorpusStagingAttemptStatus>(`${ADMIN_ROUTES.corpusStagingAttempts}/${encodeURIComponent(attemptId)}/approve?${query}`, token, { method: "POST" });
}

export function getCorpusStagingStatus(token: string, attemptId: string): Promise<CorpusStagingAttemptStatus> {
  return adminRequest<CorpusStagingAttemptStatus>(`${ADMIN_ROUTES.corpusStagingAttempts}/${encodeURIComponent(attemptId)}`, token);
}

export function listUsers(token: string, approvalState?: ApprovalState): Promise<UserProfile[]> {
  const query = approvalState ? `?approval_state=${encodeURIComponent(approvalState)}` : "";
  return adminRequest<UserProfile[]>(`${ADMIN_ROUTES.users}${query}`, token);
}

export function updateUser(token: string, userId: string, approvalState: ApprovalState): Promise<UserProfile> {
  return adminRequest<UserProfile>(`${ADMIN_ROUTES.users}/${encodeURIComponent(userId)}`, token, {
    method: "PATCH",
    body: JSON.stringify({ approval_state: approvalState }),
  });
}

export function getConfiguration(token: string): Promise<QuotaPolicy> {
  return adminRequest<QuotaPolicy>(ADMIN_ROUTES.configuration, token);
}

export function updateConfiguration(token: string, policy: QuotaPolicy): Promise<QuotaPolicy> {
  return adminRequest<QuotaPolicy>(ADMIN_ROUTES.configuration, token, {
    method: "PATCH",
    body: JSON.stringify(policy),
  });
}

export function listAggregateMetrics(token: string): Promise<AggregateMetric[]> {
  return adminRequest<AggregateMetric[]>(ADMIN_ROUTES.aggregateMetrics, token);
}
