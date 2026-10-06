import { ADMIN_ROUTES, type ApprovalState, type QuotaPolicy, type UserProfile } from "./contracts";

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
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null;
    throw new AdminApiError(response.status, body?.detail ?? "The administrator service could not complete the request.");
  }
  return response.json() as Promise<T>;
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
