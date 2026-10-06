import { API_ROUTES, type UserProfile } from "./contracts";

export class IdentityApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

async function identityRequest<T>(path: string, token: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null) as { detail?: string } | null;
    throw new IdentityApiError(response.status, detail?.detail ?? "The identity service could not complete the request.");
  }
  return response.json() as Promise<T>;
}

export function registerProfile(token: string, username: string): Promise<UserProfile> {
  return identityRequest(API_ROUTES.registration, token, {
    method: "POST",
    body: JSON.stringify({ username }),
  });
}

export function getCurrentProfile(token: string): Promise<UserProfile> {
  return identityRequest(API_ROUTES.currentUser, token);
}
