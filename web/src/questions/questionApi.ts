import { API_ROUTES, type AnswerResponse, type QuotaStatus } from "../api/contracts";

export class QuestionApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly code = "question_request_failed",
    readonly retryable = false,
  ) {
    super(message);
  }
}

async function questionRequest<T>(path: string, token: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null) as {
      detail?: string | { code?: string; message?: string; retryable?: boolean };
    } | null;
    const detail = body?.detail;
    const parsed = typeof detail === "string" ? { message: detail } : detail;
    throw new QuestionApiError(
      response.status,
      parsed?.message ?? "The question service could not complete the request.",
      parsed?.code,
      parsed?.retryable,
    );
  }
  return response.json() as Promise<T>;
}

export function getQuota(token: string): Promise<QuotaStatus> {
  return questionRequest<QuotaStatus>(API_ROUTES.quota, token);
}

export function askQuestion(token: string, question: string): Promise<AnswerResponse> {
  return questionRequest<AnswerResponse>(API_ROUTES.answer, token, {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}
