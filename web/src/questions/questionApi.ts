import { API_ROUTES, type AnswerResponse, type AnswerOutcome, type QuotaStatus } from "../api/contracts";

export type QuestionAnswer = Omit<AnswerResponse, "outcome" | "citations" | "synthesis"> & {
  outcome: AnswerResponse["outcome"] | "evidence_refusal" | "operational_error";
  citations: AnswerResponse["citations"];
  synthesis: boolean;
  claims?: string[];
  evidence?: AnswerOutcome["evidence"];
};

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

function normalizeAnswer(response: QuestionAnswer): QuestionAnswer {
  return {
    ...response,
    citations: response.citations ?? (response.evidence ?? []).map(({ citation_id, page, section }) => ({ citation_id, page, section })),
    synthesis: response.synthesis ?? false,
  };
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

export function askQuestion(token: string, question: string): Promise<QuestionAnswer> {
  return questionRequest<QuestionAnswer>(API_ROUTES.answer, token, {
    method: "POST",
    body: JSON.stringify({ question }),
  }).then(normalizeAnswer);
}
