import { useEffect, useRef, useState, type FormEvent } from "react";
import type { AnswerResponse, QuotaStatus } from "../api/contracts";
import { askQuestion, getQuota, QuestionApiError, type QuestionAnswer } from "./questionApi";

type QuestionPanelProps = { token: string; onSignOut: () => void };

function resetLabel(quota: QuotaStatus): string {
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(quota.resets_at));
}

function outcomeLabel(answer: QuestionAnswer): string {
  if (answer.outcome === "operational_error" || answer.error) return "Answer unavailable";
  if (answer.outcome === "evidence_refusal") return "Evidence-based refusal";
  if (answer.outcome === "answer") return "Strong evidence";
  if (answer.outcome === "qualified") return "Qualified answer · partial evidence";
  return "Evidence-based refusal";
}

export function QuestionPanel({ token, onSignOut }: QuestionPanelProps) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<QuestionAnswer | null>(null);
  const [quota, setQuota] = useState<QuotaStatus | null>(null);
  const [loadingQuota, setLoadingQuota] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [retryableError, setRetryableError] = useState<QuestionApiError | null>(null);
  const questionInput = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    let active = true;
    void getQuota(token).then((currentQuota) => {
      if (active) setQuota(currentQuota);
    }).catch(() => {
      if (active) setError("Could not load your request quota. Try again shortly.");
    }).finally(() => {
      if (active) setLoadingQuota(false);
    });
    return () => { active = false; };
  }, [token]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) {
      setError("Enter a question before sending it.");
      questionInput.current?.focus();
      return;
    }
    setSubmitting(true);
    setError("");
    setRetryableError(null);
    try {
      const response = await askQuestion(token, trimmedQuestion);
      setAnswer(response);
      setQuota(response.quota);
    } catch (cause) {
      const apiError = cause instanceof QuestionApiError ? cause : null;
      setError(apiError?.message ?? "The question could not be completed. Try again shortly.");
      setRetryableError(apiError?.retryable ? apiError : null);
      if (cause instanceof QuestionApiError && cause.status === 429) {
        void getQuota(token).then(setQuota).catch(() => undefined);
      }
    } finally {
      setSubmitting(false);
    }
  }

  function clearSession() {
    setQuestion("");
    setAnswer(null);
    setError("");
    setRetryableError(null);
    questionInput.current?.focus();
  }

  return (
    <section className="question-panel" aria-labelledby="question-title">
      <div className="question-header">
        <div>
          <p className="eyebrow">ASK COMPASS</p>
          <h2 id="question-title">What are you trying to understand?</h2>
          <p className="helper-text">DAMA-DMBOK, Second Edition · answers use only the approved corpus.</p>
        </div>
        <div className="question-actions">
          {quota && <span className="quota-chip">{Math.max(0, quota.user_limit - quota.user_used)} requests left today</span>}
          <button className="secondary" type="button" onClick={onSignOut}>Sign out</button>
        </div>
      </div>

      {error && <div className="error" role="alert" aria-live="assertive">
        <p>{error}</p>
        {retryableError && <button className="link" type="button" onClick={() => void submit({ preventDefault: () => undefined } as FormEvent<HTMLFormElement>)}>
          Retry this question
        </button>}
      </div>}
      <div className="question-layout" aria-busy={submitting}>
        <section className="question-card" aria-label="Question and answer">
          <div className="question-card-header">
            <strong>New question</strong>
            <button className="link" type="button" onClick={clearSession}>Clear</button>
          </div>
          {answer ? <article className={`answer answer-${answer.outcome}`} aria-live="polite">
            <div className="answer-labels">
              <span className="evidence-chip">{outcomeLabel(answer)}</span>
              {answer.synthesis && <span className="synthesis-chip">Synthesis</span>}
            </div>
            <p className="answer-text">{answer.answer_text ?? answer.error?.message ?? "The corpus could not support an answer."}</p>
            {((answer.evidence?.length ?? 0) > 0 || answer.citations.length > 0) && <div className="citation-list" aria-label="Citations">
              {(answer.evidence ?? answer.citations).map((citation, index) => <div className="citation" key={citation.citation_id}>
                <strong id={`citation-${citation.citation_id}`}>[{index + 1}] {citation.section}</strong>
                <span> · p. {citation.page}</span>
                {"excerpt" in citation && typeof citation.excerpt === "string" && <blockquote aria-labelledby={`citation-${citation.citation_id}`}>{citation.excerpt}</blockquote>}
              </div>)}
            </div>}
          </article> : <p className="empty-answer">Ask a definition, relationship, or scenario question to see grounded evidence here.</p>}
          <form className="question-composer" onSubmit={submit} aria-busy={submitting}>
            <label htmlFor="question-input">Ask a DMBOK question</label>
            <div className="composer-row">
              <textarea
                ref={questionInput}
                id="question-input"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask about a definition, relationship, or scenario…"
                maxLength={2000}
                disabled={submitting}
                aria-describedby="question-help"
              />
              <button className="primary" type="submit" disabled={submitting || loadingQuota}>
                {submitting ? "Finding evidence…" : "Send"}
              </button>
            </div>
            <span id="question-help" className="visually-hidden">Questions are answered from the approved DMBOK corpus.</span>
          </form>
        </section>

        <aside className="trace-card" aria-labelledby="trace-title">
          <h3 id="trace-title">Request trace</h3>
          <p className="helper-text">Available only while this answer is open. Nothing here is saved.</p>
          {answer ? <>
            <div className="trace-row"><span>Evidence</span><strong>{answer.trace.retrieved_passages.length} passages</strong></div>
            <div className="trace-row"><span>Model</span><strong>{answer.trace.selected_model}</strong></div>
            {answer.trace.model_attempts?.map((attempt, index) => <div className="trace-row" key={`${attempt.model}-${index}`}>
              <span>Model attempt {index + 1}</span><strong>{attempt.model} · {attempt.outcome}{attempt.status_code ? ` (${attempt.status_code})` : ""}</strong>
            </div>)}
            {Object.entries(answer.trace.timings_ms).map(([name, value]) => <div className="trace-row" key={name}><span>{name.replaceAll("_", " ")}</span><strong>{Math.round(value)} ms</strong></div>)}
            <details className="passage-details"><summary>View passage details</summary>
              {answer.trace.retrieved_passages.map((passage) => <p key={passage.chunk_id}>p. {passage.page} · {passage.section}</p>)}
            </details>
          </> : <p className="helper-text">The evidence and model trace will appear after your question is answered.</p>}
          {quota && <p className="quota-reset">Quota resets {resetLabel(quota)}.</p>}
        </aside>
      </div>
    </section>
  );
}
