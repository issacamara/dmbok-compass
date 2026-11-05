import { useEffect, useState, type ChangeEvent, type FormEvent } from "react";
import type { BatchAttestationDecision, BatchReviewStatus, EvaluationItemResult, EvaluationMetric, EvaluationReviewProjection, EvaluationRun, ExplicitExceptionDecision, ImportValidationError } from "../../api/contracts";
import { attestBatchReview, EvaluationApiError, getEvaluationRun, getReviewSummary, uploadAndRunEvaluation, type EvaluationUploadRunResponse } from "./api";

type EvaluationPanelProps = { token: string };
type EvaluationStatus = Pick<EvaluationRun, "run_id" | "status" | "metrics" | "item_results" | "error">;

const metricLabels: Record<string, string> = { retrieval_success: "Top-5 retrieval success", grounded_claims: "Grounded claims", citation_correctness: "Citation correctness", answer_quality: "Answer quality", refusal_correctness: "Refusal correctness", response_time: "Response time" };

export function EvaluationPanel({ token }: EvaluationPanelProps) {
  const [source, setSource] = useState<unknown | null>(null);
  const [runId, setRunId] = useState("");
  const [run, setRun] = useState<EvaluationStatus | null>(null);
  const [summary, setSummary] = useState<EvaluationUploadRunResponse | null>(null);
  const [validationErrors, setValidationErrors] = useState<ImportValidationError[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [question, setQuestion] = useState("");
  const [category, setCategory] = useState("definitions");
  const [expected, setExpected] = useState("");
  const [passage, setPassage] = useState("");
  const [submittedQuestion, setSubmittedQuestion] = useState("");
  const [singleQuestionOpen, setSingleQuestionOpen] = useState(false);
  const [review, setReview] = useState<EvaluationReviewProjection | null>(null);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [reviewBusy, setReviewBusy] = useState(false);
  const [reviewError, setReviewError] = useState("");
  const [reviewMessage, setReviewMessage] = useState("");
  const [reviewRefresh, setReviewRefresh] = useState(0);
  const [reviewDecision, setReviewDecision] = useState<BatchAttestationDecision | "">("");
  const [reviewReason, setReviewReason] = useState("");
  const [exceptionDecisions, setExceptionDecisions] = useState<Record<string, { decision: "rejected" | "inapplicable" | ""; reason: string }>>({});

  useEffect(() => {
    if (!runId || !run || !["queued", "running"].includes(run.status)) return;
    const timer = window.setTimeout(() => { void getEvaluationRun(token, runId).then(setRun).catch(showError); }, 500);
    return () => window.clearTimeout(timer);
  }, [run, runId, token]);

  useEffect(() => {
    if (!runId || run?.status !== "completed") { setReview(null); return; }
    let current = true;
    setReviewLoading(true); setReviewError(""); setReview(null);
    void getReviewSummary(token, runId).then((result) => { if (current) setReview(result); }).catch((cause: unknown) => {
      if (!current) return;
      setReviewError(cause instanceof EvaluationApiError && [401, 403].includes(cause.status)
        ? "You do not have administrator access to review this run."
        : cause instanceof EvaluationApiError && cause.status === 409
          ? "This review is stale. Refresh it to load the current run state."
          : cause instanceof Error ? cause.message : "The review summary could not be loaded.");
    }).finally(() => { if (current) setReviewLoading(false); });
    return () => { current = false; };
  }, [run?.status, runId, token, reviewRefresh]);

  function showError(cause: unknown) { setError(cause instanceof EvaluationApiError && cause.status === 403 ? "You do not have administrator access." : cause instanceof Error ? cause.message : "The evaluation service could not complete the request."); }

  async function selectFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]; event.target.value = ""; if (!file) return;
    setError(""); setMessage(""); setSummary(null); setValidationErrors([]);
    try { setSource(JSON.parse(await file.text())); setMessage("Evaluation JSON is ready to run."); } catch { setSource(null); setError("The selected file must contain valid JSON."); }
  }

  async function runEvaluation(upload: unknown = [{ question, category, expected_answer_or_rubric: expected, supporting_passage: passage }]) {
    if (upload === null) return;
    setBusy(true); setError(""); setMessage(""); setRun(null); setRunId(""); setReview(null); setReviewError(""); setReviewMessage(""); setReviewDecision(""); setExceptionDecisions({}); setReviewReason("");
    const submitted = Array.isArray(upload) && upload.length === 1 && typeof upload[0] === "object" && upload[0] !== null && "question" in upload[0] ? String((upload[0] as { question?: unknown }).question ?? "") : "";
    setSubmittedQuestion(submitted);
    try {
      const result = await uploadAndRunEvaluation(token, upload);
      setSummary(result); setValidationErrors(result.validation_errors);
      setMessage(result.status === "accepted" ? `Accepted ${result.imported_count} of ${result.submitted_count} records; ${result.skipped_count} skipped.` : `No valid records were accepted. ${result.skipped_count} records were skipped.`);
      if (result.status === "accepted" && result.run_id) { setRunId(result.run_id); setRun({ run_id: result.run_id, status: "queued", metrics: [], item_results: [], error: null }); }
    } catch (cause) { showError(cause); } finally { setBusy(false); }
  }

  async function submitBatchDecision(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!runId || !review || !reviewDecision || !exceptionsReady) return;
    const decisions: ExplicitExceptionDecision[] = review.exceptions.map((item) => {
      const selected = exceptionDecisions[item.item_id];
      return { item_id: item.item_id, decision: selected?.decision as "rejected" | "inapplicable", reason: selected?.reason.trim() || null };
    });
    setReviewBusy(true); setReviewError(""); setReviewMessage("");
    try {
      await attestBatchReview(token, runId, { decision: reviewDecision, exception_decisions: decisions, reason: reviewDecision === "rejected" ? reviewReason.trim() : null });
      setReviewMessage("Batch review decision saved."); setReviewRefresh((value) => value + 1);
    } catch (cause) {
      setReviewError(cause instanceof EvaluationApiError && [401, 403].includes(cause.status)
        ? "You do not have administrator access to review this run."
        : cause instanceof EvaluationApiError && cause.status === 409
          ? "This review is stale or already decided. Refresh it before trying again."
          : cause instanceof Error ? cause.message : "The review decision could not be saved.");
    } finally { setReviewBusy(false); }
  }

  const exceptionsReady = review?.exceptions.every((item) => {
    const selected = exceptionDecisions[item.item_id];
    return selected?.decision && !!selected.reason.trim();
  }) ?? false;
  const reviewOpen = !!review && ["eligible", "exception"].includes(review.summary.status) && !review.summary.decision;
  const decisionReady = reviewOpen && !!reviewDecision && (reviewDecision !== "rejected" || !!reviewReason.trim()) && exceptionsReady;

  function refreshReview() {
    setReviewDecision(""); setReviewReason(""); setExceptionDecisions({}); setReviewRefresh((value) => value + 1);
  }

  return <section className="admin-card evaluation-panel" aria-labelledby="evaluation-title">
    <div className="section-heading"><div><h3 id="evaluation-title">Evaluation</h3><p>Upload a JSON evaluation set and run it against the server-selected current corpus.</p></div>{run && <span className={`state state-${run.status}`}>{run.status}</span>}</div>
    {error && <p className="error" role="alert">{error}</p>}{message && <p className="status" role="status">{message}</p>}
    <button type="button" onClick={() => setSingleQuestionOpen((open) => !open)}>{singleQuestionOpen ? "Hide individual validator" : "Run individual check"}</button>
    {singleQuestionOpen && <fieldset disabled={busy}><legend>Validate one question</legend>
      <label htmlFor="evaluation-question">Question<input id="evaluation-question" value={question} onChange={(event) => setQuestion(event.target.value)} /></label>
      <label htmlFor="evaluation-category">Category<select id="evaluation-category" value={category} onChange={(event) => setCategory(event.target.value)}><option value="definitions">Definitions</option><option value="explanations">Explanations</option><option value="comparisons">Comparisons</option><option value="study">Study</option><option value="scenarios">Scenarios</option></select></label>
      <label htmlFor="evaluation-expected">Expected answer or rubric<textarea id="evaluation-expected" value={expected} onChange={(event) => setExpected(event.target.value)} /></label>
      <label htmlFor="evaluation-passage">Supporting passage<textarea id="evaluation-passage" value={passage} onChange={(event) => setPassage(event.target.value)} /></label>
      <button type="button" onClick={() => void runEvaluation()} disabled={busy || !question.trim() || !expected.trim() || !passage.trim()}>{busy ? "Running…" : "Run question"}</button>
    </fieldset>}
    <details><summary>Run a JSON evaluation set</summary><label htmlFor="evaluation-json">Evaluation JSON<input id="evaluation-json" type="file" accept="application/json,.json" onChange={selectFile} disabled={busy} /></label><button type="button" onClick={() => void runEvaluation(source)} disabled={busy || source === null}>Run evaluation set</button></details>
    {validationErrors.length > 0 && <section aria-labelledby="validation-title"><h4 id="validation-title">Validation feedback</h4><ul className="error">{validationErrors.map((item) => <li key={`${item.index}-${item.code}`}>Record {item.index + 1}: {item.message}</li>)}</ul></section>}
    {submittedQuestion && <p role="status">Submitted question: {submittedQuestion}</p>}
    {summary && <dl aria-label="Import counts"><div><dt>Submitted</dt><dd>{summary.submitted_count}</dd></div><div><dt>Accepted</dt><dd>{summary.imported_count}</dd></div><div><dt>Rejected</dt><dd>{summary.skipped_count}</dd></div></dl>}
    {run && <RunStatus run={run} />}
    {run?.status === "completed" && runId && <section className="batch-review" aria-labelledby="batch-review-title">
      <h4 id="batch-review-title">Batch evidence review</h4>
      {reviewLoading && <p role="status">Loading evidence review…</p>}
      {reviewError && <p className="error" role="alert">{reviewError}</p>}
      {reviewMessage && <p className="status" role="status">{reviewMessage}</p>}
      {reviewError && <button type="button" onClick={refreshReview} disabled={reviewLoading}>Refresh review</button>}
      {review && <>
        <dl aria-label="Review summary" className="review-summary"><div><dt>Status</dt><dd>{reviewStatusLabel(review.summary.status)}</dd></div><div><dt>Eligible</dt><dd>{review.summary.eligible_count}</dd></div><div><dt>Exceptions</dt><dd>{review.summary.exception_count}</dd></div><div><dt>Representative sample</dt><dd>{review.summary.sampled_count}</dd></div><div><dt>Attested</dt><dd>{review.summary.attested_count}</dd></div></dl>
        {!reviewOpen && <p role="status">This review is stale or already decided and cannot be changed.</p>}
        <h5>Representative sample</h5>
        {review.sample.length === 0 ? <p>No representative samples are available.</p> : <ul className="review-list">{review.sample.map((item, index) => <li key={`${item.page}-${item.section}-${index}`}><p><strong>{item.question}</strong></p><p>{item.rubric}</p><blockquote>{item.support_excerpt}</blockquote><p>Page {item.page} · {item.section} · {reviewStatusLabel(item.status)}</p></li>)}</ul>}
        <h5>Exceptions requiring a decision</h5>
        {review.exceptions.length === 0 && <p>No exceptions require a decision.</p>}
        <form onSubmit={submitBatchDecision} aria-describedby="exception-guidance">
          <p id="exception-guidance">Choose a decision for every exception. Record identity stays hidden; decisions are linked automatically.</p>
          {review.exceptions.length > 0 && <ul className="review-list">{review.exceptions.map((item, index) => <li key={`${item.page ?? ""}-${item.section ?? ""}-${index}`}>
            <p><strong>{item.question}</strong></p><p>{item.rubric}</p>{item.support_excerpt && <blockquote>{item.support_excerpt}</blockquote>}
            <p>{item.page != null ? `Page ${item.page} · ` : ""}{item.section ?? "Source location unavailable"} · {item.status}: {item.reason}</p>
            <label htmlFor={`exception-decision-${index}`}>Decision<select id={`exception-decision-${index}`} value={exceptionDecisions[item.item_id]?.decision ?? ""} disabled={reviewBusy || !!reviewError || !reviewOpen} onChange={(event) => setExceptionDecisions((current) => ({ ...current, [item.item_id]: { decision: event.target.value as "rejected" | "inapplicable" | "", reason: current[item.item_id]?.reason ?? "" } }))}><option value="">Choose a decision</option><option value="rejected">Reject exception</option><option value="inapplicable">Mark inapplicable</option></select></label>
            {exceptionDecisions[item.item_id]?.decision && <label htmlFor={`exception-reason-${index}`}>Exception reason<textarea id={`exception-reason-${index}`} value={exceptionDecisions[item.item_id]?.reason ?? ""} disabled={reviewBusy || !!reviewError || !reviewOpen} onChange={(event) => setExceptionDecisions((current) => ({ ...current, [item.item_id]: { decision: current[item.item_id]?.decision ?? "", reason: event.target.value } }))} /></label>}
          </li>)}</ul>
          }
          <fieldset disabled={reviewBusy || !!reviewError || !reviewOpen}><legend>Batch decision</legend>
            <label className="review-choice"><input type="radio" name="batch-review-decision" value="attested" checked={reviewDecision === "attested"} onChange={() => setReviewDecision("attested")} />Attest representative sample</label>
            <label className="review-choice"><input type="radio" name="batch-review-decision" value="rejected" checked={reviewDecision === "rejected"} onChange={() => setReviewDecision("rejected")} />Reject batch review</label>
            {reviewDecision === "rejected" && <label htmlFor="batch-review-reason">Batch rejection reason<textarea id="batch-review-reason" value={reviewReason} onChange={(event) => setReviewReason(event.target.value)} /></label>}
            <button type="submit" disabled={reviewBusy || !!review.summary.decision || !decisionReady}>{reviewBusy ? "Saving…" : "Save batch decision"}</button>
          </fieldset>
        </form>
      </>}
    </section>}
  </section>;
}

function reviewStatusLabel(status: BatchReviewStatus) { return status.replaceAll("_", " "); }

function RunStatus({ run }: { run: EvaluationStatus }) {
  if (run.status === "failed") return <p className="error" role="alert">{run.error ?? "The evaluation failed safely. No results are available."}</p>;
  if (run.status !== "completed") return <p role="status">Evaluation is processing…</p>;
  return <>{run.item_results.length === 0 ? <p role="status">No questions evaluated.</p> : <section aria-labelledby="items-title"><h4 id="items-title">Question results</h4><ul>{run.item_results.map((item) => <ItemResult key={item.item_id} item={item} />)}</ul></section>}{run.metrics.length > 0 && <section aria-labelledby="metrics-title"><h4 id="metrics-title">Aggregate results</h4><ul className="gate-list">{run.metrics.map((metric) => <MetricResult key={metric.metric_name} metric={metric} />)}</ul></section>}</>;
}

function ItemResult({ item }: { item: EvaluationItemResult }) { return <li><strong>{item.item_id}</strong><span>Retrieval {item.retrieval_success ? "pass" : "fail"}; grounded claims {item.grounded ? "pass" : "fail"}; citations {item.citation_correct ? "pass" : "fail"}; answer quality {item.answer_quality ? "pass" : "fail"}; refusal {item.refusal_correct ? "pass" : "fail"}; {item.response_time_ms} ms</span></li>; }

function MetricResult({ metric }: { metric: EvaluationMetric }) { return <li className={metric.passed ? "gate-passed" : "gate-failed"}><strong>{metricLabels[metric.metric_name] ?? metric.metric_name}</strong><span>{metric.numerator} / {metric.denominator} · {metric.percentage}% · {metric.passed ? "Pass" : "Fail"}</span></li>; }
