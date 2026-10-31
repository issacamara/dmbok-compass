import { useEffect, useState, type ChangeEvent } from "react";
import type { EvaluationItemResult, EvaluationMetric, EvaluationRun, ImportValidationError } from "../../api/contracts";
import { EvaluationApiError, getEvaluationRun, uploadAndRunEvaluation, type EvaluationUploadRunResponse } from "./api";

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

  useEffect(() => {
    if (!runId || !run || !["queued", "running"].includes(run.status)) return;
    const timer = window.setTimeout(() => { void getEvaluationRun(token, runId).then(setRun).catch(showError); }, 500);
    return () => window.clearTimeout(timer);
  }, [run, runId, token]);

  function showError(cause: unknown) { setError(cause instanceof EvaluationApiError && cause.status === 403 ? "You do not have administrator access." : cause instanceof Error ? cause.message : "The evaluation service could not complete the request."); }

  async function selectFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]; event.target.value = ""; if (!file) return;
    setError(""); setMessage(""); setSummary(null); setValidationErrors([]);
    try { setSource(JSON.parse(await file.text())); setMessage("Evaluation JSON is ready to run."); } catch { setSource(null); setError("The selected file must contain valid JSON."); }
  }

  async function runEvaluation(upload: unknown = [{ question, category, expected_answer_or_rubric: expected, supporting_passage: passage }]) {
    if (upload === null) return;
    setBusy(true); setError(""); setMessage(""); setRun(null); setRunId("");
    const submitted = Array.isArray(upload) && upload.length === 1 && typeof upload[0] === "object" && upload[0] !== null && "question" in upload[0] ? String((upload[0] as { question?: unknown }).question ?? "") : "";
    setSubmittedQuestion(submitted);
    try {
      const result = await uploadAndRunEvaluation(token, upload);
      setSummary(result); setValidationErrors(result.validation_errors);
      setMessage(result.status === "accepted" ? `Accepted ${result.imported_count} of ${result.submitted_count} records; ${result.skipped_count} skipped.` : `No valid records were accepted. ${result.skipped_count} records were skipped.`);
      if (result.status === "accepted" && result.run_id) { setRunId(result.run_id); setRun({ run_id: result.run_id, status: "queued", metrics: [], item_results: [], error: null }); }
    } catch (cause) { showError(cause); } finally { setBusy(false); }
  }

  return <section className="admin-card evaluation-panel" aria-labelledby="evaluation-title">
    <div className="section-heading"><div><h3 id="evaluation-title">Evaluation</h3><p>Upload a JSON evaluation set and run it against the server-selected current corpus.</p></div>{run && <span className={`state state-${run.status}`}>{run.status}</span>}</div>
    {error && <p className="error" role="alert">{error}</p>}{message && <p className="status" role="status">{message}</p>}
    <fieldset disabled={busy}><legend>Validate one question</legend>
      <label htmlFor="evaluation-question">Question<input id="evaluation-question" value={question} onChange={(event) => setQuestion(event.target.value)} /></label>
      <label htmlFor="evaluation-category">Category<select id="evaluation-category" value={category} onChange={(event) => setCategory(event.target.value)}><option value="definitions">Definitions</option><option value="explanations">Explanations</option><option value="comparisons">Comparisons</option><option value="study">Study</option><option value="scenarios">Scenarios</option></select></label>
      <label htmlFor="evaluation-expected">Expected answer or rubric<textarea id="evaluation-expected" value={expected} onChange={(event) => setExpected(event.target.value)} /></label>
      <label htmlFor="evaluation-passage">Supporting passage<textarea id="evaluation-passage" value={passage} onChange={(event) => setPassage(event.target.value)} /></label>
      <button type="button" onClick={() => void runEvaluation()} disabled={busy || !question.trim() || !expected.trim() || !passage.trim()}>{busy ? "Running…" : "Run question"}</button>
    </fieldset>
    <details><summary>Run a JSON evaluation set</summary><label htmlFor="evaluation-json">Evaluation JSON<input id="evaluation-json" type="file" accept="application/json,.json" onChange={selectFile} disabled={busy} /></label><button type="button" onClick={() => void runEvaluation(source)} disabled={busy || source === null}>Run evaluation set</button></details>
    {validationErrors.length > 0 && <section aria-labelledby="validation-title"><h4 id="validation-title">Validation feedback</h4><ul className="error">{validationErrors.map((item) => <li key={`${item.index}-${item.code}`}>Record {item.index + 1}: {item.message}</li>)}</ul></section>}
    {submittedQuestion && <p role="status">Submitted question: {submittedQuestion}</p>}
    {summary && <dl aria-label="Import counts"><div><dt>Submitted</dt><dd>{summary.submitted_count}</dd></div><div><dt>Accepted</dt><dd>{summary.imported_count}</dd></div><div><dt>Rejected</dt><dd>{summary.skipped_count}</dd></div></dl>}
    {run && <RunStatus run={run} />}
  </section>;
}

function RunStatus({ run }: { run: EvaluationStatus }) {
  if (run.status === "failed") return <p className="error" role="alert">{run.error ?? "The evaluation failed safely. No results are available."}</p>;
  if (run.status !== "completed") return <p role="status">Evaluation is processing…</p>;
  return <>{run.item_results.length === 0 ? <p role="status">No questions evaluated.</p> : <section aria-labelledby="items-title"><h4 id="items-title">Question results</h4><ul>{run.item_results.map((item) => <ItemResult key={item.item_id} item={item} />)}</ul></section>}{run.metrics.length > 0 && <section aria-labelledby="metrics-title"><h4 id="metrics-title">Aggregate results</h4><ul className="gate-list">{run.metrics.map((metric) => <MetricResult key={metric.metric_name} metric={metric} />)}</ul></section>}</>;
}

function ItemResult({ item }: { item: EvaluationItemResult }) { return <li><strong>{item.item_id}</strong><span>Retrieval {item.retrieval_success ? "pass" : "fail"}; grounded claims {item.grounded ? "pass" : "fail"}; citations {item.citation_correct ? "pass" : "fail"}; answer quality {item.answer_quality ? "pass" : "fail"}; refusal {item.refusal_correct ? "pass" : "fail"}; {item.response_time_ms} ms</span></li>; }

function MetricResult({ metric }: { metric: EvaluationMetric }) { return <li className={metric.passed ? "gate-passed" : "gate-failed"}><strong>{metricLabels[metric.metric_name] ?? metric.metric_name}</strong><span>{metric.numerator} / {metric.denominator} · {metric.percentage}% · {metric.passed ? "Pass" : "Fail"}</span></li>; }
