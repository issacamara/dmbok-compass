import { useEffect, useState, type FormEvent } from "react";
import type { EvaluationRun, EvaluationMetric } from "../../api/contracts";
import { EvaluationApiError, getEvaluationRun, launchEvaluation, recordReleaseDecision } from "./api";

type EvaluationPanelProps = { token: string };
type RunMode = "full" | "subset";

const metricLabels: Record<string, string> = {
  retrieval_success: "Top-5 retrieval success", grounded_claims: "Grounded claims", citation_correctness: "Citation correctness",
  answer_quality: "Answer quality", refusal_correctness: "Refusal correctness", response_time: "Response time",
};

export function EvaluationPanel({ token }: EvaluationPanelProps) {
  const [corpusVersion, setCorpusVersion] = useState("corpus-v1");
  const [configurationVersion, setConfigurationVersion] = useState("config-v1");
  const [modelVersion, setModelVersion] = useState("model-v1");
  const [providerVersion, setProviderVersion] = useState("provider-v1");
  const [scorerVersion, setScorerVersion] = useState("scorer-v1");
  const [itemIds, setItemIds] = useState("");
  const [runMode, setRunMode] = useState<RunMode>("full");
  const [run, setRun] = useState<EvaluationRun | null>(null);
  const [decision, setDecision] = useState<"approved" | "rejected">("approved");
  const [rationale, setRationale] = useState("");
  const [gateReportIds, setGateReportIds] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (!run || !["queued", "running"].includes(run.status)) return;
    const timer = window.setTimeout(() => { void getEvaluationRun(token, run.run_id).then(setRun).catch(showError); }, 500);
    return () => window.clearTimeout(timer);
  }, [run, token]);

  function showError(cause: unknown) {
    setError(cause instanceof EvaluationApiError && cause.status === 403 ? "You do not have administrator access." : cause instanceof Error ? cause.message : "The evaluation service could not complete the request.");
  }

  async function startRun(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true); setError(""); setMessage(""); setRun(null);
    try {
      const selected = itemIds.split(/[,\n\s]+/).map((item) => item.trim()).filter(Boolean);
      const started = await launchEvaluation(token, { corpus_version_id: corpusVersion, configuration_version_id: configurationVersion, model_version_id: modelVersion, ...(runMode === "subset" ? { item_ids: selected } : {}) });
      setRun(started); setMessage(runMode === "full" ? "Full evaluation started." : "Subset evaluation started.");
    } catch (cause) { showError(cause); } finally { setBusy(false); }
  }

  async function submitDecision(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); if (!run) return;
    setBusy(true); setError(""); setMessage("");
    try {
      await recordReleaseDecision(token, { release_id: `release-${run.run_id}`, evaluation_run_id: run.run_id, dataset_version_id: run.dataset_version_id, corpus_version_id: run.corpus_version_id, configuration_version_id: run.configuration_version_id, model_version_id: run.model_version_id, provider_version_id: providerVersion, scorer_version_id: scorerVersion, gate_report_ids: gateReportIds.split(/[,\n\s]+/).map((item) => item.trim()).filter(Boolean), decision, rationale, exception_approved: false, exception_rationale: null });
      setMessage(`Release decision recorded: ${decision}.`); setConfirmed(false);
    } catch (cause) { showError(cause); } finally { setBusy(false); }
  }

  const metrics = run?.metrics ?? [];
  const allPassed = metrics.length > 0 && metrics.every((metric) => metric.passed);
  return <section className="admin-card evaluation-panel" aria-labelledby="evaluation-title">
    <div className="section-heading"><div><h3 id="evaluation-title">Evaluation and release decision</h3><p>Run a complete dataset or an explicit subset, then review immutable version-bound gate evidence.</p></div>{run && <span className={`state state-${run.status}`}>{run.status}</span>}</div>
    {error && <p className="error" role="alert">{error}</p>}{message && <p className="status" role="status">{message}</p>}
    <form className="evaluation-form" onSubmit={startRun}><fieldset><legend>Run scope</legend><p>The active imported dataset is selected automatically.</p><label><input type="radio" checked={runMode === "full"} onChange={() => setRunMode("full")} /> Full evaluation set</label><label><input type="radio" checked={runMode === "subset"} onChange={() => setRunMode("subset")} /> Selected subset</label>{runMode === "subset" && <><label htmlFor="evaluation-items">Item IDs</label><textarea id="evaluation-items" value={itemIds} onChange={(event) => setItemIds(event.target.value)} placeholder="item-1, item-2" required /></>}</fieldset><fieldset><legend>Release candidate versions</legend><VersionInput id="corpus-version" label="Corpus version" value={corpusVersion} onChange={setCorpusVersion} /><VersionInput id="configuration-version" label="Configuration version" value={configurationVersion} onChange={setConfigurationVersion} /><VersionInput id="model-version" label="Model version" value={modelVersion} onChange={setModelVersion} /></fieldset><button type="submit" disabled={busy}>{busy ? "Starting…" : `Run ${runMode} evaluation`}</button></form>
    {run && <RunReport run={run} />}
    {run?.status === "completed" && <form className="decision-form" onSubmit={submitDecision}><h4>Sponsor decision</h4><p className={allPassed ? "status" : "error"}>{metrics.length === 0 ? "Gate results are missing." : allPassed ? "All reported gates passed." : "One or more gates failed."}</p><VersionInput id="provider-version" label="Provider version" value={providerVersion} onChange={setProviderVersion} /><VersionInput id="scorer-version" label="Scorer version" value={scorerVersion} onChange={setScorerVersion} /><label htmlFor="gate-reports">Gate report IDs<input id="gate-reports" value={gateReportIds} onChange={(event) => setGateReportIds(event.target.value)} placeholder="report-1, report-2" required /></label><label htmlFor="decision">Decision<select id="decision" value={decision} onChange={(event) => setDecision(event.target.value as "approved" | "rejected")}><option value="approved">Approve release</option><option value="rejected">Reject release</option></select></label><label htmlFor="rationale">Rationale<textarea id="rationale" value={rationale} onChange={(event) => setRationale(event.target.value)} required /></label><label className="confirmation"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} required /> I confirm this sponsor decision is based on the displayed evidence.</label><button type="submit" disabled={busy || !confirmed}>{busy ? "Recording…" : "Record sponsor decision"}</button></form>}
  </section>;
}

function VersionInput({ id, label, value, onChange }: { id: string; label: string; value: string; onChange: (value: string) => void }) { return <label htmlFor={id}>{label}<input id={id} value={value} onChange={(event) => onChange(event.target.value)} required /></label>; }

function RunReport({ run }: { run: EvaluationRun }) {
  return <section className="evaluation-report" aria-labelledby="report-title"><h4 id="report-title">Run {run.run_id}</h4><dl className="version-context"><div><dt>Dataset</dt><dd>{run.dataset_version_id}</dd></div><div><dt>Corpus</dt><dd>{run.corpus_version_id}</dd></div><div><dt>Configuration</dt><dd>{run.configuration_version_id}</dd></div><div><dt>Model</dt><dd>{run.model_version_id}</dd></div></dl>{run.status !== "completed" ? <p role="status">{run.status === "failed" ? run.error ?? "Evaluation failed." : "Evaluation is processing…"}</p> : run.metrics.length === 0 ? <p role="status">No gate results were reported.</p> : <ul className="gate-list">{run.metrics.map((metric) => <GateResult key={metric.metric_name} metric={metric} />)}</ul>}</section>;
}

function GateResult({ metric }: { metric: EvaluationMetric }) { return <li className={metric.passed ? "gate-passed" : "gate-failed"}><div><strong>{metricLabels[metric.metric_name] ?? metric.metric_name}</strong><span>{metric.numerator} / {metric.denominator}</span></div><span>{metric.percentage}% · threshold {metric.threshold}% · {metric.passed ? "Pass" : "Fail"}</span></li>; }
