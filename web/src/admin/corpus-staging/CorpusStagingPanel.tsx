import { useEffect, useRef, useState, type FormEvent } from "react";
import { approveCorpusStagingAttempt, getCorpusStagingStatus, getCorpusStagingTemplate, submitCorpusStagingAttempt, validateCorpusStagingAttempt } from "../../api/admin";
import type { CorpusStagingAttemptStatus } from "../../api/contracts";

const ATTEMPT_KEY = "dmbok-compass.corpus-staging-attempt";

export function CorpusStagingPanel({ token }: { token: string }) {
  const [attemptId, setAttemptId] = useState(() => window.localStorage.getItem(ATTEMPT_KEY) ?? "");
  const [attempt, setAttempt] = useState<CorpusStagingAttemptStatus | null>(null);
  const [pdf, setPdf] = useState<File | null>(null);
  const [structure, setStructure] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const errorSummary = useRef<HTMLDivElement>(null);

  useEffect(() => { if (error) errorSummary.current?.focus(); }, [error]);

  async function refresh(id = attemptId) {
    if (!id) return;
    try {
      const status = await getCorpusStagingStatus(token, id);
      setAttempt(status);
      if (status.state === "failed" || status.state === "staged") setMessage("");
    } catch { setError("Could not refresh this staging attempt. Check access or try again shortly."); }
  }

  useEffect(() => { void refresh(); }, [token]);
  useEffect(() => {
    if (!attemptId || !attempt || !["queued", "running", "validating"].includes(attempt.state)) return;
    const timer = window.setInterval(() => { void refresh(attemptId); }, 2000);
    return () => window.clearInterval(timer);
  }, [attemptId, attempt?.state, token]);

  async function downloadTemplate() {
    setError("");
    try {
      const template = await getCorpusStagingTemplate(token);
      const href = URL.createObjectURL(new Blob([JSON.stringify(template, null, 2)], { type: "application/json" }));
      const link = document.createElement("a"); link.href = href; link.download = "dmbok-structure-template-v1.json"; link.click();
      window.setTimeout(() => URL.revokeObjectURL(href), 1000);
    } catch { setError("Could not download the structure template. Try again shortly."); }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setError(""); setMessage("");
    if (!pdf || !structure) { setError("Choose both a source PDF and its structure JSON file."); return; }
    setBusy(true);
    try {
      const submitted = await submitCorpusStagingAttempt(token, pdf, structure);
      setAttemptId(submitted.attempt_id); window.localStorage.setItem(ATTEMPT_KEY, submitted.attempt_id); setAttempt(submitted);
      const preview = await validateCorpusStagingAttempt(token, submitted.attempt_id);
      setAttempt(preview); setMessage("Review every section match before approving this attempt.");
    } catch (cause) { setError(safeError(cause)); }
    finally { setBusy(false); }
  }

  async function approve() {
    if (!attempt) return;
    setBusy(true); setError("");
    try { const updated = await approveCorpusStagingAttempt(token, attempt.attempt_id, attempt.revision); setAttempt(updated); setMessage("Approval recorded. Staging progress will update here."); }
    catch (cause) { setError(safeError(cause)); await refresh(attempt.attempt_id); }
    finally { setBusy(false); }
  }

  const approvable = attempt?.state === "awaiting_review" && !!attempt.preview?.length && attempt.preview.every((entry) => entry.match_status === "matched" && entry.correction_codes.length === 0) && attempt.correction_codes.length === 0;
  return <section className="admin-card staging-card" aria-labelledby="corpus-staging-title">
    <div className="section-heading"><div><h3 id="corpus-staging-title">Stage a corpus</h3><p>Upload a PDF with its completed structure file. A staged corpus does not become active until separately released.</p></div>
      <button type="button" className="secondary" onClick={() => void downloadTemplate()}>Download structure template</button></div>
    {error && <div id="staging-error-summary" className="error" role="alert" tabIndex={-1} ref={errorSummary}>
      <p>{error}</p><a href={attempt ? "#staging-refresh" : "#staging-pdf"}>Review staging controls</a>
    </div>}
    {message && <p className="status" role="status">{message}</p>}
    <form className="staging-form" onSubmit={(event) => void submit(event)}>
      <label htmlFor="staging-pdf">Source PDF</label><input id="staging-pdf" type="file" accept="application/pdf,.pdf" onChange={(event) => setPdf(event.target.files?.[0] ?? null)} />
      <label htmlFor="staging-structure">Structure JSON</label><input id="staging-structure" type="file" accept="application/json,.json" onChange={(event) => setStructure(event.target.files?.[0] ?? null)} />
      <button type="submit" disabled={busy || !pdf || !structure}>{busy ? "Working…" : "Upload and preview"}</button>
    </form>
    {attempt && <section className="staging-review" aria-labelledby="staging-review-title">
      <div className="section-heading"><h4 id="staging-review-title">Staging attempt</h4><button id="staging-refresh" className="secondary" type="button" onClick={() => void refresh()} disabled={busy}>Refresh status</button></div>
      <p role="status">Status: {attempt.state.replaceAll("_", " ")}{attempt.completed_stage ? ` · completed ${attempt.completed_stage}` : ""}</p>
      {attempt.preview && <><h5>Review section matches</h5><ul className="staging-entry-list">{attempt.preview.map((entry) => <li key={entry.entry_index}>
        <strong>{entry.section_path.join(" › ") || `Entry ${entry.entry_index + 1}`}</strong>
        <span>Pages {entry.page_start}–{entry.page_end} · {entry.match_status}</span>
        {entry.correction_codes.length > 0 && <p className="error">Correction needed: {entry.correction_codes.map(correctionText).join("; ")}. Correct this entry in the structure JSON and submit a new attempt.</p>}
        {entry.match_status !== "matched" && entry.correction_codes.length === 0 && <p className="error">This section could not be matched. Correct its heading or page range in the structure JSON and submit a new attempt.</p>}
      </li>)}</ul></>}
      {attempt.correction_codes.length > 0 && <p className="error">{attempt.correction_codes.map(correctionText).join("; ")}. Correct the structure JSON and submit a new attempt.</p>}
      {attempt.state === "failed" && <p className="error">{attempt.error?.message ?? "Staging failed safely. Review the correction guidance and submit a new attempt."}</p>}
      {attempt.state === "staged" && <p className="status">Staged version {attempt.staged_corpus_version_id}; {attempt.parent_count ?? 0} parents and {attempt.chunk_count ?? attempt.child_count ?? 0} chunks.</p>}
      <button type="button" disabled={!approvable || busy} onClick={() => void approve()}>Approve and create chunks</button>
    </section>}
  </section>;
}

function safeError(cause: unknown): string {
  if (typeof cause === "object" && cause !== null && "status" in cause && cause.status === 409) return "The reviewed attempt changed. Refresh and review the current revision before approving.";
  if (cause instanceof Error && cause.message && cause.message.length < 200) return cause.message;
  return "The staging request could not be completed. Try again shortly.";
}

function correctionText(code: string): string {
  const messages: Record<string, string> = {
    invalid_structure: "the structure file is invalid", source_pair_mismatch: "the files do not match", missing_page_coverage: "some pages need section coverage",
    ambiguous_heading: "a heading matches more than one section", unmatched_passage: "a passage has no matching section", stale_revision: "the reviewed revision is stale",
    job_interrupted: "staging was interrupted", job_failed: "staging did not complete",
  };
  return messages[code] ?? "the entry needs correction";
}
