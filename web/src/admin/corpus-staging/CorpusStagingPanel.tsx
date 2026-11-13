import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  approveCorpusStagingAttempt,
  correctCorpusStagingStructure,
  getCorpusStagingStatus,
  proposeCorpusStagingStructure,
  startCorpusStagingParse,
  submitPdfOnlyCorpusStagingAttempt,
} from "../../api/admin";
import type { CorpusStagingAttemptStatus, CorpusStagingStructureCorrectionEntry } from "../../api/contracts";

const ATTEMPT_KEY = "dmbok-compass.corpus-staging-attempt";
const SUBMISSION_KEY = "dmbok-compass.corpus-staging-submission";
const ACTIVE_PARSING_STATES = ["queued", "parsing"];

function nextSubmissionId() {
  const id = crypto.randomUUID();
  window.localStorage.setItem(SUBMISSION_KEY, id);
  return id;
}

export function CorpusStagingPanel({ token }: { token: string }) {
  const [attemptId, setAttemptId] = useState(() => window.localStorage.getItem(ATTEMPT_KEY) ?? "");
  const [existingAttemptId, setExistingAttemptId] = useState("");
  const [attempt, setAttempt] = useState<CorpusStagingAttemptStatus | null>(null);
  const [pdf, setPdf] = useState<File | null>(null);
  const [submissionId, setSubmissionId] = useState(() => window.localStorage.getItem(SUBMISSION_KEY) ?? nextSubmissionId());
  const [physicalPageSelection, setPhysicalPageSelection] = useState("21, 25, 58, 78, 105");
  const [sectionPaths, setSectionPaths] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const errorSummary = useRef<HTMLDivElement>(null);

  useEffect(() => { if (error) errorSummary.current?.focus(); }, [error]);
  useEffect(() => {
    if (!attempt) { setSectionPaths([]); return; }
    setSectionPaths(attempt.preview?.map((entry) => entry.section_path.join(" › ")) ?? []);
  }, [attempt?.revision, attempt?.preview]);

  async function refresh(id = attemptId) {
    if (!id) return;
    try {
      const status = await getCorpusStagingStatus(token, id);
      setAttempt(status);
      if (status.state === "failed" || status.state === "staged") setMessage("");
      if (status.state === "validating" && status.hosted_parsing_state === "awaiting_review" && !status.preview) {
        const proposal = await proposeCorpusStagingStructure(token, id);
        setAttempt(proposal);
        setMessage("Review the generated outline and save any corrections before approval.");
      }
    } catch (cause) { setError(safeError(cause, "Could not refresh this staging attempt. Check access or try again shortly.")); }
  }

  useEffect(() => { void refresh(); }, [token]);
  useEffect(() => {
    const parsing = ACTIVE_PARSING_STATES.includes(attempt?.hosted_parsing_state ?? "");
    const ingesting = attempt?.state === "queued" || attempt?.state === "running";
    if (!attemptId || !attempt || (!parsing && !ingesting)) return;
    const timer = window.setInterval(() => { void refresh(attemptId); }, 2000);
    return () => window.clearInterval(timer);
  }, [attemptId, attempt?.state, attempt?.hosted_parsing_state, token]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setError(""); setMessage("");
    if (!pdf) { setError("Choose a source PDF to begin staging."); return; }
    const pages = parsePhysicalPageSelection(physicalPageSelection);
    if (!pages) { setError("Enter up to five unique physical page numbers, separated by commas."); return; }
    setBusy(true);
    try {
      const submitted = await submitPdfOnlyCorpusStagingAttempt(token, pdf, pages, submissionId);
      setAttemptId(submitted.attempt_id);
      window.localStorage.setItem(ATTEMPT_KEY, submitted.attempt_id);
      setAttempt(submitted);
      const started = await startCorpusStagingParse(token, submitted.attempt_id);
      const parsed = started.hosted_parsing_state === "awaiting_review" && !started.preview
        ? await proposeCorpusStagingStructure(token, submitted.attempt_id)
        : started;
      setAttempt(parsed);
      setSubmissionId(nextSubmissionId());
      setMessage(parsed.hosted_parsing_state === "awaiting_review"
        ? "Review the generated outline and save any corrections before approval."
        : parsed.hosted_parsing_state === "awaiting_budget"
        ? "Parsing is waiting for available budget. Refresh status or retry parsing when budget is available."
        : "PDF uploaded. Parsing will update here; you can leave and return to this staging attempt.");
      setPdf(null);
    } catch (cause) { setError(safeError(cause)); }
    finally { setBusy(false); }
  }

  async function retryParsing() {
    if (!attempt) return;
    setBusy(true); setError(""); setMessage("");
    try {
      const started = await startCorpusStagingParse(token, attempt.attempt_id);
      const status = started.hosted_parsing_state === "awaiting_review" && !started.preview
        ? await proposeCorpusStagingStructure(token, attempt.attempt_id)
        : started;
      setAttempt(status);
      setMessage(status.hosted_parsing_state === "awaiting_budget"
        ? "Parsing is still waiting for available budget. Refresh or retry later."
        : status.hosted_parsing_state === "parsing"
          ? "Resumed the saved parse job. Status will update here."
        : status.hosted_parsing_state === "awaiting_review"
          ? "Review the generated outline and save any corrections before approval."
          : "Parsing was requested. Status will update here.");
    } catch (cause) { setError(safeError(cause)); await refresh(attempt.attempt_id); }
    finally { setBusy(false); }
  }

  async function loadExistingAttempt() {
    const id = existingAttemptId.trim();
    if (!id) return;
    setBusy(true); setError(""); setMessage("");
    try {
      const status = await getCorpusStagingStatus(token, id);
      setAttemptId(status.attempt_id);
      window.localStorage.setItem(ATTEMPT_KEY, status.attempt_id);
      setAttempt(status);
      if (status.state === "validating" && status.hosted_parsing_state === "awaiting_review" && !status.preview) {
        const proposal = await proposeCorpusStagingStructure(token, status.attempt_id);
        setAttempt(proposal);
        setMessage("Review the generated outline and save any corrections before approval.");
      } else {
        setMessage("Opened the existing staging attempt.");
      }
    } catch (cause) { setError(safeError(cause, "Could not open this staging attempt. Check the ID or your access.")); }
    finally { setBusy(false); }
  }

  async function saveOutline() {
    if (!attempt?.preview) return;
    const entries: CorpusStagingStructureCorrectionEntry[] = attempt.preview.map((entry, index) => ({
      section_path: (sectionPaths[index] ?? "").split(/[›/]/).map((part) => part.trim()).filter(Boolean),
      block_ids: entry.block_ids ?? [],
    }));
    if (entries.some((entry) => entry.section_path.length === 0 || entry.block_ids.length === 0)) {
      setError("Each outline entry needs a section path and matched source text. Refresh the outline and try again."); return;
    }
    setBusy(true); setError(""); setMessage("");
    try {
      const updated = await correctCorpusStagingStructure(token, attempt.attempt_id, attempt.revision, entries);
      setAttempt(updated);
      setMessage("Outline corrections saved for this revision. Review the matches before approving.");
    } catch (cause) {
      setError(safeError(cause, "The outline could not be saved. Refresh and review the current revision."));
      await refresh(attempt.attempt_id);
    } finally { setBusy(false); }
  }

  async function approve() {
    if (!attempt) return;
    setBusy(true); setError(""); setMessage("");
    try {
      const updated = await approveCorpusStagingAttempt(token, attempt.attempt_id, attempt.revision);
      setAttempt(updated);
      setMessage("Approval recorded. Staging progress will update here.");
    } catch (cause) {
      setError(safeError(cause, "The reviewed attempt changed. Refresh and review the current revision before approving."));
      await refresh(attempt.attempt_id);
    } finally { setBusy(false); }
  }

  const approvable = attempt?.state === "awaiting_review" && !!attempt.preview?.length
    && attempt.preview.every((entry) => entry.match_status === "matched" && entry.correction_codes.length === 0)
    && attempt.correction_codes.length === 0
    && attempt.preview.every((entry, index) => (sectionPaths[index] ?? "") === entry.section_path.join(" › "));

  return <section className="admin-card staging-card" aria-labelledby="corpus-staging-title">
    <div className="section-heading"><div><h3 id="corpus-staging-title">Stage a corpus</h3><p>Upload a PDF to generate an editable outline. A staged corpus does not become active until separately released.</p></div></div>
    {error && <div id="staging-error-summary" className="error" role="alert" tabIndex={-1} ref={errorSummary}>
      <p>{error}</p><a href={attempt ? "#staging-refresh" : "#staging-pdf"}>Review staging controls</a>
    </div>}
    {message && <p className="status" role="status">{message}</p>}
    <div className="staging-form">
      <label htmlFor="staging-existing-attempt">Open an existing staging attempt</label>
      <input id="staging-existing-attempt" type="text" value={existingAttemptId} onChange={(event) => setExistingAttemptId(event.target.value)} />
      <button type="button" className="secondary" onClick={() => void loadExistingAttempt()} disabled={busy || !existingAttemptId.trim()}>Open attempt</button>
    </div>
    <form className="staging-form" onSubmit={(event) => void submit(event)}>
      <label htmlFor="staging-pdf">Source PDF</label>
      <input id="staging-pdf" type="file" accept="application/pdf,.pdf" onChange={(event) => setPdf(event.target.files?.[0] ?? null)} />
      <label htmlFor="staging-physical-pages">Physical pages to send for parsing</label>
      <input id="staging-physical-pages" type="text" inputMode="numeric" value={physicalPageSelection} onChange={(event) => setPhysicalPageSelection(event.target.value)} aria-describedby="staging-physical-pages-help" />
      <small id="staging-physical-pages-help">Enter up to five 1-based physical page numbers, separated by commas. The source PDF remains pinned in full.</small>
      <button type="submit" disabled={busy || !pdf}>{busy ? "Working…" : "Upload PDF and parse"}</button>
    </form>
    {attempt && <section className="staging-review" aria-labelledby="staging-review-title">
      <div className="section-heading"><h4 id="staging-review-title">Staging review</h4><button id="staging-refresh" className="secondary" type="button" onClick={() => void refresh()} disabled={busy}>Refresh status</button></div>
      <p>Attempt ID: <code>{attempt.attempt_id}</code></p>
      <p role="status">Status: {(attempt.state === "queued" || attempt.state === "running" || attempt.state === "staged" || attempt.state === "failed" ? attempt.state : attempt.hosted_parsing_state ?? attempt.state).replaceAll("_", " ")}{attempt.completed_stage ? ` · completed ${attempt.completed_stage}` : ""}</p>
      {attempt.hosted_parsing_state === "awaiting_budget" && <div className="status"><p>Hosted parsing is waiting for available budget. No outline has been approved.</p><button type="button" className="secondary" onClick={() => void retryParsing()} disabled={busy}>Retry parsing</button></div>}
      {attempt.hosted_parsing_state === "parsing" && <div className="status"><p>Resume the saved parse job. The PDF will not be uploaded again.</p><button type="button" className="secondary" onClick={() => void retryParsing()} disabled={busy}>Resume parsing</button></div>}
      {attempt.hosted_parsing_state === "failed" && attempt.state !== "failed" && <div className="error"><p>Parsing did not complete. Refresh status or upload the PDF to start a new attempt.</p></div>}
      {attempt.preview && <>
        <h5>Generated outline</h5>
        <ul className="staging-entry-list">{attempt.preview.map((entry, index) => <li key={`${entry.entry_index}-${index}`}>
          <label htmlFor={`staging-section-${index}`}>Section path · pages {entry.page_start}–{entry.page_end}</label>
          <input id={`staging-section-${index}`} aria-label={`Section path for pages ${entry.page_start} to ${entry.page_end}`} value={sectionPaths[index] ?? ""} onChange={(event) => setSectionPaths((current) => current.map((path, pathIndex) => pathIndex === index ? event.target.value : path))} disabled={busy} />
          <span>Match: {entry.match_status}</span>
          {entry.excerpt && <blockquote className="staging-excerpt">{entry.excerpt}</blockquote>}
          {entry.correction_codes.length > 0 && <p className="error">Review needed: {entry.correction_codes.map(correctionText).join("; ")}.</p>}
        </li>)}</ul>
        {attempt.state === "awaiting_review" && <button type="button" className="secondary" onClick={() => void saveOutline()} disabled={busy}>Save outline corrections</button>}
      </>}
      {attempt.correction_codes.length > 0 && <p className="error">{attempt.correction_codes.map(correctionText).join("; ")}. Edit the outline and save corrections.</p>}
      {attempt.state === "failed" && <p className="error">{attempt.error?.message ?? "Staging failed safely. Refresh status or upload the PDF to start a new attempt."}</p>}
      {attempt.state === "failed" && attempt.completed_stage === "ingestion" && attempt.error?.retryable && <button type="button" className="secondary" disabled={busy} onClick={() => void approve()}>Retry ingestion</button>}
      {attempt.state === "staged" && <p className="status">Staged version {attempt.staged_corpus_version_id}; {attempt.parent_count ?? 0} parents and {attempt.chunk_count ?? attempt.child_count ?? 0} chunks. The active corpus is unchanged.</p>}
      <button type="button" disabled={!approvable || busy} onClick={() => void approve()}>Approve outline and create chunks</button>
    </section>}
  </section>;
}

function parsePhysicalPageSelection(value: string): number[] | null {
  const parts = value.split(",").map((part) => part.trim());
  if (parts.length < 1 || parts.length > 5 || parts.some((part) => !/^\d+$/.test(part))) return null;
  const pages = parts.map(Number);
  if (pages.some((page) => page < 1) || new Set(pages).size !== pages.length) return null;
  return pages.sort((left, right) => left - right);
}

function safeError(cause: unknown, fallback = "The staging request could not be completed. Try again shortly."): string {
  if (typeof cause === "object" && cause !== null && "status" in cause && cause.status === 409) {
    if ("code" in cause && cause.code === "stale_staging_attempt") return "This reviewed revision is stale. Refresh and review the current outline before saving or approving.";
    if ("code" in cause && cause.code === "staging_attempt_not_parseable") return "This parse attempt cannot be resumed. Refresh status or upload the PDF to start a new attempt.";
  }
  if (cause instanceof Error && cause.message && cause.message.length < 200) return cause.message;
  return fallback;
}

function correctionText(code: string): string {
  const messages: Record<string, string> = {
    invalid_structure: "the generated structure needs correction", source_pair_mismatch: "the source does not match this review", missing_page_coverage: "some pages need section coverage",
    ambiguous_heading: "a heading matches more than one section", unmatched_passage: "some source text has no matching section", stale_revision: "the reviewed revision is stale",
    job_interrupted: "staging was interrupted", job_failed: "staging did not complete",
  };
  return messages[code] ?? "the entry needs correction";
}
