import { useEffect, useId, useState, type ChangeEvent, type FormEvent } from "react";
import type { ActiveEvaluationGeneration, BatchAttestationDecision, BatchReviewStatus, EvaluationDatasetRevision, EvaluationItemResult, EvaluationMetric, EvaluationReviewProjection, EvaluationRun, ExceptionDecisionKind, ExplicitExceptionDecision, ImportValidationError, IndividualEvidenceReview, QuestionCategory, RunItemAssessmentProjection } from "../../api/contracts";
import { addEvaluationDatasetRow, assessRunItem, attestBatchReview, decideIndividualReview, EvaluationApiError, getActiveDataset, getEvaluationDatasetRevision, getEvaluationRun, getIndividualReview, getReviewSummary, getRunItemAssessmentContext, launchActiveEvaluation, removeEvaluationDatasetRow, uploadAndRunEvaluation, type EvaluationUploadRunResponse } from "./api";

type EvaluationPanelProps = { token: string };
type EvaluationStatus = Pick<EvaluationRun, "run_id" | "status" | "metrics" | "item_results" | "error">;
type SemanticDecision = { claim_support: boolean; citation_support: boolean; rubric_passed: boolean; refusal_judgment: boolean };

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
  const [exceptionDecisions, setExceptionDecisions] = useState<Record<string, { decision: ExceptionDecisionKind | ""; reason: string }>>({});
  const [activeDataset, setActiveDataset] = useState<ActiveEvaluationGeneration | null>(null);
  const [activeDatasetError, setActiveDatasetError] = useState("");
  const [datasetRevision, setDatasetRevision] = useState<EvaluationDatasetRevision | null>(null);
  const [datasetRevisionError, setDatasetRevisionError] = useState("");
  const [datasetBusy, setDatasetBusy] = useState(false);
  const [datasetMessage, setDatasetMessage] = useState("");
  const [rowQuestion, setRowQuestion] = useState("");
  const [rowCategory, setRowCategory] = useState<QuestionCategory>("definitions");
  const [rowOutcome, setRowOutcome] = useState<"answer" | "qualified" | "evidence_refusal">("answer");
  const [rowRubric, setRowRubric] = useState("");
  const [rowSupport, setRowSupport] = useState("");
  const [rowScope, setRowScope] = useState("");
  const [selectedItems, setSelectedItems] = useState<string[]>([]);
  const [individualReviews, setIndividualReviews] = useState<Record<string, IndividualEvidenceReview>>({});
  const [individualErrors, setIndividualErrors] = useState<Record<string, string>>({});
  const [individualError, setIndividualError] = useState("");
  const [individualBusy, setIndividualBusy] = useState<string | null>(null);
  const [assessmentContexts, setAssessmentContexts] = useState<Record<string, RunItemAssessmentProjection>>({});
  const [assessmentDecisions, setAssessmentDecisions] = useState<Record<string, Partial<SemanticDecision>>>({});
  const [assessmentErrors, setAssessmentErrors] = useState<Record<string, string>>({});
  const [assessmentBusy, setAssessmentBusy] = useState<string | null>(null);
  const [refusalConfirmations, setRefusalConfirmations] = useState<Record<string, boolean>>({});

  useEffect(() => {
    let current = true;
    void getActiveDataset(token).then((dataset) => { if (current) setActiveDataset(dataset); }).catch((cause: unknown) => {
      if (!current) return;
      if (cause instanceof EvaluationApiError && cause.status === 404) setActiveDatasetError("No active dataset is available.");
      else setActiveDatasetError(cause instanceof EvaluationApiError && [401, 403].includes(cause.status) ? "You do not have administrator access to view the active dataset." : "The active dataset could not be loaded.");
    });
    void getEvaluationDatasetRevision(token).then((revision) => { if (current) { setDatasetRevision(revision); setDatasetRevisionError(""); } }).catch((cause: unknown) => {
      if (!current) return;
      setDatasetRevisionError(cause instanceof EvaluationApiError && cause.status === 404 ? "No versioned evaluation dataset is available." : "The versioned evaluation dataset could not be loaded.");
    });
    return () => { current = false; };
  }, [token]);

  useEffect(() => {
    if (!runId || !run || !["queued", "running"].includes(run.status)) return;
    const timer = window.setTimeout(() => { void getEvaluationRun(token, runId).then(setRun).catch(showError); }, 500);
    return () => window.clearTimeout(timer);
  }, [run, runId, token]);

  useEffect(() => {
    setAssessmentContexts({});
    setAssessmentDecisions({});
    setAssessmentErrors({});
    setAssessmentBusy(null);
    setRefusalConfirmations({});
  }, [runId]);

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

  useEffect(() => {
    if (!review || !runId || run?.status !== "completed") { setAssessmentContexts({}); setAssessmentErrors({}); return; }
    let current = true;
    const itemIds = [...new Set([
      ...review.sample.map((item) => item.item_id),
      ...review.exceptions.filter((item) => item.expected_refusal).map((item) => item.item_id),
    ])];
    setAssessmentContexts({}); setAssessmentErrors({}); setRefusalConfirmations({});
    void Promise.allSettled(itemIds.map((itemId) => getRunItemAssessmentContext(token, runId, itemId))).then((results) => {
      if (!current) return;
      const contexts: Record<string, RunItemAssessmentProjection> = {};
      const errors: Record<string, string> = {};
      results.forEach((result, index) => {
        const itemId = itemIds[index];
        if (result.status === "fulfilled") contexts[itemId] = result.value;
        else errors[itemId] = safeReviewError(result.reason);
      });
      setAssessmentContexts(contexts); setAssessmentErrors(errors);
      setAssessmentDecisions((existing) => {
        const next = { ...existing };
        for (const [itemId, context] of Object.entries(contexts)) if (!next[itemId] && context.assessment) {
          next[itemId] = { claim_support: context.assessment.claim_support ?? undefined, citation_support: context.assessment.citation_support ?? undefined, rubric_passed: context.assessment.rubric_passed ?? undefined, refusal_judgment: context.assessment.refusal_judgment ?? undefined };
        }
        return next;
      });
    });
    return () => { current = false; };
  }, [review, run?.status, runId, token]);

  useEffect(() => {
    if (!runId || run?.status !== "completed" || !run.item_results.length) { setIndividualReviews({}); setIndividualErrors({}); return; }
    let current = true;
    setIndividualReviews({}); setIndividualErrors({});
    void Promise.allSettled(run.item_results.map((item) => getIndividualReview(token, runId, item.item_id)))
      .then((results) => {
        if (!current) return;
        const reviews: Record<string, IndividualEvidenceReview> = {};
        const errors: Record<string, string> = {};
        results.forEach((result, index) => {
          const itemId = run.item_results[index].item_id;
          if (result.status === "fulfilled") reviews[itemId] = result.value;
          else errors[itemId] = safeReviewError(result.reason);
        });
        setIndividualReviews(reviews); setIndividualErrors(errors);
      });
    return () => { current = false; };
  }, [run?.status, run?.item_results, runId, token]);

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

  async function runActiveDataset(itemIds?: string[]) {
    setBusy(true); setError(""); setMessage(""); setRun(null); setRunId(""); setReview(null); setIndividualReviews({}); setIndividualErrors({}); setIndividualError("");
    try {
      const result = await launchActiveEvaluation(token, itemIds);
      setRunId(result.run_id); setRun(result);
      setMessage(itemIds ? `Started evaluation for ${itemIds.length} selected items.` : "Started evaluation for the full active dataset.");
    } catch (cause) { showError(cause); } finally { setBusy(false); }
  }

  async function refreshDatasetRevision() {
    try {
      const current = await getEvaluationDatasetRevision(token);
      setDatasetRevision(current);
      setSelectedItems((selected) => selected.filter((itemId) => current.rows.some((row) => row.item_id === itemId)));
      setDatasetRevisionError("");
    } catch {
      setDatasetRevisionError("The versioned evaluation dataset could not be refreshed.");
    }
  }

  async function saveDatasetRow(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!datasetRevision) return;
    setDatasetBusy(true); setDatasetMessage(""); setDatasetRevisionError("");
    try {
      await addEvaluationDatasetRow(token, {
        expected_revision: datasetRevision.revision,
        operation_key: crypto.randomUUID(),
        question: rowQuestion.trim(),
        category: rowCategory,
        expected_outcome: rowOutcome,
        expected_answer_or_rubric: rowRubric.trim(),
        supporting_passage: rowOutcome === "evidence_refusal" ? null : rowSupport.trim(),
        scope_rationale: rowOutcome === "evidence_refusal" ? rowScope.trim() : null,
      });
      await refreshDatasetRevision();
      setDatasetMessage("Dataset row added to a new revision.");
      setRowQuestion(""); setRowRubric(""); setRowSupport(""); setRowScope("");
    } catch (cause) {
      if (cause instanceof EvaluationApiError && cause.status === 409) {
        await refreshDatasetRevision();
        setDatasetRevisionError("The dataset changed while you were editing. The current revision is loaded; review it and submit your edit again.");
      } else setDatasetRevisionError(cause instanceof Error ? cause.message : "The dataset row could not be added.");
    } finally { setDatasetBusy(false); }
  }

  async function removeDatasetRow(itemId: string) {
    if (!datasetRevision || !window.confirm("Remove this question from the active dataset?")) return;
    setDatasetBusy(true); setDatasetMessage(""); setDatasetRevisionError("");
    try {
      await removeEvaluationDatasetRow(token, itemId, { expected_revision: datasetRevision.revision, operation_key: crypto.randomUUID() });
      await refreshDatasetRevision();
      setSelectedItems((selected) => selected.filter((id) => id !== itemId));
      setDatasetMessage("Dataset row removed in a new revision.");
    } catch (cause) {
      if (cause instanceof EvaluationApiError && cause.status === 409) {
        await refreshDatasetRevision();
        setDatasetRevisionError("The dataset changed while you were editing. The current revision is loaded; review it before removing a row.");
      } else setDatasetRevisionError(cause instanceof Error ? cause.message : "The dataset row could not be removed.");
    } finally { setDatasetBusy(false); }
  }

  async function saveSemanticAssessment(itemId: string) {
    const context = assessmentContexts[itemId];
    const decision = assessmentDecisions[itemId];
    if (!context || !runId || !decision || [decision.claim_support, decision.citation_support, decision.rubric_passed, decision.refusal_judgment].some((value) => value === undefined)) return;
    setAssessmentBusy(itemId); setAssessmentErrors((current) => ({ ...current, [itemId]: "" }));
    try {
      const saved = await assessRunItem(token, runId, itemId, {
        generation_id: context.generation_id,
        attempt_id: context.attempt_id,
        candidate_id: context.candidate_id,
        scorer_version: context.scorer_version,
        output_digest: context.output_digest,
        claim_support: decision.claim_support as boolean,
        citation_support: decision.citation_support as boolean,
        rubric_passed: decision.rubric_passed as boolean,
        refusal_judgment: decision.refusal_judgment as boolean,
      });
      setAssessmentContexts((current) => ({ ...current, [itemId]: { ...context, assessment: { assessment_id: saved.assessment_id, claim_support: saved.claim_support, citation_support: saved.citation_support, rubric_passed: saved.rubric_passed, refusal_judgment: saved.refusal_judgment, reviewed_at: saved.reviewed_at } } }));
    } catch (cause) { setAssessmentErrors((current) => ({ ...current, [itemId]: safeReviewError(cause) })); }
    finally { setAssessmentBusy(null); }
  }

  async function reviewItem(itemId: string, decision: "approve" | "reject") {
    if (!runId) return;
    setIndividualBusy(itemId); setIndividualError("");
    try {
      const result = await decideIndividualReview(token, runId, itemId, decision);
      setIndividualReviews((current) => ({ ...current, [itemId]: { ...current[itemId], status: result.status, reviewer_actor: result.reviewer_actor, reviewed_at: result.reviewed_at } }));
    } catch (cause) { setIndividualError(safeReviewError(cause)); }
    finally { setIndividualBusy(null); }
  }

  async function submitBatchDecision(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!runId || !review || !reviewDecision || !exceptionsReady) return;
    const decisions: ExplicitExceptionDecision[] = review.exceptions.map((item) => {
      const selected = exceptionDecisions[item.item_id];
      return { item_id: item.item_id, decision: selected?.decision as ExceptionDecisionKind, reason: selected?.decision === "attested" ? null : selected?.reason.trim() || null };
    });
    const semanticItemIds = [...new Set([
      ...review.sample.map((item) => item.item_id),
      ...review.exceptions.filter((item) => item.expected_refusal).map((item) => item.item_id),
    ])];
    const semanticAssessments = semanticItemIds.flatMap((itemId) => {
      const assessment = assessmentContexts[itemId]?.assessment;
      return assessment ? [{ item_id: itemId, assessment_id: assessment.assessment_id }] : [];
    });
    const refusalConfirmationItemIds = review.exceptions.filter((item) => item.expected_refusal).map((item) => item.item_id);
    setReviewBusy(true); setReviewError(""); setReviewMessage("");
    try {
      await attestBatchReview(token, runId, {
        decision: reviewDecision,
        exception_decisions: decisions,
        semantic_assessments: reviewDecision === "attested" ? semanticAssessments : [],
        refusal_confirmation_item_ids: reviewDecision === "attested" ? refusalConfirmationItemIds : [],
        reason: reviewDecision === "rejected" ? reviewReason.trim() : null,
      });
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
    return selected?.decision && (selected.decision === "attested" ? !!item.expected_refusal && !!refusalConfirmations[item.item_id] : !!selected.reason.trim());
  }) ?? false;
  const semanticItemIds = review ? [...new Set([
    ...review.sample.map((item) => item.item_id),
    ...review.exceptions.filter((item) => item.expected_refusal).map((item) => item.item_id),
  ])] : [];
  const semanticAssessmentsReady = reviewDecision !== "attested" || semanticItemIds.every((itemId) => {
    const assessment = assessmentContexts[itemId]?.assessment;
    return !!assessment && [assessment.claim_support, assessment.citation_support, assessment.rubric_passed, assessment.refusal_judgment].every((value) => value !== null)
      && (!review?.exceptions.some((item) => item.item_id === itemId && item.expected_refusal) || assessment.refusal_judgment === true);
  });
  const refusalConfirmationsReady = reviewDecision !== "attested" || (review?.exceptions.filter((item) => item.expected_refusal).every((item) => refusalConfirmations[item.item_id]) ?? false);
  const reviewOpen = !!review && ["eligible", "exception"].includes(review.summary.status) && !review.summary.decision;
  const decisionReady = reviewOpen && !!reviewDecision && (reviewDecision !== "rejected" || !!reviewReason.trim()) && exceptionsReady && semanticAssessmentsReady && refusalConfirmationsReady;

  function refreshReview() {
    setReviewDecision(""); setReviewReason(""); setExceptionDecisions({}); setReviewRefresh((value) => value + 1);
  }

  return <section className="admin-card evaluation-panel" aria-labelledby="evaluation-title">
    <div className="section-heading"><div><h3 id="evaluation-title">Evaluation</h3><p>Run the active evaluation dataset or upload a JSON evaluation set.</p></div>{run && <span className={`state state-${run.status}`}>{run.status}</span>}</div>
    {error && <p className="error" role="alert">{error}</p>}{message && <p className="status" role="status">{message}</p>}
    <section aria-labelledby="active-dataset-title"><h4 id="active-dataset-title">Active dataset</h4>
      {activeDatasetError && <p className="error" role="note">{activeDatasetError}</p>}
      {datasetRevisionError && <p className="error" role="alert">{datasetRevisionError}</p>}
      {datasetMessage && <p className="status" role="status">{datasetMessage}</p>}
      {datasetRevision ? <><p>Revision {datasetRevision.revision} · {datasetRevision.rows.length} questions</p>
        {datasetRevision.rows.map((item, index) => {
          const reviewStatus = activeDataset?.item_reviews.find((reviewItem) => reviewItem.item_id === item.item_id)?.review_status;
          return <article className="review-choice" key={item.item_id}>
            <label><input type="checkbox" aria-label={`Select dataset question ${index + 1}`} checked={selectedItems.includes(item.item_id)} disabled={busy || datasetBusy} onChange={(event) => setSelectedItems((current) => event.target.checked ? [...current, item.item_id] : current.filter((id) => id !== item.item_id))} />{item.question}</label>
            <p>{item.category} · {item.expected_outcome.replaceAll("_", " ")}{reviewStatus ? ` · ${reviewStatus}` : ""}</p>
            <p>{item.expected_answer_or_rubric}</p>
            {item.scope_rationale && <p>Refusal scope: {item.scope_rationale}</p>}
            <button type="button" disabled={busy || datasetBusy || datasetRevision.rows.length === 1} onClick={() => void removeDatasetRow(item.item_id)}>Remove question</button>
          </article>;
        })}
        <button type="button" disabled={busy || datasetBusy} onClick={() => void runActiveDataset()}>{busy ? "Starting…" : "Run full active dataset"}</button>
        <button type="button" disabled={busy || datasetBusy || selectedItems.length === 0} onClick={() => void runActiveDataset(selectedItems)}>{busy ? "Starting…" : `Run selected items${selectedItems.length ? ` (${selectedItems.length})` : ""}`}</button>
        <details><summary>Add a question</summary><form onSubmit={(event) => void saveDatasetRow(event)}>
          <label htmlFor="dataset-question">Question<input id="dataset-question" value={rowQuestion} onChange={(event) => setRowQuestion(event.target.value)} required maxLength={2000} /></label>
          <label htmlFor="dataset-category">Category<select id="dataset-category" value={rowCategory} onChange={(event) => setRowCategory(event.target.value as QuestionCategory)}><option value="definitions">Definitions</option><option value="explanations">Explanations</option><option value="comparisons">Comparisons</option><option value="study">Study</option><option value="scenarios">Scenarios</option></select></label>
          <label htmlFor="dataset-outcome">Expected outcome<select id="dataset-outcome" value={rowOutcome} onChange={(event) => setRowOutcome(event.target.value as typeof rowOutcome)}><option value="answer">Answer</option><option value="qualified">Qualified answer</option><option value="evidence_refusal">Evidence refusal</option></select></label>
          <label htmlFor="dataset-rubric">Expected answer or rubric<textarea id="dataset-rubric" value={rowRubric} onChange={(event) => setRowRubric(event.target.value)} required maxLength={10000} /></label>
          {rowOutcome === "evidence_refusal" ? <label htmlFor="dataset-scope">Why this question is outside the corpus<textarea id="dataset-scope" value={rowScope} onChange={(event) => setRowScope(event.target.value)} required maxLength={5000} /></label> : <label htmlFor="dataset-support">Supporting passage<textarea id="dataset-support" value={rowSupport} onChange={(event) => setRowSupport(event.target.value)} required maxLength={20000} /></label>}
          <button type="submit" disabled={datasetBusy}>{datasetBusy ? "Saving…" : "Add question"}</button>
        </form></details>
      </> : <p>{datasetRevisionError ? "Refresh the dataset to try again." : "Loading versioned dataset…"}</p>}
    </section>
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
    {individualError && <p className="error" role="alert">{individualError}</p>}
    {run && <RunStatus run={run} reviews={individualReviews} reviewErrors={individualErrors} busyItem={individualBusy} onReview={reviewItem} />}
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
        {review.sample.length === 0 ? <p>No representative samples are available.</p> : <ul className="review-list">{review.sample.map((item, index) => <li key={`${item.page}-${item.section}-${index}`}><p><strong>{item.question}</strong></p><p>{item.rubric}</p><blockquote>{item.support_excerpt}</blockquote><p>Page {item.page} · {item.section} · {reviewStatusLabel(item.status)}</p><SemanticAssessmentReview context={assessmentContexts[item.item_id]} values={assessmentDecisions[item.item_id] ?? {}} error={assessmentErrors[item.item_id]} busy={assessmentBusy === item.item_id} onChange={(key, value) => setAssessmentDecisions((current) => ({ ...current, [item.item_id]: { ...current[item.item_id], [key]: value } }))} onSave={() => void saveSemanticAssessment(item.item_id)} /></li>)}</ul>}
        <h5>Exceptions requiring a decision</h5>
        {review.exceptions.length === 0 && <p>No exceptions require a decision.</p>}
        <form onSubmit={submitBatchDecision} aria-describedby="exception-guidance">
          <p id="exception-guidance">Choose a decision for every exception. Record identity stays hidden; decisions are linked automatically.</p>
          {review.exceptions.length > 0 && <ul className="review-list">{review.exceptions.map((item, index) => <li key={`${item.page ?? ""}-${item.section ?? ""}-${index}`}>
            <p><strong>{item.question}</strong></p><p>{item.rubric}</p>{item.support_excerpt && <blockquote>{item.support_excerpt}</blockquote>}
            <p>{item.page != null ? `Page ${item.page} · ` : ""}{item.section ?? "Source location unavailable"} · {item.status}: {item.reason}</p>
            {item.expected_refusal && <><label className="review-choice"><input type="checkbox" checked={!!refusalConfirmations[item.item_id]} disabled={reviewBusy || !!reviewError || !reviewOpen} onChange={(event) => setRefusalConfirmations((current) => ({ ...current, [item.item_id]: event.target.checked }))} />I confirm this question is outside the corpus and should be answered with a refusal.</label><SemanticAssessmentReview context={assessmentContexts[item.item_id]} values={assessmentDecisions[item.item_id] ?? {}} error={assessmentErrors[item.item_id]} busy={assessmentBusy === item.item_id} onChange={(key, value) => setAssessmentDecisions((current) => ({ ...current, [item.item_id]: { ...current[item.item_id], [key]: value } }))} onSave={() => void saveSemanticAssessment(item.item_id)} /></>}
            <label htmlFor={`exception-decision-${index}`}>Decision<select id={`exception-decision-${index}`} value={exceptionDecisions[item.item_id]?.decision ?? ""} disabled={reviewBusy || !!reviewError || !reviewOpen} onChange={(event) => setExceptionDecisions((current) => ({ ...current, [item.item_id]: { decision: event.target.value as ExceptionDecisionKind | "", reason: current[item.item_id]?.reason ?? "" } }))}><option value="">Choose a decision</option>{item.expected_refusal && <option value="attested">Confirm refusal case</option>}<option value="rejected">Reject exception</option><option value="inapplicable">Mark inapplicable</option></select></label>
            {exceptionDecisions[item.item_id]?.decision && exceptionDecisions[item.item_id]?.decision !== "attested" && <label htmlFor={`exception-reason-${index}`}>Exception reason<textarea id={`exception-reason-${index}`} value={exceptionDecisions[item.item_id]?.reason ?? ""} disabled={reviewBusy || !!reviewError || !reviewOpen} onChange={(event) => setExceptionDecisions((current) => ({ ...current, [item.item_id]: { decision: current[item.item_id]?.decision ?? "", reason: event.target.value } }))} /></label>}
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

type SemanticDecisionKey = keyof SemanticDecision;
const semanticDecisionLabels: Record<SemanticDecisionKey, string> = {
  claim_support: "Claims supported by evidence",
  citation_support: "Citations support the claims",
  rubric_passed: "Response meets the rubric",
  refusal_judgment: "Refusal judgment is correct",
};

function SemanticAssessmentReview({ context, values, error, busy, onChange, onSave }: {
  context?: RunItemAssessmentProjection;
  values: Partial<SemanticDecision>;
  error?: string;
  busy: boolean;
  onChange: (key: SemanticDecisionKey, value: boolean) => void;
  onSave: () => void;
}) {
  const formId = useId();
  const complete = [values.claim_support, values.citation_support, values.rubric_passed, values.refusal_judgment].every((value) => value !== undefined);
  return <section aria-label="Exact output assessment">
    <h6>Review generated answer</h6>
    {context ? <><blockquote>{context.output_text}</blockquote>
      {context.expected_refusal && <p>This item expects an evidence refusal.</p>}
      {Object.entries(semanticDecisionLabels).map(([key, label]) => {
        const field = key as SemanticDecisionKey;
        return <fieldset key={field}><legend>{label}</legend>
          <label><input type="radio" name={`assessment-${formId}-${field}`} checked={values[field] === true} onChange={() => onChange(field, true)} />Yes</label>
          <label><input type="radio" name={`assessment-${formId}-${field}`} checked={values[field] === false} onChange={() => onChange(field, false)} />No</label>
        </fieldset>;
      })}
      {context.assessment && <p role="status">Exact output assessment saved.</p>}
      <button type="button" onClick={onSave} disabled={busy || !complete}>{busy ? "Saving assessment…" : context.assessment ? "Update exact output assessment" : "Save exact output assessment"}</button>
    </> : <p role="status">Loading exact run output for review…</p>}
    {error && <p role="alert">{error}</p>}
  </section>;
}

function safeReviewError(cause: unknown) {
  return cause instanceof EvaluationApiError && [401, 403].includes(cause.status)
    ? "You do not have administrator access to review this item."
    : cause instanceof EvaluationApiError && cause.status === 404
      ? "This item's review is unavailable for this run."
      : cause instanceof EvaluationApiError && cause.status === 409
        ? "This item review is stale or already decided. Refresh the run before trying again."
      : cause instanceof Error ? cause.message : "The item decision could not be saved.";
}

function RunStatus({ run, reviews, reviewErrors, busyItem, onReview }: { run: EvaluationStatus; reviews: Record<string, IndividualEvidenceReview>; reviewErrors: Record<string, string>; busyItem: string | null; onReview: (itemId: string, decision: "approve" | "reject") => void }) {
  if (run.status === "failed") return <p className="error" role="alert">{run.error ?? "The evaluation failed safely. No results are available."}</p>;
  if (run.status !== "completed") return <p role="status">Evaluation is processing…</p>;
  return <>{run.item_results.length === 0 ? <p role="status">No questions evaluated.</p> : <section aria-labelledby="items-title"><h4 id="items-title">Question results</h4><ul>{run.item_results.map((item) => <ItemResult key={item.item_id} item={item} review={reviews[item.item_id]} reviewError={reviewErrors[item.item_id]} busy={busyItem === item.item_id} onReview={onReview} />)}</ul></section>}{run.metrics.length > 0 && <section aria-labelledby="metrics-title"><h4 id="metrics-title">Aggregate results</h4><ul className="gate-list">{run.metrics.map((metric) => <MetricResult key={metric.metric_name} metric={metric} />)}</ul></section>}</>;
}

function ItemResult({ item, review, reviewError, busy, onReview }: { item: EvaluationItemResult; review?: IndividualEvidenceReview; reviewError?: string; busy: boolean; onReview: (itemId: string, decision: "approve" | "reject") => void }) { const decided = review?.status === "review_approved" || review?.status === "review_rejected"; const canApprove = review?.status === "eligible" || (review?.status === "inapplicable" && review.expected_refusal); return <li><strong>Item review: {review ? review.status.replaceAll("_", " ") : reviewError ? "unavailable" : "loading"}</strong>{reviewError && <p role="note">{reviewError}</p>}<span>Retrieval {item.retrieval_success ? "pass" : "fail"}; grounded claims {item.grounded ? "pass" : "fail"}; citations {item.citation_correct ? "pass" : "fail"}; answer quality {item.answer_quality ? "pass" : "fail"}; refusal {item.refusal_correct ? "pass" : "fail"}; {item.response_time_ms} ms</span>{review && <><p>{review.question}</p>{review.support_excerpt && <blockquote>{review.support_excerpt}</blockquote>}<p>{review.page != null ? `Page ${review.page} · ` : ""}{review.section ?? "Source location unavailable"} · proposal status: {review.status.replaceAll("_", " ")}</p><button type="button" onClick={() => onReview(item.item_id, "approve")} disabled={busy || decided || !canApprove}>Approve proposal</button><button type="button" onClick={() => onReview(item.item_id, "reject")} disabled={busy || decided}>Reject proposal</button></>}</li>; }

function MetricResult({ metric }: { metric: EvaluationMetric }) { return <li className={metric.passed ? "gate-passed" : "gate-failed"}><strong>{metricLabels[metric.metric_name] ?? metric.metric_name}</strong><span>{metric.numerator} / {metric.denominator} · {metric.percentage}% · {metric.passed ? "Pass" : "Fail"}</span></li>; }
