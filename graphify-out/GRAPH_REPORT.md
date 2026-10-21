# Graph Report - dmbok-compass  (2026-10-10)

## Corpus Check
- 146 files · ~166,967 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 9 file(s) not represented in the graph (top: (none) 3, .example 2, .cache 1)

## Summary
- 1119 nodes · 2973 edges · 57 communities (42 shown, 15 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 232 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Community 45
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 53
- Community 56

## God Nodes (most connected - your core abstractions)
1. `UserProfile` - 56 edges
2. `GenerationError` - 45 edges
3. `CorpusVersion` - 34 edges
4. `GenerationRequest` - 34 edges
5. `OpenRouterAdapter` - 32 edges
6. `GenerationErrorCode` - 28 edges
7. `ContractModel` - 27 edges
8. `QuotaStatus` - 27 edges
9. `GroundedAnswerPolicy` - 26 edges
10. `IdentityError` - 26 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `validate_release_evidence()`  [INFERRED]
  scripts/verify_release_evidence.py → backend/app/release/evidence.py
- `probe_user()` --uses--> `RequestSample`  [INFERRED]
  scripts/run_performance_probe.py → backend/app/performance.py
- `validated_citations()` --uses--> `GenerationCandidate`  [INFERRED]
  backend/app/answering/citations.py → backend/model/protocol.py
- `GroundedAnswerPolicy` --uses--> `GenerationError`  [INFERRED]
  backend/app/answering/pipeline.py → backend/model/protocol.py
- `GroundedAnswerPolicy` --uses--> `GenerationRequest`  [INFERRED]
  backend/app/answering/pipeline.py → backend/model/protocol.py

## Import Cycles
- None detected.

## Communities (57 total, 15 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (40): evaluation_service(), get_evaluation_status(), launch_evaluation(), EvaluationDataset, EvaluationItem, GoldAnnotation, EvaluationDatasetError, EvaluationDatasetStore (+32 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (27): record_release_decision(), InMemoryReleaseDecisionStore, ReleaseDecisionError, ReleaseDecisionRequest, ReleaseDecisionService, ReleaseDecisionStore, ReleaseEvidenceError, ReleaseEvidenceResult (+19 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (23): ActiveCorpusStore, classify_evidence(), _elapsed_ms(), EvidenceBundle, GroundedAnswerPolicy, PassageRetriever, QuestionAnswerer, RetrievedAnswerService (+15 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (23): QuotaPolicy, get_quota_repository(), quota_repository(), reserve_request_quota(), update_configuration(), FirestoreQuotaRepository, reserve_in_transaction(), InMemoryQuotaRepository (+15 more)

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (21): configured_question_answerer(), OpenRouterAdapter, TimeoutBudget, request(), response_payload(), test_openrouter_allows_fallback_after_primary_timeout(), handler(), test_openrouter_does_not_fallback_on_authentication_failure() (+13 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (10): CorpusVersion, FirestoreCorpusVersionStore, CorpusCompletenessError, CorpusVersionError, CorpusVersionStore, InMemoryCorpusVersionStore, chunk(), test_activation_is_idempotent_and_rollback_restores_previous_pointer() (+2 more)

### Community 6 - "Community 6"
Cohesion: 0.13
Nodes (18): UserProfile, write_profile(), get_admin_user(), get_approved_user(), IdentityPrincipal, require_admin(), require_approved_user(), list_users() (+10 more)

### Community 7 - "Community 7"
Cohesion: 0.13
Nodes (14): build_report(), forecast_provider_cost_eur(), latency_summary(), percentile(), RequestSample, json_safe_text(), sample(), test_cost_forecast_blends_primary_and_fallback_rates() (+6 more)

### Community 8 - "Community 8"
Cohesion: 0.11
Nodes (11): TextChunk, VertexDocumentEmbedder, CorpusEmbeddingWriter, CorpusIngestionJob, IngestionResult, FakeClient, FakeCollection, FakeDocument (+3 more)

### Community 9 - "Community 9"
Cohesion: 0.14
Nodes (13): RetrievalTrace, TelemetryEvent, _elapsed_ms(), error_event(), InMemoryTelemetryAdapter, LoggingTelemetryAdapter, response_event(), answer() (+5 more)

### Community 10 - "Community 10"
Cohesion: 0.12
Nodes (9): extract_pdf(), ExtractionReport, _normalise_text(), PageProvenance, _recover_section(), _section_heading(), _materialise_fixture(), test_extracts_one_based_pages_and_carries_section_provenance() (+1 more)

### Community 11 - "Community 11"
Cohesion: 0.14
Nodes (12): EvaluationLaunchRequest, ApiError, Citation, ContractModel, DocumentChunk, EvaluationItemResult, EvaluationMetric, EvaluationRun (+4 more)

### Community 12 - "Community 12"
Cohesion: 0.14
Nodes (8): AggregateMetric, list_aggregate_metrics(), AggregateMetricRepository, FirestoreAggregateMetricRepository, InMemoryAggregateMetricRepository, metric(), test_admin_metrics_endpoint_returns_only_aggregate_contract(), test_in_memory_metrics_are_sorted_and_content_free()

### Community 13 - "Community 13"
Cohesion: 0.14
Nodes (8): create_vertex_document_embedder(), create_source_manifest(), SourceManifest, main(), _required(), test_manifest_binds_version_to_immutable_source_generation(), test_manifest_rejects_unapproved_source_metadata(), test_replacing_source_object_creates_a_new_corpus_version()

### Community 14 - "Community 14"
Cohesion: 0.13
Nodes (16): ApprovalUpdate, RegistrationRequest, RegistrationStatus, aggregate_metric_repository(), current_quota(), current_user(), get_aggregate_metric_repository(), get_configuration() (+8 more)

### Community 15 - "Community 15"
Cohesion: 0.08
Nodes (21): jsdom, react, react-dom, @testing-library/jest-dom, @types/react, @types/react-dom, typescript, vite (+13 more)

### Community 16 - "Community 16"
Cohesion: 0.19
Nodes (21): actions, AdminPanel(), changeUser(), load(), savePolicy(), AdminPanelProps, filters, isForbidden() (+13 more)

### Community 17 - "Community 17"
Cohesion: 0.17
Nodes (8): ModelAttempt, _adapter_name(), FallbackAdapter, _render_contents(), GenerationRequest, GenerationResult, ModelAdapter, test_cross_provider_fallback_preserves_attempts()

### Community 18 - "Community 18"
Cohesion: 0.14
Nodes (11): classify_evidence(), EvidenceBundle, retrieval_success_metric(), RetrievalCase, _validate_thresholds(), passage(), test_classifies_evidence_at_explicit_boundaries(), test_filters_invalid_scores_sorts_ties_deterministically_and_caps_at_five() (+3 more)

### Community 19 - "Community 19"
Cohesion: 0.22
Nodes (11): _AsyncModels, GeminiAdapter, FakeClient, FakeModels, request(), response(), test_gemini_adapter_maps_structured_output_and_usage(), test_invalid_provider_json_is_not_retried() (+3 more)

### Community 20 - "Community 20"
Cohesion: 0.21
Nodes (10): FakeAdapter, _normalize_gemini_error(), GenerationError, GenerationErrorCode, parse_candidate(), request(), test_fake_adapter_enforces_timeout_budget(), test_fake_adapter_normalizes_configured_failure() (+2 more)

### Community 21 - "Community 21"
Cohesion: 0.17
Nodes (19): EvaluationApiError, EvaluationLaunch, getEvaluationRun(), launchEvaluation(), recordReleaseDecision(), ReleaseDecisionInput, request(), EvaluationPanel() (+11 more)

### Community 22 - "Community 22"
Cohesion: 0.17
Nodes (17): firebase, getCurrentProfile(), IdentityApiError, identityRequest(), registerProfile(), authError(), AuthPanel(), logout() (+9 more)

### Community 23 - "Community 23"
Cohesion: 0.13
Nodes (13): @testing-library/react, @testing-library/user-event, @vitejs/plugin-react, vitest, api, completedRun, adminApi, approvedUser (+5 more)

### Community 24 - "Community 24"
Cohesion: 0.15
Nodes (8): FirestoreChunkRepository, FakeClient, FakeCollection, FakeDocument, FakeEmbeddingClient, FakeEmbeddingModels, test_chunk_upsert_is_deterministic_and_corpus_version_scoped(), test_embedding_writer_persists_the_embedding_with_the_chunk()

### Community 25 - "Community 25"
Cohesion: 0.11
Nodes (18): compilerOptions, allowJs, allowSyntheticDefaultImports, esModuleInterop, isolatedModules, jsx, lib, module (+10 more)

### Community 27 - "Community 27"
Cohesion: 0.29
Nodes (9): chunk_pages(), _normalized_tokens(), SourcePage, _validate_page(), test_chunking_never_crosses_pages_and_preserves_provenance(), test_chunking_normalizes_whitespace_and_is_repeatable(), test_chunking_rejects_invalid_configuration(), test_chunking_rejects_invalid_page_provenance() (+1 more)

### Community 28 - "Community 28"
Cohesion: 0.15
Nodes (4): create_vertex_query_embedder(), _embedding_values(), EmbeddingClient, QueryEmbedder

### Community 29 - "Community 29"
Cohesion: 0.21
Nodes (6): create_gemini_adapter(), _response_payload(), _usage_metadata(), _usage_value(), GenerationCandidate, UsageMetadata

### Community 30 - "Community 30"
Cohesion: 0.16
Nodes (12): AggregateMetric, API_ROUTES, ApiError, Citation, DocumentChunk, EvaluationItemResult, EvaluationMetric, Outcome (+4 more)

### Community 31 - "Community 31"
Cohesion: 0.27
Nodes (11): AnswerResponse, QuotaStatus, askQuestion(), getQuota(), QuestionApiError, questionRequest(), outcomeLabel(), QuestionPanel() (+3 more)

### Community 32 - "Community 32"
Cohesion: 0.19
Nodes (8): update_profile(), IdentityError, _next_approved_count(), _validate_approval_transition(), test_approved_user_counter_changes_only_on_state_transitions(), test_invalid_firebase_token_is_rejected(), test_admin_metrics_endpoint_requires_admin_dependency(), deny()

### Community 33 - "Community 33"
Cohesion: 0.19
Nodes (3): normalize_error(), Passage, main()

### Community 34 - "Community 34"
Cohesion: 0.33
Nodes (8): FirestorePassageRetriever, VertexQueryEmbedder, FakeCollection, FakeEmbeddingClient, FakeFirestore, test_query_embedder_uses_query_task_and_document_dimensions(), test_retriever_filters_active_version_and_returns_ranked_top_five(), test_retriever_rejects_invalid_limits_and_empty_active_version()

### Community 35 - "Community 35"
Cohesion: 0.30
Nodes (8): AnswerResponse, QuotaStatus, validate_telemetry_payload(), quota(), test_answer_contract_supports_all_outcomes_and_requires_safe_shape(), test_answer_response_cannot_be_forwarded_as_telemetry_payload(), test_telemetry_is_allowlisted_and_rejects_interaction_content(), trace()

### Community 36 - "Community 36"
Cohesion: 0.24
Nodes (3): DocumentCollection, DocumentReference, FirestoreClient

### Community 37 - "Community 37"
Cohesion: 0.21
Nodes (5): current_profile(), FirestoreUserRepository, get_current_profile(), get_user_repository(), update_user_approval()

### Community 39 - "Community 39"
Cohesion: 0.18
Nodes (3): FakeDocument, FakeFilteredQuery, FakeNearestQuery

### Community 42 - "Community 42"
Cohesion: 0.31
Nodes (4): FakeClient, FakeModels, test_document_embedding_rejects_wrong_vertex_dimension(), test_document_embedding_uses_fixed_model_task_and_dimension()

### Community 43 - "Community 43"
Cohesion: 0.25
Nodes (8): devDependencies, jsdom, @testing-library/jest-dom, @testing-library/react, @testing-library/user-event, @types/react, @types/react-dom, vitest

### Community 44 - "Community 44"
Cohesion: 0.25
Nodes (7): compilerOptions, allowSyntheticDefaultImports, composite, module, moduleResolution, skipLibCheck, include

### Community 46 - "Community 46"
Cohesion: 0.29
Nodes (7): dependencies, firebase, react, react-dom, typescript, vite, @vitejs/plugin-react

## Knowledge Gaps
- **6 isolated node(s):** `dmbok-compass-backend`, `typescript`, `vite`, `@types/react`, `@types/react-dom` (+1 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 358 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `OpenRouterAdapter` connect `Community 4` to `Community 33`, `Community 14`, `Community 17`, `Community 20`, `Community 29`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Are the 21 inferred relationships involving `UserProfile` (e.g. with `current_profile()` and `FirestoreUserRepository`) actually correct?**
  _`UserProfile` has 21 INFERRED edges - model-reasoned connections that need verification._
- **What connects `dmbok-compass-backend`, `typescript`, `vite` to the rest of the system?**
  _6 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.052795031055900624 - nodes in this community are weakly interconnected._
- **Why does `UserProfile` connect `Community 6` to `Community 0`, `Community 1`, `Community 32`, `Community 3`, `Community 37`, `Community 41`, `Community 11`, `Community 12`, `Community 14`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `GenerationError` (e.g. with `GroundedAnswerPolicy` and `FakeAdapter`) actually correct?**
  _`GenerationError` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.08106473079249849 - nodes in this community are weakly interconnected._