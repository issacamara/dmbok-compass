# LangChain compatibility and retirement audit

The application now uses the validated native LangChain stack in its production-shaped API, ingestion, evaluation, and maintenance callers. `backend/requirements-compat.lock` is the compatibility-only lock for this isolated spike, including the maintained Firestore wheel; the active application image and CI use `backend/requirements.lock`, generated from `backend/pyproject.toml` with the test extra. These locks must not be substituted for one another. The checks here are regression and target-environment compatibility gates, not a pending migration.

## Native caller inventory (2026-10-16)

The retirement audit found no superseded live RAG engine to remove. The six
mechanisms below have one native implementation path; application policy and
provenance helpers remain intentionally in place around those boundaries.

| Mechanism | Live caller | Native boundary | Audit result |
| --- | --- | --- | --- |
| PDF loading | `app.ingestion.extract.extract_pdf` | `langchain_community.document_loaders.PyPDFLoader` | retained page/section provenance policy |
| Page-bounded splitting | `app.corpus.chunking.chunk_pages` | `langchain_text_splitters.CharacterTextSplitter` | retained deterministic page-boundary policy |
| Embedding | `app.corpus.embedding.VertexEmbeddings` | `langchain_google_genai.GoogleGenerativeAIEmbeddings` | native document/query task types and 768 dimensions |
| Vector writes | `app.corpus.indexing.CorpusEmbeddingWriter` | `langchain_google_firestore.FirestoreVectorStore.add_documents` | native persistence with deterministic IDs |
| Retrieval | `app.retrieval.search.FirestorePassageRetriever` | `FirestoreVectorStore.similarity_search_with_score` | native filtered top-five search with preserved scores |
| Generation | `app.rag.LangChainRagFactory` | native Vertex/OpenRouter adapters via `model.langchain` | native structured generation and bounded fallback |

`app.answering` continues to own grounding, refusal, citation reconstruction,
privacy, quota, and response contracts. Those are product-policy boundaries,
not obsolete RAG engines, and were not deleted. `app.main`, the ingestion
entrypoint, and the evaluation worker all construct their services through
`LangChainRagFactory`; no direct legacy RAG caller remains.

This is a verified no-op runtime removal: no evidenced obsolete path remained
after the audit. The prior `document_chunks` artifact/index and release
evidence remain retained for the owner-approved rollback period. Promotion
and rollback signoff continue to be recorded by
`documents/release/langchain-rollback.md`; this audit does not promote traffic
or expire retained resources.
From `backend/`, create a disposable Python 3.11 environment and install `requirements-compat.lock` through the normal package resolver. Verify the wheel SHA-256 in the vendored package README first, then run `python -m pip check` and `python compat/check_local.py`. The same check is packaged in `compat/Dockerfile` for the target Linux Python 3.11 image: `docker build -f compat/Dockerfile -t dmbok-langchain-compat .`. The local check covers PDF loading, word-sized chunking, Vertex embedding task/dimension configuration, 501 native vector writes, nested metadata, filtered cosine scores, native chat structured-output construction and restricted fallback composition. It uses no cloud credentials.

For the Firestore check, set `DMBOK_COMPAT_PROJECT_ID` and `DMBOK_COMPAT_COLLECTION` to an explicitly approved **development** project and indexed collection, with application default credentials that can read/write/delete in that collection. Run `python compat/check_live_firestore.py`. It writes three uniquely named documents, verifies filtered native scores against a direct Firestore distance query, and deletes those documents in a `finally` block. The collection needs a 768-dimensional cosine vector index and a composite `metadata.corpus_version_id` filter index.

For the Vertex embedding route, set `DMBOK_COMPAT_PROJECT_ID` and `DMBOK_COMPAT_VERTEX_LOCATION` for the same development project, then run `python compat/check_live_vertex_embeddings.py`. It invokes native LangChain document and query embeddings with authored text and verifies finite 768-dimensional vectors.

For OpenRouter, set `OPENROUTER_API_KEY`, `OPENROUTER_PRIMARY_MODEL`, `OPENROUTER_FALLBACK_MODEL` and `OPENROUTER_JSON_SCHEMA_SUPPORTED` to the effective development route settings, then run `python compat/check_live_openrouter.py`. It uses an authored trivial prompt and calls both configured routes once. The local check covers the fallback exception filter; the later application integration must also verify deadline, attempt trace and quota behavior through its public contract. Test any enabled Vertex answer route separately before migrating that mode.

The migration remains blocked if the Python 3.11 image cannot install the lock, the indexed development Firestore check fails, or the configured live provider routes cannot produce the required vectors and structured output. A successful local fake check alone does not clear the gate. No production corpus or interaction is used here.
