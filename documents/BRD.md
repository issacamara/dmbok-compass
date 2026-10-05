# DMBOK Compass — Business Requirements Document

| Document field | Value |
|---|---|
| Status | Approved |
| Version | 0.1 |
| Date | 4 October 2026 |
| Initiative owner | Sponsor / product owner / administrator (same individual) |
| Priority scale | Must, Should, Could, Won't for now |

## 1. Executive summary and business need

DMBOK Compass is a small, access-controlled, browser-based retrieval-augmented generation assistant for the searchable PDF of *DAMA-DMBOK, Second Edition*. Its purpose is to reduce the time spent searching the book and make it easier to connect related concepts across chapters.

The assistant will answer English-language questions using only evidence retrieved from the approved corpus. It will provide citations containing the page, chapter or section, and supporting excerpt. It may synthesize multiple passages when the synthesis is clearly identified and supported. When the corpus contains only partial evidence, the answer will state its limited confidence. When the corpus contains no relevant evidence, the assistant will refuse rather than fill the gap with general model knowledge.

The product is primarily a learning project and personal tool, but up to 10 administrator-approved users may access it. “Production-grade” means sound engineering, security, traceability, testing, and recoverability; it does not imply a contractual uptime commitment. Release readiness will be determined by a repeatable evaluation pipeline measuring retrieval quality, grounding, citation correctness, answer quality, refusal behavior, and response time.

The publisher's rights notice expressly restricts reproduction and transmission, including use in information storage and retrieval systems. The sponsor has decided to proceed without written publisher permission and accepts this as a high-severity unresolved risk. This document does not establish that the intended use is legally permitted.

## 2. Objectives and measurable success criteria

### 2.1 Objectives

| ID | Objective |
|---|---|
| OBJ-01 | Reduce the effort required to locate definitions, concepts, and guidance in the DMBOK PDF. |
| OBJ-02 | Enable evidence-backed comparison and synthesis across DMBOK chapters and knowledge areas. |
| OBJ-03 | Ensure answers remain grounded in the DMBOK corpus and provide verifiable source citations. |
| OBJ-04 | Prevent release unless retrieval, grounding, answer quality, refusal behavior, and performance meet agreed thresholds. |
| OBJ-05 | Operate a secure, recoverable learning service for a maximum of 10 approved users within the sponsor's cost constraint. |

### 2.2 Success measures and release gates

The initial thresholds may be recalibrated after the first baseline run, but any change must be recorded and approved by the sponsor before it is used for a release decision.

| ID | Measure | Initial pass threshold | Measurement basis |
|---|---|---:|---|
| SM-01 | Top-5 retrieval success | At least 90% | An approved relevant passage appears in the first five retrieved passages for answerable questions in the human-reviewed gold subset. |
| SM-02 | Grounded factual claims | At least 95% | Factual claims in evaluated answers are supported by cited corpus passages. |
| SM-03 | Citation correctness | At least 95% | Citations accurately support the claims to which they are attached. |
| SM-04 | Answer quality | At least 85% | Answers meet the approved correctness and usefulness rubric. |
| SM-05 | Correct refusal behavior | At least 95% | Questions designated unanswerable from the corpus trigger the required refusal. |
| SM-06 | End-to-end response time | At least 95% within 15 seconds | Complete answer and citations under normal test conditions with up to three simultaneous users. |
| SM-07 | Approved-user ceiling | 100% compliance | No more than 10 approved user accounts can access question answering. |
| SM-08 | Infrastructure cost | No more than €5 per month | Sponsor review of infrastructure usage and charges; hosted LLMs are expected to use free tiers. |

## 3. Stakeholders and user groups

| Stakeholder or group | Responsibilities and needs |
|---|---|
| Sponsor / product owner | Owns scope, budget, risk acceptance, priorities, and release approval. |
| Administrator | Approves or rejects registrations, manages approved users, configures usage caps, monitors aggregate usage, and runs evaluations. |
| Evaluator / corpus owner | Supplies approximately 200 representative questions and reviews generated reference answers and passages for a 30–50-question gold subset. |
| Approved user | Asks DMBOK questions and receives grounded answers, supporting excerpts, and citations. |
| Registration applicant | Submits an email address, username, and password for administrator review; cannot use the assistant until approved. |
| Hosted model provider | Processes question context and retrieved excerpts subject to its service availability and data-handling terms. |

The same individual currently performs the sponsor, product owner, administrator, evaluator, corpus-owner, support, and release-approver roles.

## 4. Current state and future state

### 4.1 Current state

The sponsor searches the DMBOK PDF manually. Locating relevant passages is time-consuming, and connecting concepts distributed across chapters requires additional reading and comparison. There is no repeatable evaluation set or automated quality gate.

### 4.2 Future state

An approved user signs in, asks a question in English, and receives a complete answer within the target response time. The answer is based only on retrieved DMBOK passages and includes page, chapter or section, and supporting excerpts. The response exposes a request-scoped retrieval trace without saving the conversation. Administrators can approve users, enforce configurable quotas, monitor anonymous aggregate usage, and run a versioned evaluation suite before release.

## 5. Scope and out of scope

### 5.1 In scope

- Responsive English-language browser chat for desktop and mobile.
- Public registration requests followed by administrator approval.
- Email-and-password authentication and account recovery.
- Maximum of 10 approved users and support for three simultaneous users.
- Concept explanations, definition lookup, knowledge-area comparisons, study support, and scenario guidance.
- Corpus-only retrieval, synthesis, confidence qualification, refusal behavior, and passage-level citations.
- Primary and fallback hosted model support.
- Configurable per-user and global request caps.
- Minimal administration for users, limits, aggregate usage, and evaluations.
- Offline ingestion of one searchable DMBOK PDF through an administrator maintenance procedure.
- Repeatable pre-release evaluation using approximately 200 questions and a human-reviewed gold subset.
- Daily backup of persistent application data and tested restoration.

### 5.2 Out of scope for the first release

- Anonymous question answering.
- More than 10 approved users.
- Languages other than English.
- Saved chat history or persistent interaction traces.
- User feedback, ratings, or comments.
- Multifactor authentication.
- Admin-screen PDF upload, corpus editing, or re-indexing controls.
- Multiple books or non-DMBOK sources.
- Contractual uptime, paid support, or high-availability infrastructure.
- Native mobile applications.

## 6. Business process requirements

| ID | Process requirement | Owner | Priority | Acceptance criteria |
|---|---|---|---|---|
| BPR-001 | Registration and approval | Administrator | Must | An applicant can submit the required account information; the account cannot ask questions until the administrator approves it. |
| BPR-002 | Grounded question answering | Product owner | Must | An approved user can submit an English question and receive either a cited corpus-grounded answer or the defined evidence-based refusal. |
| BPR-003 | Corpus maintenance | Administrator | Must | The source PDF can be ingested and its index rebuilt through a documented administrator-only maintenance procedure. |
| BPR-004 | Pre-release evaluation | Evaluator | Must | A release candidate can be evaluated against the versioned evaluation set, with each release-gate metric reported as pass or fail. |
| BPR-005 | Release decision | Sponsor | Must | The sponsor can approve release only after reviewing evaluation results, accepted risks, and unresolved exceptions. |
| BPR-006 | Backup and recovery | Administrator | Must | Persistent data is backed up daily and a documented restore test demonstrates recovery within 24 hours. |

## 7. Functional requirements

| ID | Requirement | Source / stakeholder | Priority | Acceptance criteria |
|---|---|---|---|---|
| FR-001 | The system shall let any visitor submit a registration request containing email, username, and password. | Sponsor | Must | A valid request is recorded as pending; missing or invalid required fields are rejected with a clear message. |
| FR-002 | The system shall verify control of the submitted email address before the account can be approved. | Confirmed checkpoint assumption | Must | An unverified account cannot become active; completing verification marks the email as verified. |
| FR-003 | The administrator shall be able to approve or reject pending registrations and deactivate approved users. | Sponsor | Must | Each action changes access immediately and is visible in the user-management view. |
| FR-004 | The system shall prevent approval of more than 10 active approved users. | Sponsor | Must | An attempt to approve an eleventh active user is blocked with an explanatory message. |
| FR-005 | Approved users shall be able to sign in with email and password, sign out, and securely reset a forgotten password. | Sponsor | Must | Valid credentials grant access; invalid, pending, rejected, or deactivated accounts do not; a verified user can complete the reset flow. |
| FR-006 | An approved user shall be able to submit an English-language DMBOK question through a browser chat interface. | Sponsor | Must | The system accepts the question, shows processing state, and returns a result or a clear operational error. |
| FR-007 | The system shall answer using only evidence retrieved from *DAMA-DMBOK, Second Edition*. | Sponsor | Must | Evaluation finds no uncited external factual content in answers counted as grounded. |
| FR-008 | The system shall support definitions, concept explanations, knowledge-area comparisons, study questions, and scenario guidance. | Sponsor | Must | The approved evaluation set contains representative questions from each use case and the system processes each category. |
| FR-009 | The system may synthesize multiple corpus passages and shall clearly label the result as synthesis or interpretation. | Sponsor | Must | A multi-passage synthesized answer is visibly labeled and every material claim maps to at least one citation. |
| FR-010 | Each cited answer shall identify the page, chapter or section, and supporting excerpt for every citation. | Sponsor | Must | Every displayed citation contains all three elements and can be associated with the supported answer claim. |
| FR-011 | The system shall apply three evidence outcomes: normal answer for strong evidence, qualified low-confidence answer for partial evidence, and refusal for no relevant evidence. | Sponsor | Must | Tests for all three evidence conditions produce the specified behavior without unsupported factual completion. |
| FR-012 | Each live answer shall expose a request-scoped retrieval trace containing retrieved passages, available relevance scores, model used, and timing. | Sponsor | Must | The trace is inspectable with the answer during the request session and is unavailable after the request/session lifecycle ends. |
| FR-013 | The system shall not retain questions, retrieved passages, generated answers, or conversation histories after request processing completes. | Sponsor | Must | Storage inspection after completed requests finds no persisted interaction content in application data or logs. |
| FR-014 | The system shall automatically attempt a configured fallback hosted model when the primary model is unavailable or rate-limited. | Sponsor | Must | A simulated primary-provider failure routes the request to the fallback and preserves grounding, citation, privacy, and quota behavior. |
| FR-015 | The system shall enforce a configurable per-user daily request limit with a default of 100. | Sponsor | Must | The 101st request by one user in the same quota day is refused by default; an administrator can change the limit without redeployment. |
| FR-016 | The system shall enforce a configurable global daily request limit. | Sponsor | Must | Requests are refused after the configured global cap is reached, and the administrator can change the cap without redeployment. |
| FR-017 | The system shall clearly inform a user when a request is blocked by an account or global quota and when the quota resets. | Sponsor | Must | The refusal identifies the applicable limit and reset timing without exposing another user's usage. |
| FR-018 | The administrator shall be able to view anonymous aggregate request counts, quota consumption, provider failures, and infrastructure-cost indicators without viewing user question content. | Sponsor | Must | The administration view reports the agreed aggregates and contains no stored question, answer, or retrieved-passage text. |
| FR-019 | The system shall run the complete evaluation set or a selected evaluation subset against a release candidate. | Sponsor / evaluator | Must | An administrator can initiate an evaluation and receives a uniquely identified result for the tested configuration and dataset version. |
| FR-020 | Evaluation shall calculate retrieval success, grounding, citation correctness, answer quality, refusal correctness, and response-time metrics. | Sponsor / evaluator | Must | The report shows a numerator, denominator, percentage, threshold, and pass/fail result for each applicable release gate. |
| FR-021 | The evaluation workflow shall support generated candidate annotations and explicit human approval of a 30–50-question gold subset. | Sponsor / evaluator | Must | Each gold item records its question, expected answer or rubric, relevant passage(s), review status, and dataset version. |
| FR-022 | The system shall persist evaluation datasets, approved annotations, run configuration, aggregate results, and release-gate decisions. | Sponsor / evaluator | Must | A prior evaluation can be identified and reproduced from the stored dataset version, configuration metadata, and result record without storing user chats. |

## 8. Nonfunctional requirements

| ID | Requirement | Source / stakeholder | Priority | Acceptance criteria |
|---|---|---|---|---|
| NFR-001 | At least 95% of normal tested requests shall return the complete answer and citations within 15 seconds. | Sponsor | Must | A representative performance test records at least 95% completion within 15 seconds, measured end to end. |
| NFR-002 | The service shall support up to three simultaneous users under normal use. | Sponsor | Must | A three-user concurrency test completes without data leakage or functional failure and meets NFR-001's threshold. |
| NFR-003 | Infrastructure cost shall not exceed €5 per month under the planned usage profile; hosted LLM usage shall use free tiers unless the sponsor changes this constraint. | Sponsor | Must | Monthly cost review shows no more than €5 in infrastructure charges, or release/continued operation requires an approved exception. |
| NFR-004 | All application traffic carrying credentials or protected content shall use HTTPS. | Confirmed checkpoint assumption | Must | Security verification finds no supported plaintext HTTP path for registration, authentication, administration, or question answering. |
| NFR-005 | Passwords shall be stored using an industry-standard salted password-hashing method and shall never be logged or stored in plaintext. | Confirmed checkpoint assumption | Must | Storage and log inspection reveal no plaintext passwords; security tests verify the configured password hashing. |
| NFR-006 | Authentication endpoints shall be protected by login throttling and secure reset tokens. | Confirmed checkpoint assumption | Must | Repeated failed authentication is throttled, and reset tokens are single-use, expire, and do not reveal whether unrelated accounts exist. |
| NFR-007 | The browser interface shall be usable on current desktop and mobile browsers. | Sponsor | Must | Core registration, sign-in, question, citation, and admin workflows pass responsive tests at agreed desktop and mobile viewport sizes. |
| NFR-008 | The first release shall meet basic WCAG 2.1 AA expectations for core workflows. | Sponsor | Must | Automated checks and keyboard/manual review find no unresolved critical accessibility failure in core workflows. |
| NFR-009 | Persistent data shall be backed up daily, with no more than 30 backup copies retained. | Sponsor | Must | Backup records show daily execution and automated or manual retention enforcement at 30 copies. |
| NFR-010 | Persistent application data shall be restorable within 24 hours; the corpus index may be rebuilt from the source PDF. | Sponsor | Must | A restore exercise recovers accounts, configuration, aggregate usage, and evaluation records within 24 hours, and the index rebuild procedure succeeds. |
| NFR-011 | Interaction content shall be ephemeral within the application and excluded from persistent logs, analytics, and backups. | Sponsor | Must | Privacy verification confirms that completed request content is absent from persistent application storage, analytics, logs, and backups. |
| NFR-012 | Evaluation results shall be reproducible and traceable to dataset, corpus, retrieval, prompt, model, and scoring configuration versions where those identifiers are available. | Product objective | Must | An evaluation report lists the available version identifiers needed to distinguish the tested release candidate. |
| NFR-013 | Availability shall be best-effort within free-provider and €5-per-month constraints; operational failures shall be reported clearly. | Sponsor | Must | Provider or infrastructure failure produces a clear, non-misleading service message and does not produce an ungrounded answer. |
| NFR-014 | The system shall prevent one authenticated user from accessing another user's account data or administrative functions. | Product objective | Must | Authorization tests demonstrate user isolation and reject non-admin access to administration endpoints. |

## 9. Data, reporting, integration, security, privacy, and compliance

### 9.1 Data requirements

| Data category | Treatment |
|---|---|
| Source corpus | One searchable PDF of *DAMA-DMBOK, Second Edition*, processed through a documented offline ingestion procedure. |
| Derived corpus data | Parsed structure, chunks, metadata, and retrieval index. Page and chapter/section provenance must be preserved. |
| Account data | Email, username, password hash, verification and approval state, role, and quota counters. |
| Interaction content | Question, retrieved passages, trace, and answer are transient and must not be persisted by the application. |
| Evaluation data | Approximately 200 questions, generated annotations, 30–50 human-approved gold items, scoring rubrics, dataset versions, and run results. |
| Operational data | Anonymous aggregate counts, quota consumption, provider failures, infrastructure indicators, configuration, and backup records. |

### 9.2 Reporting requirements

- The user-facing response must show citations and request-scoped retrieval trace information.
- The administration area must show only aggregate operational usage, not historical interaction content.
- Evaluation reports must show metric definitions, sample counts, measured values, thresholds, and pass/fail outcomes.
- Reports must distinguish human-reviewed gold-subset evidence from weaker automated evaluation evidence.

### 9.3 Integration requirements

- A primary hosted model and at least one configured fallback model are required.
- Provider changes must not weaken corpus-only grounding, citation, privacy, quota, or evaluation requirements.
- Specific model providers, embedding services, and deployment platforms are to be selected during solution design within the budget and provider-term constraints.

### 9.4 Security and privacy requirements

- Access is restricted to approved authenticated users; administrative functions are restricted to the administrator role.
- Multifactor authentication is explicitly out of scope. The associated administrator account-takeover risk is accepted, subject to password, throttling, HTTPS, and recovery controls.
- Users must be warned not to submit confidential or personal information because hosted providers may process or retain requests.
- Provider terms should prefer no-training and minimal-retention treatment. The minimum acceptable provider terms remain to be confirmed.
- Secrets and provider credentials must not be exposed to browser clients or committed to source control.

### 9.5 Copyright and licensing

The supplied publisher notice states that no part of the book may be reproduced or transmitted by any means, including an information storage and retrieval system, without written permission, except brief quotations in a review. The planned ingestion, chunking, embedding, transmission to model providers, and display of excerpts may fall within the restricted activities.

The sponsor has explicitly decided to proceed without publisher permission and accepts this as a high-severity unresolved risk. Administrator approval of users does not confer rights from the copyright holder. Recommended mitigation remains obtaining publisher permission and qualified legal advice before multi-user release. If a competent legal or publisher determination prohibits the planned use, release must be reconsidered or the corpus and product behavior changed.

## 10. Assumptions, constraints, dependencies, risks, and mitigations

### 10.1 Assumptions and constraints

- The €5 infrastructure limit is monthly.
- Hosted LLMs will be used through free tiers.
- The source PDF is searchable and contains sufficient structural information to recover pages and chapter/section labels.
- The sponsor will provide approximately 200 representative evaluation questions.
- The sponsor can review and approve generated annotations for 30–50 representative questions.
- There is no fixed delivery date.
- Availability is best-effort; quality, security, traceability, testing, and recoverability define “production-grade” for this initiative.

### 10.2 Dependencies

- Continued availability and acceptable capacity of at least two hosted model options.
- A deployment and storage platform that can meet the €5 monthly constraint.
- Reliable extraction of page and section metadata from the PDF.
- Sponsor availability for user approvals, annotation review, support, risk acceptance, and release decisions.
- An email-delivery mechanism for verification and password recovery.

### 10.3 Risks and mitigations

| ID | Risk | Severity | Mitigation / treatment |
|---|---|---|---|
| RISK-01 | Corpus ingestion, provider transmission, and excerpt display may infringe publisher rights. | High | Sponsor accepts the unresolved risk; recommended actions are written permission and qualified legal review before release. |
| RISK-02 | Free hosted models may be unavailable, rate-limited, changed, or discontinued. | High | Configure a fallback model, enforce quotas, report failures clearly, and avoid promising uptime. |
| RISK-03 | Hosted providers may retain or train on questions and excerpts. | High | Prefer no-training/minimal-retention terms, disclose processing, warn against sensitive inputs, and confirm acceptable terms before provider selection. |
| RISK-04 | Generated evaluation annotations may repeat system errors and overstate quality. | High | Human-review 30–50 gold items, report gold and automated results separately, and use gold results for release gates. |
| RISK-05 | €5 monthly infrastructure and free-model constraints may not support 1,000 requests per day. | High | Apply configurable per-user and global caps, monitor aggregate usage and costs, and lower limits when capacity requires it. |
| RISK-06 | No MFA increases administrator account-takeover risk. | Medium | Enforce strong password handling, HTTPS, throttling, secure recovery, and least-privilege authorization. |
| RISK-07 | PDF extraction may associate passages with incorrect pages or sections. | High | Validate citation metadata in the gold subset and make citation correctness a release gate. |
| RISK-08 | A single individual owns all operational and approval roles. | Medium | Document maintenance, recovery, evaluation, and release procedures; keep reproducible configuration and backups. |
| RISK-09 | Ephemeral interaction data limits retrospective incident diagnosis. | Medium | Expose request-scoped traces, retain non-content aggregate error metrics, and reproduce issues with explicit test questions when possible. |

## 11. Delivery, rollout, training, and support

### 11.1 Delivery approach

Delivery has no fixed deadline and should advance through readiness stages:

1. Validate PDF parsing, page/section provenance, chunking, and index reproducibility.
2. Validate corpus-only retrieval, answer synthesis, citations, confidence behavior, and refusal.
3. Add registration, approval, authentication, quotas, minimal administration, and privacy controls.
4. Build and human-review the evaluation gold subset; establish baseline results.
5. Validate performance, security, accessibility, backup, and recovery requirements.
6. Release to all approved users only after the sponsor reviews the release gates and accepted risks.

### 11.2 Rollout

There is no separate pilot cohort. At launch, all accounts approved by the administrator may use the service, up to the 10-user limit.

### 11.3 Training and support

- Formal training is not required.
- The interface should explain citation use, confidence labels, refusal behavior, quotas, and the warning against sensitive inputs.
- The sponsor provides administration and user support.
- Maintenance procedures must cover ingestion, index rebuild, configuration, evaluation, backup, restore, and provider failover.

## 12. Open questions, traceability, and approval

### 12.1 Open questions

| ID | Open question | Owner | Why it matters |
|---|---|---|---|
| OQ-01 | What is the default global daily request cap? | Sponsor | It determines maximum load and protection against provider and budget exhaustion. |
| OQ-02 | Which primary and fallback providers meet the functional, capacity, cost, and data-handling requirements? | Sponsor / solution design | Provider limits and terms affect availability, privacy, and feasibility. |
| OQ-03 | What provider retention and training terms are minimally acceptable? | Sponsor | The application cannot promise non-retention if a provider retains request data. |
| OQ-04 | What distribution of answerable, partially answerable, unanswerable, and use-case categories will the ~200-question set contain? | Sponsor / evaluator | Balanced coverage is necessary for meaningful quality and refusal metrics. |
| OQ-05 | Will publisher permission or qualified legal advice be obtained before multi-user release? | Sponsor | Licensing remains a high-severity unresolved release risk even though the sponsor has accepted it. |

### 12.2 Objective traceability

| Objective | Supporting requirements | Success measures / acceptance evidence |
|---|---|---|
| OBJ-01 | FR-006, FR-008, FR-010, NFR-001 | SM-04, SM-06; user workflow acceptance test |
| OBJ-02 | FR-008, FR-009, FR-010 | SM-02, SM-03, SM-04; comparison and scenario gold questions |
| OBJ-03 | FR-007, FR-009, FR-010, FR-011, FR-012, FR-013 | SM-01, SM-02, SM-03, SM-05; gold-subset evaluation |
| OBJ-04 | FR-019, FR-020, FR-021, FR-022, NFR-012 | SM-01 through SM-06; versioned evaluation report and sponsor release decision |
| OBJ-05 | FR-001 through FR-005, FR-014 through FR-018, NFR-002 through NFR-011, NFR-013, NFR-014 | SM-07, SM-08; security tests, cost review, backup/restore test |

### 12.3 Approval record

| Role | Name | Decision | Date |
|---|---|---|---|
| Sponsor / product owner / release approver | Sponsor | Approved | 4 October 2026 |

## Next Steps

1. **Sponsor:** review this draft and identify corrections or approve it as the requirements baseline.
2. **Sponsor:** supply the approximately 200 evaluation questions and decide the default global daily cap.
3. **Sponsor / evaluator:** review and approve generated annotations for a representative 30–50-question gold subset.
4. **Sponsor / solution design:** select primary and fallback providers after reviewing quotas, retention, training, and cost terms.
5. **Sponsor:** reconsider publisher permission or obtain qualified legal advice before multi-user release; record the final release-risk decision.
6. **Solution design:** translate the approved BRD into a reuse-first architecture and delivery backlog without weakening the release gates.
