# Publisher-rights release decision

**Status:** accepted release-risk decision; multi-user release remains gated  
**Decision date:** 2026-10-04  
**Decision owner:** sponsor / release approver  
**Evidence:** [BRD §1](../BRD.md#1-executive-summary-and-business-need), [BRD §9.5](../BRD.md#95-copyright-and-licensing), [Architecture A-10](../ARCHITECTURE.md#11-assumptions)

## Decision

The sponsor decides to continue the implementation without written publisher
permission and accepts the resulting publisher-rights risk for the current
project scope. This is an explicit release-risk decision, not a legal
conclusion or confirmation that the intended use is permitted.

The decision covers the planned use of the supplied publisher material for:

- corpus ingestion and storage;
- embeddings and retrieval indexes;
- transmission of retrieved context to the configured answer providers; and
- displaying grounded excerpts and citations to authenticated users.

The decision is limited to this project and its intended learning use. It does
not grant rights to reproduce, redistribute, or sublicense the source material.
Administrator approval of users does not confer rights from the publisher.

## Release gate

Multi-user release is **not approved by this record alone**. Before enabling
multi-user access, the sponsor must either:

1. record written publisher permission covering the intended use; or
2. obtain qualified legal advice and record the resulting release decision and
   scope here.

Until one of those outcomes is recorded, production access remains limited to
the approved pre-release validation scope and the final release gate must fail
for multi-user access.

## Risk treatment

The unresolved risk is high severity because ingestion, provider transmission,
and excerpt display may fall within restrictions in the publisher notice. The
implementation must continue to minimize excerpts, preserve provenance, and
retain this decision with the release evidence. Reconsideration is required if
the intended audience, corpus, provider route, or interaction model changes.

This record does not replace provider data-terms approval, corpus provenance,
or the final release-decision evidence.
