# DMBOK Compass mockups

Open `index.html` in a browser. The mockup is a lightweight interactive concept, not production UI. Use the top navigation to move between the public landing state, Q&A answer state, administrator workspace, and sign-in state.

## Design direction

- **Tone:** calm, editorial, evidence-first, and trustworthy rather than “black-box AI”.
- **Inspiration:** 21st.dev’s spacious component cards, gradient surfaces, large typographic hero, and AI-chat/navigation patterns. Reference: https://21st.dev/
- **Palette:** deep ink for authority, leaf green for grounded actions, mint for positive evidence, and warm accents to make the tool feel approachable.
- **Primary journeys:** approved user asks a question → reads a qualified answer → verifies citations/trace; applicant requests access → verifies email → waits for approval; administrator reviews applicants → checks quotas → runs/reviews evaluation gates.

## Key UX decisions

1. Answer confidence, synthesis, citations, and trace are visible in the answer itself; they are not hidden behind a generic “AI” label.
2. “Nothing here is saved” is repeated near the trace because ephemeral interaction content is a core privacy promise.
3. Admin metrics are aggregate-only: no user question or answer content appears in the dashboard.
4. The ten-user ceiling, daily quota, and release-gate results are represented as operational product concepts, not buried settings.
5. The layout collapses to a single column below 800px and keeps the composer, evidence, and request trace accessible on mobile.

## Accessibility notes

The concept uses semantic landmarks, visible text labels, native form controls, color plus text/status cues, and large click targets. Before implementation, run keyboard and screen-reader checks, contrast checks, and responsive tests against the final component library; this mockup does not claim WCAG conformance.
