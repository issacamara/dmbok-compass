# Three-user performance and cost evidence

The release latency gate requires three distinct approved users and at least
95% of all requests to return the requested functional outcome within 30
seconds. Every request remains in the denominator. Operational failures,
outcome mismatches, evidence refusals, fallback latency, and first-request
latency are reported separately. The 15-second `response_time` measure is a
diagnostic and does not determine the release-quality result.

The probe retains no question, answer, citation, bearer token, raw provider
error, API hostname, path, or query string in its report. Set
`--expected-outcome` to `answer`, `qualified`, or `refusal` to state the
expected response for the probe question.

## Run the probe

Use three approved-user Firebase ID tokens against the preview API.
Provider-cap, provider-rate, and GCP forecast values come only from the
committed `documents/decisions/release-cost-approval.json` record in the
canonical repository's trusted `origin/development` ref, which must be an
ancestor of the tested checkout. The report records that exact commit SHA. The
record must bind to the frozen candidate and include dated source references.
No command-line values can override it. The current repository has no approved
record, so cost gates stay blocked until the sponsor decision and dated
pricing/billing evidence are committed. Synthetic records in unit tests do not
provide project cost approval.

```bash
PYTHONPATH=backend python scripts/run_performance_probe.py \
  --base-url https://preview.example.com \
  --token "$USER_1_TOKEN" --token "$USER_2_TOKEN" --token "$USER_3_TOKEN" \
  --requests-per-user 10 \
  --expected-outcome answer \
  --stage-usage-evidence stage-usage.json \
  --output performance-evidence.json
```

Provider-cap and cost evidence are read from the committed decision record and
must match the frozen candidate identity. Held-out and tuning evidence must
come from an in-process `BenchmarkReport` returned by `evaluate()` for the same
candidate and dataset as the completed evaluation run. Serialized digests alone
cannot pass that binding check.

Parser, generator, and judge stage evidence includes used and reserved units,
plus used and reserved EUR costs. Unknown values keep the release evidence
blocked. Keep generated reports only as approved release artifacts and ensure
they contain no secrets before committing them.
