# Three-user latency and cost evidence

The release performance contract is three simultaneous approved users. The
normal mix must have a p95 end-to-end response time at or below 15 seconds;
fallback responses are reported separately because they may exceed that target.
The probe retains no question, answer, citation, token, or provider error text.

## Run the probe

Use three approved-user Firebase ID tokens against the preview or production
API. Revalidate current OpenRouter prices before supplying the four price
arguments; the defaults are zero so an unpriced run cannot accidentally claim
provider cost evidence.

```bash
PYTHONPATH=backend python scripts/run_performance_probe.py \
  --base-url https://preview.example.com \
  --token "$USER_1_TOKEN" --token "$USER_2_TOKEN" --token "$USER_3_TOKEN" \
  --requests-per-user 10 \
  --primary-input-eur-per-million 0 \
  --primary-output-eur-per-million 0 \
  --fallback-input-eur-per-million 0 \
  --fallback-output-eur-per-million 0 \
  --gcp-cost-forecast-eur 0 \
  --output performance-evidence.json
```

The command exits non-zero when any request fails or the successful normal-mix
p95 exceeds 15 seconds. Commit the generated JSON only when it is an approved
release artifact and contains no secrets. Its `schema_version` is
`performance-evidence-v1`, which is the versioned report consumed by release
evidence.

The cost forecast uses the configured daily request cap, token assumptions,
observed fallback rate, current model prices, and the supplied GCP forecast.
The report checks their combined monthly forecast against the €5 GCP envelope;
the provider component is still a forecast, not a billing statement, and must
be revalidated against provider billing before live release.
