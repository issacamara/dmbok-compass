run "monitoring_contract" {
  command = plan

  variables {
    project_id         = "dmbok-compass-dev"
    billing_account_id = "000000-000000-000000"
    corpus_bucket_name = "dmbok-compass-dev-corpus"
  }

  assert {
    condition     = google_billing_budget.monthly.amount[0].specified_amount[0].currency_code == "EUR"
    error_message = "The operations budget must be denominated in EUR."
  }

  assert {
    condition     = google_billing_budget.monthly.amount[0].specified_amount[0].units == "5"
    error_message = "The operations budget must be capped at 5 EUR."
  }

  assert {
    condition     = length(google_billing_budget.monthly.threshold_rules) == 4
    error_message = "The budget must expose three current-spend and one forecast threshold."
  }

  assert {
    condition     = length(google_logging_metric.operational) == 6
    error_message = "All six content-free operational log metrics must be provisioned."
  }

  assert {
    condition     = google_monitoring_alert_policy.request_latency.conditions[0].condition_threshold[0].threshold_value == 15000
    error_message = "The latency alert must use the 15-second service target."
  }
}
