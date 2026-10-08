locals {
  # Match only content-free operational markers. Application and job code must
  # never include interaction content or credentials in these logs.
  operational_log_metrics = {
    request_failures = {
      filter      = "textPayload:\"telemetry\" AND textPayload:\"\\\"outcome\\\": \\\"error\\\"\""
      description = "Content-free request failures emitted by the telemetry adapter."
    }
    fallback_requests = {
      filter      = "textPayload:\"telemetry\" AND textPayload:\"provider_model\" AND textPayload:\"${var.fallback_model}\""
      description = "Requests served by the configured model fallback."
    }
    quota_exhaustions = {
      filter      = "textPayload:\"daily_quota_exceeded\" OR textPayload:\"global_daily_quota_exceeded\""
      description = "Requests refused by an account or global daily quota."
    }
    ingestion_failures = {
      filter      = "textPayload:\"dmbok_ingestion_failure\""
      description = "Content-free ingestion job failure markers."
    }
    backup_failures = {
      filter      = "textPayload:\"dmbok_backup_failure\""
      description = "Content-free backup job failure markers."
    }
    retrieval_failures = {
      filter      = "textPayload:\"dmbok_retrieval_failure\""
      description = "Content-free retrieval failure markers."
    }
  }
}

resource "google_logging_metric" "operational" {
  for_each = local.operational_log_metrics

  project     = var.project_id
  name        = "dmbok_${each.key}"
  description = each.value.description
  filter      = each.value.filter

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

resource "google_monitoring_alert_policy" "operational" {
  for_each = local.operational_log_metrics

  project               = var.project_id
  display_name          = "DMBOK ${replace(each.key, "_", " ")}"
  combiner              = "OR"
  notification_channels = var.notification_channel_ids
  depends_on            = [google_logging_metric.operational]

  documentation {
    content   = "${each.value.description} This alert contains no interaction content."
    mime_type = "text/markdown"
  }

  conditions {
    display_name = "${each.key} in the last five minutes"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/dmbok_${each.key}\" resource.type=\"global\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "300s"

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_SUM"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  alert_strategy {
    auto_close = "86400s"
  }
}

resource "google_monitoring_alert_policy" "request_latency" {
  project               = var.project_id
  display_name          = "DMBOK request latency over 15 seconds"
  combiner              = "OR"
  notification_channels = var.notification_channel_ids

  documentation {
    content   = "The 95th percentile Cloud Run request latency exceeded the NFR-001 target."
    mime_type = "text/markdown"
  }

  conditions {
    display_name = "Cloud Run p95 request latency"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" metric.type=\"run.googleapis.com/request_latencies\""
      comparison      = "COMPARISON_GT"
      threshold_value = 15000
      duration        = "300s"

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_PERCENTILE_95"
        cross_series_reducer = "REDUCE_MAX"
      }
    }
  }
}

resource "google_monitoring_alert_policy" "request_errors" {
  project               = var.project_id
  display_name          = "DMBOK Cloud Run request failures"
  combiner              = "OR"
  notification_channels = var.notification_channel_ids

  documentation {
    content   = "Cloud Run returned a 5xx response. Investigate content-free error telemetry and service health."
    mime_type = "text/markdown"
  }

  conditions {
    display_name = "Cloud Run 5xx responses"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" metric.type=\"run.googleapis.com/request_count\" metric.label.\"response_code_class\"=\"5xx\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "300s"

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }
}

resource "google_billing_budget" "monthly" {
  billing_account = var.billing_account_id
  display_name    = "DMBOK Compass monthly infrastructure budget"

  budget_filter {
    projects = ["projects/${var.project_id}"]
  }

  amount {
    specified_amount {
      currency_code = "EUR"
      units         = "5"
    }
  }

  threshold_rules {
    threshold_percent = 0.5
    spend_basis       = "CURRENT_SPEND"
  }

  threshold_rules {
    threshold_percent = 0.9
    spend_basis       = "CURRENT_SPEND"
  }

  threshold_rules {
    threshold_percent = 1
    spend_basis       = "CURRENT_SPEND"
  }

  threshold_rules {
    threshold_percent = 0.9
    spend_basis       = "FORECASTED_SPEND"
  }
}

resource "google_monitoring_dashboard" "operations" {
  project = var.project_id

  dashboard_json = jsonencode({
    displayName = "DMBOK Compass operations"
    gridLayout = {
      columns = "2"
      widgets = [
        {
          title = "Request latency (p95)"
          xyChart = {
            dataSets = [{
              plotType   = "LINE"
              targetAxis = "Y1"
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "resource.type=\"cloud_run_revision\" metric.type=\"run.googleapis.com/request_latencies\""
                  aggregation = {
                    alignmentPeriod    = "300s"
                    perSeriesAligner   = "ALIGN_PERCENTILE_95"
                    crossSeriesReducer = "REDUCE_MAX"
                  }
                }
              }
            }]
          }
        },
        {
          title = "Content-free operational events"
          xyChart = {
            dataSets = [for name in keys(local.operational_log_metrics) : {
              plotType = "STACKED_BAR"
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"logging.googleapis.com/user/dmbok_${name}\" resource.type=\"global\""
                  aggregation = {
                    alignmentPeriod    = "300s"
                    perSeriesAligner   = "ALIGN_SUM"
                    crossSeriesReducer = "REDUCE_SUM"
                  }
                }
              }
            }]
          }
        }
      ]
    }
  })
}
