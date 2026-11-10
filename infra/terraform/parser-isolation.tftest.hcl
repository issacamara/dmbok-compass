mock_provider "google" {}
mock_provider "google-beta" {}

run "parser_secret_is_restricted_to_ingestion" {
  command = plan

  variables {
    project_id             = "dmbok-compass-dev"
    billing_account_id     = "000000-000000-000000"
    corpus_bucket_name     = "dmbok-compass-dev-corpus"
    hosting_site_id        = "dmbok-compass-dev"
    runtime_secret_version = "1"
  }

  assert {
    condition     = google_service_account.workload["parser"].account_id != google_service_account.workload["worker"].account_id && google_service_account.workload["parser"].account_id != google_service_account.workload["runtime"].account_id
    error_message = "Parsing must have a dedicated identity."
  }

  assert {
    condition     = google_secret_manager_secret.parser.secret_id == "llamacloud-parser-api-key"
    error_message = "The parser must use its separate secret."
  }

  assert {
    condition     = !contains(keys(local.secret_access), "runtime:llamacloud-parser-api-key") && !contains(keys(local.secret_access), "worker:llamacloud-parser-api-key")
    error_message = "API and evaluation identities must not receive parser secret access."
  }

  assert {
    condition     = google_cloud_run_v2_job.parser.template[0].template[0].containers[0].args == tolist(["-m", "app.ingestion.parsing_entrypoint"])
    error_message = "The parser job must use its fixed entry point."
  }

  assert {
    condition     = google_cloud_run_v2_job_iam_member.runtime_parser_executor.role == "roles/run.jobsExecutor" && google_cloud_run_v2_job_iam_member.runtime_parser_invoker.role == "roles/run.invoker"
    error_message = "The API may run only the fixed parser job without execution overrides."
  }
}
