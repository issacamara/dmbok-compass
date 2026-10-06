resource "google_service_account" "workload" {
  for_each = local.service_accounts

  project      = var.project_id
  account_id   = each.value
  display_name = "DMBOK Compass ${each.key} identity"

  depends_on = [google_project_service.required]
}

locals {
  runtime_roles = toset([
    "roles/aiplatform.user",
    "roles/datastore.user",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
  ])

  worker_roles = toset([
    "roles/aiplatform.user",
    "roles/datastore.user",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
  ])
}

resource "google_project_iam_member" "runtime" {
  for_each = local.runtime_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.workload["runtime"].email}"
}

resource "google_project_iam_member" "worker" {
  for_each = local.worker_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.workload["worker"].email}"
}

resource "google_storage_bucket_iam_member" "runtime_corpus_reader" {
  bucket = google_storage_bucket.corpus.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.workload["runtime"].email}"
}

resource "google_storage_bucket_iam_member" "worker_corpus_writer" {
  bucket = google_storage_bucket.corpus.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.workload["worker"].email}"
}

locals {
  secret_access = {
    for pair in setproduct(["runtime", "worker"], var.secret_names) :
    "${pair[0]}:${pair[1]}" => pair
  }
}

resource "google_secret_manager_secret_iam_member" "workload_access" {
  for_each = local.secret_access

  secret_id = google_secret_manager_secret.managed[each.value[1]].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.workload[each.value[0]].email}"
}

resource "google_project_iam_member" "build_artifact_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.workload["build"].email}"
}

resource "google_project_iam_member" "build_cloud_run_developer" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.workload["build"].email}"
}

resource "google_project_iam_member" "build_hosting_deployer" {
  project = var.project_id
  role    = "roles/firebasehosting.admin"
  member  = "serviceAccount:${google_service_account.workload["build"].email}"
}

resource "google_service_account_iam_member" "build_runtime_user" {
  for_each = toset(["runtime", "worker"])

  service_account_id = google_service_account.workload[each.value].name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.workload["build"].email}"
}

# Cloud Scheduler receives only an identity here. The run.invoker grant is
# intentionally attached to each Cloud Run job/service when that resource is
# introduced, keeping this foundation from granting project-wide invocation.
