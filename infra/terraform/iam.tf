resource "google_service_account" "workload" {
  for_each = local.service_accounts

  project      = var.project_id
  account_id   = each.value
  display_name = "DMBOK Compass ${each.key} identity"

  depends_on = [google_project_service.required]
}

resource "google_iam_workload_identity_pool" "github_actions" {
  project                   = var.project_id
  workload_identity_pool_id = "${var.name_prefix}-github"
  display_name              = "${var.name_prefix} GitHub Actions"
  description               = "OIDC identities for the repository's GitHub Actions delivery workflow."

  depends_on = [google_project_service.required]
}

resource "google_iam_workload_identity_pool_provider" "github_actions" {
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github_actions.workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  display_name                       = "GitHub Actions OIDC"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }

  attribute_condition = "assertion.repository == '${var.github_repository}' && assertion.ref == 'refs/heads/main'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

locals {
  runtime_roles = toset([
    "roles/aiplatform.user",
    "roles/datastore.user",
    "roles/firebaseauth.viewer",
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

resource "google_service_account_iam_member" "github_actions_workload_identity_user" {
  service_account_id = google_service_account.workload["build"].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github_actions.name}/attribute.repository/${var.github_repository}"
}

# Cloud Scheduler receives only an identity here. The run.invoker grant is
# intentionally attached to each Cloud Run job/service when that resource is
# introduced, keeping this foundation from granting project-wide invocation.
