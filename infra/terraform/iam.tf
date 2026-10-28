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
    "google.subject"           = "assertion.sub"
    "attribute.repository"     = "assertion.repository"
    "attribute.workflow"       = "assertion.workflow"
    "attribute.workflow_event" = "assertion.workflow + ':' + assertion.event_name"
  }

  # PR plans use the pull request ref and base_ref; merged applies use the
  # protected branch ref. Both paths require the matching GitHub Environment.
  attribute_condition = "assertion.repository == '${var.github_repository}' && (assertion.ref == 'refs/heads/${var.github_branch}' || (assertion.event_name == 'pull_request' && assertion.base_ref == '${var.github_branch}')) && assertion.environment == '${var.github_environment}'"

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

locals {
  terraform_plan_roles = toset([
    "roles/datastore.viewer",
    "roles/iam.securityReviewer",
    "roles/viewer",
  ])

  terraform_apply_roles = toset([
    "roles/artifactregistry.admin",
    "roles/cloudscheduler.admin",
    "roles/datastore.owner",
    "roles/firebase.admin",
    "roles/firebasehosting.admin",
    "roles/iam.serviceAccountAdmin",
    "roles/iam.serviceAccountUser",
    "roles/iam.workloadIdentityPoolAdmin",
    "roles/identitytoolkit.admin",
    "roles/logging.configWriter",
    "roles/monitoring.editor",
    "roles/resourcemanager.projectIamAdmin",
    "roles/run.admin",
    "roles/secretmanager.admin",
    "roles/serviceusage.serviceUsageAdmin",
    "roles/storage.admin",
  ])
}

resource "google_project_iam_member" "terraform_plan" {
  for_each = local.terraform_plan_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.workload["build"].email}"
}

resource "google_project_iam_member" "terraform_apply" {
  for_each = local.terraform_apply_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.workload["build"].email}"
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

resource "google_storage_bucket_iam_member" "build_release_evidence_reader" {
  bucket = google_storage_bucket.release_evidence.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.workload["build"].email}"
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

resource "google_project_iam_member" "build_corpus_reader" {
  project = var.project_id
  role    = "roles/datastore.viewer"
  member  = "serviceAccount:${google_service_account.workload["build"].email}"
}

# Production copies the exact development image digest without rebuilding it.
# This grant is confined to the development repository.
resource "google_artifact_registry_repository_iam_member" "production_image_reader" {
  count      = var.project_id == "dev-dmbok-compass" ? 1 : 0
  project    = var.project_id
  location   = var.region
  repository = google_artifact_registry_repository.containers.repository_id
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${var.name_prefix}-build@${var.production_project_id}.iam.gserviceaccount.com"
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

# The API dispatches the evaluation job with a per-run environment override.
# Cloud Run requires the caller to act as the job's worker identity for this
# form of invocation, in addition to the job-level run.invoker grant.
resource "google_service_account_iam_member" "runtime_worker_user" {
  service_account_id = google_service_account.workload["worker"].name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.workload["runtime"].email}"
}

resource "google_service_account_iam_member" "github_actions_workload_identity_user" {
  service_account_id = google_service_account.workload["build"].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github_actions.name}/attribute.workflow_event/${var.github_deploy_workflow}:push"
}

resource "google_service_account_iam_member" "github_actions_terraform_workload_identity_user" {
  for_each = toset(["terraform:push", "terraform:workflow_dispatch"])

  service_account_id = google_service_account.workload["build"].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github_actions.name}/attribute.workflow_event/${each.value}"
}

resource "google_service_account_iam_member" "github_actions_terraform_plan_workload_identity_user" {
  for_each = toset(["terraform:pull_request", "terraform:workflow_dispatch"])

  service_account_id = google_service_account.workload["build"].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github_actions.name}/attribute.workflow_event/${each.value}"
}

# Cloud Scheduler receives only an identity here. The run.invoker grant is
# intentionally attached to each Cloud Run job/service when that resource is
# introduced, keeping this foundation from granting project-wide invocation.
