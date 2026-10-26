resource "google_storage_bucket" "corpus" {
  project                     = var.project_id
  name                        = var.corpus_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  depends_on = [google_project_service.required]
}

data "google_storage_bucket" "terraform_state" {
  name = var.github_environment == "production" ? "prod-dmbok-compass-tfstate" : "dev-dmbok-compass-tfstate"
}

resource "google_storage_bucket_iam_member" "terraform_plan_state_access" {
  bucket = data.google_storage_bucket.terraform_state.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.workload["build"].email}"
}

resource "google_storage_bucket_iam_member" "terraform_apply_state_access" {
  bucket = data.google_storage_bucket.terraform_state.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.workload["build"].email}"
}
