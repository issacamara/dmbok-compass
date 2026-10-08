output "service_accounts" {
  description = "Workload identities to use for runtime, jobs, scheduling, builds, and hosting."
  value = {
    for name, account in google_service_account.workload : name => account.email
  }
}

output "corpus_bucket_name" {
  description = "Bucket containing the source PDF and versioned ingestion artifacts."
  value       = google_storage_bucket.corpus.name
}

output "artifact_repository" {
  description = "Fully qualified Artifact Registry repository."
  value       = google_artifact_registry_repository.containers.name
}

output "firestore_database" {
  description = "The default Firestore Native database."
  value       = google_firestore_database.default.name
}

output "hosting_site" {
  description = "Firebase Hosting site used by the web application."
  value       = var.hosting_site_id
}

output "github_workload_identity_provider" {
  description = "Workload Identity provider resource name for GitHub Actions authentication."
  value       = google_iam_workload_identity_pool_provider.github_actions.name
}
