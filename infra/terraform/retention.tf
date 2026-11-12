resource "google_firestore_field" "evaluation_output_ttl" {
  project    = var.project_id
  database   = google_firestore_database.default.name
  collection = "evaluation_candidate_outputs"
  field      = "expires_at"

  ttl_config {}
}
