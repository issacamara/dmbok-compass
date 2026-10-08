resource "google_firestore_backup_schedule" "daily" {
  project  = var.project_id
  database = google_firestore_database.default.name

  # A daily backup retained for 30 days produces at most 30 scheduled copies.
  retention = "2592000s"

  daily_recurrence {}
}