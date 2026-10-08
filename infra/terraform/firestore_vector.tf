resource "google_firestore_index" "document_chunks_vector" {
  project    = var.project_id
  database   = "(default)"
  collection = "document_chunks"

  fields {
    field_path = "corpus_version_id"
    order      = "ASCENDING"
  }

  fields {
    field_path = "embedding"

    vector_config {
      dimension = 768

      flat {}
    }
  }

  lifecycle {
    # Firestore adds the document-name field to composite indexes server-side.
    # Ignore that provider-managed field so routine plans do not replace the
    # active vector index.
    ignore_changes = [fields]
  }
}
