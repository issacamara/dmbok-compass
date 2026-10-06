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
}
