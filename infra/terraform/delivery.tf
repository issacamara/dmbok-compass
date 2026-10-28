variable "api_image" {
  description = "Immutable API/worker image reference; production uses a registry digest."
  type        = string
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}

variable "api_service_name" {
  description = "Cloud Run service name for the API."
  type        = string
  default     = "dmbok-compass-api"
}

variable "ingestion_job_name" {
  description = "Cloud Run job name for corpus ingestion."
  type        = string
  default     = "dmbok-compass-ingestion"
}

variable "evaluation_job_name" {
  description = "Cloud Run job name for release evaluation."
  type        = string
  default     = "dmbok-compass-evaluation"
}


variable "corpus_version_id" {
  description = "Immutable corpus version coupled to this API and its admin jobs."
  type        = string
  default     = "unconfigured"
}

resource "google_cloud_run_v2_service" "api" {
  name     = var.api_service_name
  location = var.region
  project  = var.project_id

  template {
    service_account = google_service_account.workload["runtime"].email
    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }
    containers {
      image = var.api_image
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }
      env {
        name  = "RAG_VECTOR_COLLECTION"
        value = "document_chunks_v2"
      }
      env {
        name  = "RAG_VECTOR_SCHEMA_VERSION"
        value = "2"
      }
      env {
        name  = "RAG_EMBEDDING_MODEL"
        value = "gemini-embedding-001"
      }
      env {
        name  = "RAG_EMBEDDING_DIMENSIONS"
        value = "768"
      }
      env {
        name  = "RAG_RETRIEVAL_STRATEGY"
        value = "dense"
      }
      env {
        name  = "RAG_CORPUS_VERSION_ID"
        value = var.corpus_version_id
      }
      env {
        name  = "RAG_SCHEMA_VERSION"
        value = "v2"
      }
      env {
        name  = "RAG_VECTOR_INDEX"
        value = "document_chunks_v2_vector"
      }
      env {
        name  = "EVALUATION_JOB_NAME"
        value = var.evaluation_job_name
      }
      env {
        name  = "ANSWER_PROVIDER"
        value = "openrouter"
      }
      env {
        name  = "ANSWER_GENERATION_TIMEOUT_MS"
        value = "16000"
      }
      env {
        name  = "ANSWER_REQUEST_TIMEOUT_MS"
        value = "28000"
      }
      env {
        name  = "ANSWER_OUTPUT_TOKEN_CAP"
        value = "1024"
      }
      env {
        name  = "ANSWER_MAX_MODEL_ATTEMPTS"
        value = "2"
      }
      env {
        name  = "OPENROUTER_JSON_SCHEMA_SUPPORTED"
        value = "true"
      }
      env {
        name  = "RUNTIME_SECRET_VERSION"
        value = var.runtime_secret_version
      }
      env {
        name  = "OPENROUTER_PRIMARY_MODEL"
        value = "meta-llama/llama-3.1-8b-instruct"
      }
      env {
        name  = "OPENROUTER_FALLBACK_MODEL"
        value = "google/gemma-4-26b-a4b-it:free"
      }
      env {
        name = "OPENROUTER_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.managed["answer-provider-api-key"].secret_id
            version = var.runtime_secret_version
          }
        }
      }
      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }
      startup_probe {
        http_get {
          # Startup only verifies that the process serves HTTP. Release
          # readiness depends on a selected Firestore release.
          path = "/health/live"
        }
        initial_delay_seconds = 0
        timeout_seconds       = 2
        period_seconds        = 5
        failure_threshold     = 12
      }
      liveness_probe {
        http_get {
          path = "/health/live"
        }
        initial_delay_seconds = 10
        timeout_seconds       = 2
        period_seconds        = 10
        failure_threshold     = 3
      }
    }
    max_instance_request_concurrency = 8
  }

  traffic {
    type = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    # Cloud Run requires initial service creation to allocate traffic. The
    # delivery workflow creates subsequent revisions with zero traffic.
    percent = 100
  }

  lifecycle {
    # CD workflows own revisions and runtime env expansion after provisioning.
    # Terraform remains the owner of the service, identity, probes, and sizing.
    ignore_changes = [
      template[0].containers[0].image,
      template[0].containers[0].env,
    ]
  }

  depends_on = [
    google_project_service.required,
    google_secret_manager_secret_iam_member.workload_access,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_job" "ingestion" {
  name     = var.ingestion_job_name
  location = var.region
  project  = var.project_id

  template {
    template {
      service_account = google_service_account.workload["worker"].email
      max_retries     = 1
      timeout         = "3600s"
      containers {
        image   = var.api_image
        command = ["python"]
        args    = ["-m", "app.ingestion.entrypoint"]
        env {
          name  = "GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }
        env {
          name  = "GOOGLE_CLOUD_LOCATION"
          value = var.region
        }
        env {
          name  = "RAG_VECTOR_COLLECTION"
          value = "document_chunks_v2"
        }
        env {
          name  = "RAG_VECTOR_SCHEMA_VERSION"
          value = "2"
        }
        env {
          name  = "RAG_EMBEDDING_MODEL"
          value = "gemini-embedding-001"
        }
        env {
          name  = "RAG_EMBEDDING_DIMENSIONS"
          value = "768"
        }
        env {
          name  = "RAG_RETRIEVAL_STRATEGY"
          value = "dense"
        }
        env {
          name  = "RAG_CORPUS_VERSION_ID"
          value = var.corpus_version_id
        }
        env {
          name  = "RAG_SCHEMA_VERSION"
          value = "v2"
        }
        env {
          name  = "RAG_VECTOR_INDEX"
          value = "document_chunks_v2_vector"
        }
        env {
          name  = "CORPUS_BUCKET_NAME"
          value = var.corpus_bucket_name
        }
        env {
          name  = "CORPUS_OBJECT_NAME"
          value = var.corpus_object_name
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [template[0].template[0].containers[0].image, template[0].template[0].containers[0].env]
  }

  depends_on = [google_project_service.required]
}

resource "google_cloud_run_v2_job" "evaluation" {
  name     = var.evaluation_job_name
  location = var.region
  project  = var.project_id

  template {
    template {
      service_account = google_service_account.workload["worker"].email
      max_retries     = 1
      timeout         = "3600s"
      containers {
        image   = var.api_image
        command = ["python"]
        args    = ["-m", "app.evaluation.worker"]
        env {
          name  = "GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }
        env {
          name  = "GOOGLE_CLOUD_LOCATION"
          value = var.region
        }
        env {
          name  = "RAG_VECTOR_COLLECTION"
          value = "document_chunks_v2"
        }
        env {
          name  = "RAG_CORPUS_VERSION_ID"
          value = var.corpus_version_id
        }
        env {
          name  = "RAG_SCHEMA_VERSION"
          value = "v2"
        }
        env {
          name  = "RAG_VECTOR_INDEX"
          value = "document_chunks_v2_vector"
        }
        env {
          name  = "ANSWER_PROVIDER"
          value = "openrouter"
        }
        env {
          name  = "ANSWER_GENERATION_TIMEOUT_MS"
          value = "16000"
        }
        env {
          name  = "ANSWER_REQUEST_TIMEOUT_MS"
          value = "28000"
        }
        env {
          name  = "ANSWER_OUTPUT_TOKEN_CAP"
          value = "1024"
        }
        env {
          name  = "ANSWER_MAX_MODEL_ATTEMPTS"
          value = "2"
        }
        env {
          name  = "OPENROUTER_JSON_SCHEMA_SUPPORTED"
          value = "true"
        }
        env {
          name  = "RUNTIME_SECRET_VERSION"
          value = var.runtime_secret_version
        }
        env {
          name  = "OPENROUTER_PRIMARY_MODEL"
          value = "meta-llama/llama-3.1-8b-instruct"
        }
        env {
          name  = "OPENROUTER_FALLBACK_MODEL"
          value = "google/gemma-4-26b-a4b-it:free"
        }
        env {
          name = "OPENROUTER_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.managed["answer-provider-api-key"].secret_id
              version = var.runtime_secret_version
            }
          }
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [template[0].template[0].containers[0].image, template[0].template[0].containers[0].env]
  }

  depends_on = [google_project_service.required]
}

resource "google_cloud_run_v2_job_iam_member" "runtime_evaluation_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_job.evaluation.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.workload["runtime"].email}"
}
