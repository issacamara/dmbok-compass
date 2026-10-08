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
      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
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

  depends_on = [google_project_service.required]
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
        image = var.api_image
        args  = ["python", "-m", "app.ingestion.job"]
      }
    }
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
        image = var.api_image
        args  = ["python", "-m", "app.evaluation.job"]
      }
    }
  }

  depends_on = [google_project_service.required]
}
