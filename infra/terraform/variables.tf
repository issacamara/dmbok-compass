variable "project_id" {
  description = "The GCP project that owns the DMBOK Compass foundation."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,28}[a-z0-9]$", var.project_id))
    error_message = "project_id must be a valid GCP project ID."
  }
}

variable "production_project_id" {
  description = "Production project whose build identity may read development images for promotion."
  type        = string
  default     = "prod-dmbok-compass"
}

variable "billing_account_id" {
  description = "Billing account that owns the EUR monthly budget, for example 000000-000000-000000."
  type        = string

  validation {
    condition     = can(regex("^[A-Z0-9]{6}-[A-Z0-9]{6}-[A-Z0-9]{6}$", var.billing_account_id))
    error_message = "billing_account_id must use the GCP billing account format."
  }
}

variable "fallback_model" {
  description = "Configured fallback model identifier used by the content-free fallback log counter."
  type        = string
  default     = "google/gemma-4-26b-a4b-it:free"

  validation {
    condition     = length(trimspace(var.fallback_model)) > 0
    error_message = "fallback_model must not be empty."
  }
}

variable "notification_channel_ids" {
  description = "Optional Cloud Monitoring notification channel IDs for operational alerts."
  type        = list(string)
  default     = []
}

variable "region" {
  description = "Primary GCP region for regional resources."
  type        = string
  default     = "europe-west1"
}

variable "name_prefix" {
  description = "Prefix used for names that must be unique within the project."
  type        = string
  default     = "dmbok-compass"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{0,30}[a-z0-9]$", var.name_prefix))
    error_message = "name_prefix must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "corpus_bucket_name" {
  description = "Globally unique bucket name for the source PDF and immutable ingestion artifacts."
  type        = string
}

variable "corpus_object_name" {
  description = "Approved PDF object path used by the ingestion job."
  type        = string
  default     = "corpus/dmbok.pdf"
}

variable "artifact_repository_id" {
  description = "Artifact Registry repository ID for immutable application images."
  type        = string
  default     = "dmbok-compass"
}

variable "hosting_site_id" {
  description = "Firebase Hosting site ID used by the frontend deployment."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,29}[a-z0-9]$", var.hosting_site_id))
    error_message = "hosting_site_id must be a valid Firebase Hosting site ID."
  }
}

variable "github_repository" {
  description = "GitHub owner/repository allowed to exchange OIDC tokens for the CI service account."
  type        = string
  default     = "issacamara/dmbok-compass"

  validation {
    condition     = can(regex("^[^/]+/[^/]+$", var.github_repository))
    error_message = "github_repository must be in owner/repository form."
  }
}

variable "github_branch" {
  description = "Protected GitHub branch permitted to authenticate for this project."
  type        = string
  default     = "development"

  validation {
    condition     = can(regex("^(development|main)$", var.github_branch))
    error_message = "github_branch must be development or main."
  }
}

variable "github_environment" {
  description = "GitHub Environment claim required for workload identity exchange."
  type        = string
  default     = "development"

  validation {
    condition     = can(regex("^(development|production)$", var.github_environment))
    error_message = "github_environment must be development or production."
  }
}

variable "github_deploy_workflow" {
  description = "GitHub Actions workflow name allowed to use the application build identity."
  type        = string
  default     = "deploy-development"

  validation {
    condition     = can(regex("^deploy-(development|production)$", var.github_deploy_workflow))
    error_message = "github_deploy_workflow must be deploy-development or deploy-production."
  }
}

variable "secret_names" {
  description = "Secret Manager metadata to provision; values are created out of band."
  type        = set(string)
  default = [
    "answer-provider-api-key",
  ]

  validation {
    condition     = alltrue([for name in var.secret_names : can(regex("^[a-zA-Z0-9][a-zA-Z0-9_-]{0,254}$", name))])
    error_message = "Secret names must use Secret Manager's supported characters and length."
  }
}

variable "runtime_secret_version" {
  description = "Pinned Secret Manager version shared by the API and evaluation job runtime."
  type        = string

  validation {
    condition     = can(regex("^[1-9][0-9]*$", trimspace(var.runtime_secret_version)))
    error_message = "runtime_secret_version must be a positive Secret Manager version number; latest is not allowed."
  }
}
