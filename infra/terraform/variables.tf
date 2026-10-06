variable "project_id" {
  description = "The GCP project that owns the DMBOK Compass foundation."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,28}[a-z0-9]$", var.project_id))
    error_message = "project_id must be a valid GCP project ID."
  }
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

variable "artifact_repository_id" {
  description = "Artifact Registry repository ID for immutable application images."
  type        = string
  default     = "dmbok-compass"
}

variable "hosting_site_id" {
  description = "Firebase Hosting site ID used by the frontend deployment."
  type        = string
  default     = "dmbok-compass"
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
