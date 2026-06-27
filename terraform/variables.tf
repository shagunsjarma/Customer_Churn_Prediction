variable "location" {
  description = "Azure region to deploy resources"
  type        = string
  default     = "East US"
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
  default     = "rg-churn-prediction"
}

variable "acr_name" {
  description = "Name of the Azure Container Registry"
  type        = string
  default     = "acrchurnprediction"
}

variable "container_app_env_name" {
  description = "Name of the Container App Environment"
  type        = string
  default     = "cae-churn-prediction"
}

variable "container_app_name" {
  description = "Name of the Container App"
  type        = string
  default     = "ca-churn-prediction"
}

variable "image_tag" {
  description = "Tag for the Docker image"
  type        = string
  default     = "latest"
}
