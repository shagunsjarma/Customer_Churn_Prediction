output "acr_login_server" {
  description = "The login server for the Azure Container Registry"
  value       = azurerm_container_registry.acr.login_server
}

output "container_app_url" {
  description = "The URL of the deployed Container App"
  value       = azurerm_container_app.ca.latest_revision_fqdn
}
