# MCP Server Launcher for GitHub Copilot Integration
# This script starts the MCP server with proper environment variables

Write-Host "🚀 Starting MCP Server for GitHub Copilot..." -ForegroundColor Green
Write-Host ""

# Set environment variables
$env:MCP_ENVIRONMENT = "development"
$env:MCP_MODEL_NAME = "all-MiniLM-L6-v2"
$env:MCP_NOTES_ROOT = "./notes"
$env:MCP_INDEX_FILE = "notes_index.npz"
$env:MCP_META_FILE = "notes_meta.json"
$env:MCP_CONFIG_DIR = "./config"
$env:PYTHONPATH = $PSScriptRoot

# Change to the script directory
Set-Location $PSScriptRoot

# Start the MCP server
Write-Host "Starting server with configuration:" -ForegroundColor Yellow
Write-Host "  Environment: $env:MCP_ENVIRONMENT"
Write-Host "  Model: $env:MCP_MODEL_NAME"
Write-Host "  Notes Root: $env:MCP_NOTES_ROOT"
Write-Host "  Index File: $env:MCP_INDEX_FILE"
Write-Host "  Meta File: $env:MCP_META_FILE"
Write-Host ""

# Run the MCP server
& python notes_mcp_server.py `
    --index $env:MCP_INDEX_FILE `
    --meta $env:MCP_META_FILE `
    --notes_root $env:MCP_NOTES_ROOT

Write-Host ""
Write-Host "MCP Server stopped." -ForegroundColor Red
Read-Host "Press Enter to exit"
