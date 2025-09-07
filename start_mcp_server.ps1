# MCP Server Launcher for GitHub Copilot Integration
# This script starts the MCP server with proper environment variables

Write-Host "🚀 Starting Markdown Notes MCP Server..." -ForegroundColor Green
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run:"
    Write-Host "  python -m venv .venv"
    Write-Host "  .venv\Scripts\pip install -r requirements.txt"
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if index files exist
if (-not (Test-Path "notes_index.npz")) {
    Write-Host "⚠️  Index files not found. Building index..." -ForegroundColor Yellow
    & ".venv\Scripts\python.exe" build_index.py ./notes
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to build index!" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Start the MCP server
Write-Host "✅ Starting MCP server..." -ForegroundColor Green
Write-Host "📁 Notes root: ./notes" -ForegroundColor Cyan
Write-Host "📊 Index file: notes_index.npz" -ForegroundColor Cyan
Write-Host "📋 Meta file: notes_meta.json" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 The server is now running and ready for Copilot integration!" -ForegroundColor Green
Write-Host "💡 Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

try {
    & ".venv\Scripts\python.exe" notes_mcp_server.py `
        --index notes_index.npz `
        --meta notes_meta.json `
        --notes_root ./notes
} catch {
    Write-Host "❌ Error starting MCP server: $_" -ForegroundColor Red
} finally {
    Write-Host ""
    Write-Host "👋 MCP server stopped." -ForegroundColor Blue
    Read-Host "Press Enter to exit"
}
