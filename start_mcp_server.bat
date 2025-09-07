@echo off
REM MCP Server Launcher for GitHub Copilot Integration
REM This script starts the MCP server with proper environment variables

echo 🚀 Starting MCP Server for GitHub Copilot...
echo.

REM Set environment variables
set MCP_ENVIRONMENT=development
set MCP_MODEL_NAME=all-MiniLM-L6-v2
set MCP_NOTES_ROOT=./notes
set MCP_INDEX_FILE=notes_index.npz
set MCP_META_FILE=notes_meta.json
set MCP_CONFIG_DIR=./config
set PYTHONPATH=%~dp0

REM Change to the script directory
cd /d "%~dp0"

REM Start the MCP server
echo Starting server with configuration:
echo   Environment: %MCP_ENVIRONMENT%
echo   Model: %MCP_MODEL_NAME%
echo   Notes Root: %MCP_NOTES_ROOT%
echo   Index File: %MCP_INDEX_FILE%
echo   Meta File: %MCP_META_FILE%
echo.

python notes_mcp_server.py ^
    --index "%MCP_INDEX_FILE%" ^
    --meta "%MCP_META_FILE%" ^
    --notes_root "%MCP_NOTES_ROOT%"

echo.
echo MCP Server stopped.
pause
