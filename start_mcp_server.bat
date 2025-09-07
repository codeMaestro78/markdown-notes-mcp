@echo off
REM start_mcp_server.bat - Quick launcher for MCP server

echo 🚀 Starting Markdown Notes MCP Server...
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo ❌ Virtual environment not found!
    echo Please run: python -m venv .venv
    echo Then: .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if index files exist
if not exist "notes_index.npz" (
    echo ⚠️  Index files not found. Building index...
    call .venv\Scripts\python.exe build_index.py ./notes
    if errorlevel 1 (
        echo ❌ Failed to build index!
        pause
        exit /b 1
    )
)

REM Start the MCP server
echo ✅ Starting MCP server...
echo 📁 Notes root: ./notes
echo 📊 Index file: notes_index.npz
echo 📋 Meta file: notes_meta.json
echo.
echo 💡 The server is now running and ready for Copilot integration!
echo 💡 Press Ctrl+C to stop the server
echo.

.venv\Scripts\python.exe notes_mcp_server.py ^
    --index notes_index.npz ^
    --meta notes_meta.json ^
    --notes_root ./notes

echo.
echo 👋 MCP server stopped.
pause
