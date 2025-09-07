@echo off
REM Git Setup Batch Script for Markdown Notes MCP Project
REM Run this script to set up Git for your project

echo 🚀 Setting up Git for Markdown Notes MCP Project
echo =================================================

REM Check if we're in the right directory
if not exist "mcp_cli_fixed.py" (
    echo ❌ Error: Please run this script from the project root directory
    echo    cd C:\Users\Devarshi\PycharmProjects\markdown-notes-mcp
    pause
    exit /b 1
)

echo 📁 Current directory: %CD%
for /f %%A in ('dir /b /a-d ^| find /c /v ""') do set FILE_COUNT=%%A
echo 📂 Files found: %FILE_COUNT% files
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git is not installed. Please install Git first from https://git-scm.com/
    pause
    exit /b 1
)
echo ✅ Git is installed
echo.

REM Initialize git repository
echo 🔧 Step 1: Initialize Git repository
git init
if errorlevel 1 (
    echo ❌ Failed to initialize Git repository
    pause
    exit /b 1
)
echo ✅ Git repository initialized
echo.

REM Create .gitignore
echo 🔧 Step 2: Create .gitignore file
(
echo # Python
echo __pycache__/
echo *.py[cod]
echo *$py.class
echo *.so
echo .Python
echo build/
echo develop-eggs/
echo dist/
echo downloads/
echo eggs/
echo .eggs/
echo lib/
echo lib64/
echo parts/
echo sdist/
echo var/
echo wheels/
echo *.egg-info/
echo .installed.cfg
echo *.egg
echo MANIFEST
echo.
echo # Virtual environments
echo .venv/
echo venv/
echo ENV/
echo env/
echo.
echo # IDE
echo .vscode/settings.json
echo .idea/
echo *.swp
echo *.swo
echo *~
echo.
echo # OS
echo .DS_Store
echo .DS_Store?
echo ._*
echo .Spotlight-V100
echo .Trashes
echo ehthumbs.db
echo Thumbs.db
echo.
echo # Logs
echo *.log
echo logs/
echo.
echo # Temporary files
echo *.tmp
echo *.temp
echo .cache/
echo.
echo # MCP specific
echo notes_index.npz
echo notes_meta.json
echo .copilot/
echo search_results*.*
echo exported_*.*
echo backup_*.*
) > .gitignore
echo ✅ Created .gitignore
echo.

REM Add all files
echo 🔧 Step 3: Add all files to Git
git add .
if errorlevel 1 (
    echo ❌ Failed to add files to Git
    pause
    exit /b 1
)
echo ✅ All files added to Git
echo.

REM Check status
echo 📊 Step 4: Check Git status
git status --short
echo.

REM Initial commit
echo 🔧 Step 5: Create initial commit
git commit -m "Initial commit: Markdown Notes MCP with advanced CLI features

- Advanced CLI with semantic search capabilities
- MCP server integration for GitHub Copilot
- Multiple export formats (PDF, HTML, Markdown, JSON)
- Auto-tagging system for content categorization
- Comprehensive error handling and fallback systems
- Multi-model support with Sentence Transformers
- Environment-based configuration management
- Batch processing capabilities
- Performance optimization features
- Complete documentation and testing suite

Version 2.0 - All fixes applied and production-ready"
if errorlevel 1 (
    echo ❌ Failed to create initial commit
    pause
    exit /b 1
)
echo ✅ Initial commit created
echo.

REM Show final status
echo 📊 Final Git Status:
git status
echo.

echo 📝 Git Log:
git log --oneline -3
echo.

echo 🎉 Git setup completed successfully!
echo.
echo 💡 Next steps:
echo    1. Create a repository on GitHub/GitLab
echo    2. Add remote: git remote add origin ^<repository-url^>
echo    3. Push: git push -u origin main
echo    4. Continue development with regular commits
echo.
echo 🔧 Useful Git commands:
echo    git status              # Check current status
echo    git add .              # Add all changes
echo    git commit -m "msg"    # Commit changes
echo    git push               # Push to remote
echo    git pull               # Pull from remote
echo    git log --oneline      # View commit history
echo.

pause