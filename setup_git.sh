#!/usr/bin/env bash
# Git Setup Commands for Markdown Notes MCP Project
# Run these commands in your project directory

echo "🚀 Setting up Git for Markdown Notes MCP Project"
echo "================================================="

# Check if we're in the right directory
if [ ! -f "mcp_cli_fixed.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   cd C:\Users\Devarshi\PycharmProjects\markdown-notes-mcp"
    exit 1
fi

echo "📁 Current directory: $(pwd)"
echo "📂 Files found: $(find . -type f -not -path './.git/*' -not -path './.venv/*' | wc -l) files"
echo ""

# Initialize git repository
echo "🔧 Step 1: Initialize Git repository"
git init
echo ""

# Create .gitignore
echo "🔧 Step 2: Create .gitignore file"
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.venv/
venv/
ENV/
env/

# IDE
.vscode/settings.json
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
logs/

# Temporary files
*.tmp
*.temp
.cache/

# MCP specific
notes_index.npz
notes_meta.json
.copilot/
search_results*.*
exported_*.*
backup_*.*
EOF
echo "✅ Created .gitignore"
echo ""

# Add all files
echo "🔧 Step 3: Add all files to Git"
git add .
echo ""

# Check status
echo "📊 Step 4: Check Git status"
git status --short
echo ""

# Initial commit
echo "🔧 Step 5: Create initial commit"
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
echo ""

# Show final status
echo "📊 Final Git Status:"
git status
echo ""

echo "📝 Git Log:"
git log --oneline -3
echo ""

echo "🎉 Git setup completed successfully!"
echo ""
echo "💡 Next steps:"
echo "   1. Create a repository on GitHub/GitLab"
echo "   2. Add remote: git remote add origin <repository-url>"
echo "   3. Push: git push -u origin main"
echo "   4. Continue development with regular commits"
echo ""
echo "🔧 Useful Git commands:"
echo "   git status              # Check current status"
echo "   git add .              # Add all changes"
echo "   git commit -m 'msg'    # Commit changes"
echo "   git push               # Push to remote"
echo "   git pull               # Pull from remote"
echo "   git log --oneline      # View commit history"