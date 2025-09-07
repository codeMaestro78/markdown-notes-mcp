#!/usr/bin/env python3
"""
Git Setup Script for Markdown Notes MCP Project
This script initializes git repository and adds all project files
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status"""
    try:
        print(f"🔧 {description}...")
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=project_root)

        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False

    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False

def setup_git():
    """Set up git repository for the project"""
    print("🚀 Setting up Git Repository for Markdown Notes MCP")
    print("=" * 60)

    # Check if git is installed
    if not run_command("git --version", "Checking Git installation"):
        print("❌ Git is not installed. Please install Git first.")
        return False

    # Initialize git repository
    if not run_command("git init", "Initializing Git repository"):
        return False

    # Add .gitignore
    gitignore_content = """# Python
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
"""

    gitignore_path = project_root / ".gitignore"
    with open(gitignore_path, 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    print("✅ Created .gitignore file")

    # Add all files
    if not run_command("git add .", "Adding all files to Git"):
        return False

    # Initial commit
    commit_message = """Initial commit: Markdown Notes MCP with advanced CLI features

- Advanced CLI with semantic search capabilities
- MCP server integration for GitHub Copilot
- Multiple export formats (PDF, HTML, Markdown, JSON)
- Auto-tagging system for content categorization
- Comprehensive error handling and fallback systems
- Multi-model support with Sentence Transformers
- Environment-based configuration management
- Batch processing capabilities
- Performance optimization features
- Complete documentation and testing suite"""

    if not run_command(f'git commit -m "{commit_message}"', "Creating initial commit"):
        return False

    # Show status
    print("\n📊 Git Repository Status:")
    run_command("git status --short", "Checking repository status")

    print("\n📝 Git Log:")
    run_command("git log --oneline -5", "Showing recent commits")

    print("\n🎉 Git setup completed successfully!")
    print("\n💡 Next steps:")
    print("   1. Create a repository on GitHub/GitLab")
    print("   2. Add remote: git remote add origin <repository-url>")
    print("   3. Push: git push -u origin main")
    print("   4. Continue development with regular commits")

    return True

def create_git_scripts():
    """Create additional git helper scripts"""
    print("\n🔧 Creating Git Helper Scripts...")

    # Create git status script
    status_script = """#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def run_git_status():
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, cwd=Path(__file__).parent)
        if result.returncode == 0:
            if result.stdout.strip():
                print("📝 Uncommitted changes:")
                print(result.stdout)
            else:
                print("✅ Working directory is clean")
        else:
            print("❌ Git status failed")
            print(result.stderr)
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    run_git_status()
"""

    with open(project_root / "git_status.py", 'w', encoding='utf-8') as f:
        f.write(status_script)
    print("✅ Created git_status.py helper script")

    # Create commit script
    commit_script = """#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def commit_changes(message=None):
    if not message:
        print("❌ Commit message required")
        print("Usage: python git_commit.py 'Your commit message'")
        sys.exit(1)

    try:
        # Add all changes
        subprocess.run(['git', 'add', '.'], cwd=Path(__file__).parent, check=True)

        # Commit with message
        subprocess.run(['git', 'commit', '-m', message], cwd=Path(__file__).parent, check=True)

        print(f"✅ Committed: {message}")

        # Show status
        result = subprocess.run(['git', 'status', '--short'], capture_output=True, text=True, cwd=Path(__file__).parent)
        print("\\n📊 Current status:")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"❌ Git command failed: {e}")
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        commit_changes(sys.argv[1])
    else:
        commit_changes()
"""

    with open(project_root / "git_commit.py", 'w', encoding='utf-8') as f:
        f.write(commit_script)
    print("✅ Created git_commit.py helper script")

if __name__ == "__main__":
    project_root = Path(__file__).parent

    print(f"📁 Project directory: {project_root}")
    print(f"📂 Files to add: {len(list(project_root.glob('**/*')))} total files")

    if setup_git():
        create_git_scripts()
        print("\n🎯 Git Setup Complete!")
        print("\n📋 Summary:")
        print("   ✅ Git repository initialized")
        print("   ✅ All files added and committed")
        print("   ✅ .gitignore created")
        print("   ✅ Helper scripts created")
        print("   ✅ Ready for remote repository setup")
    else:
        print("\n❌ Git setup failed. Please check the errors above.")
        sys.exit(1)