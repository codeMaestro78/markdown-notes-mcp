# 🚀 **Complete Git Setup Guide for Markdown Notes MCP**

## 📋 **Step-by-Step Git Setup**

### **Navigate to Your Project Directory**
```bash
cd C:\Users\Devarshi\PycharmProjects\markdown-notes-mcp
```

---

## 🔧 **Option 1: Automated Setup (Recommended)**

### **Run the Python Setup Script**
```bash
python setup_git.py
```

**This will:**
- ✅ Initialize Git repository
- ✅ Create comprehensive `.gitignore`
- ✅ Add all project files
- ✅ Create initial commit with detailed message
- ✅ Show repository status
- ✅ Create helper scripts

---

## 🔧 **Option 2: Manual Setup**

### **Step 1: Initialize Git Repository**
```bash
git init
```

### **Step 2: Create .gitignore File**
```bash
# Create .gitignore with the following content:
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
```

### **Step 3: Add All Files**
```bash
git add .
```

### **Step 4: Check Status**
```bash
git status
```

### **Step 5: Create Initial Commit**
```bash
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
```

---

## 📊 **Check Your Git Setup**

### **View Repository Status**
```bash
git status
```

### **View Commit History**
```bash
git log --oneline
```

### **View Repository Information**
```bash
git remote -v
git branch -a
```

---

## 🌐 **Connect to Remote Repository**

### **Create Repository on GitHub/GitLab**
1. Go to GitHub.com or GitLab.com
2. Click "New Repository"
3. Name: `markdown-notes-mcp`
4. Description: `Advanced CLI for Markdown Notes with MCP integration`
5. Choose Public/Private
6. **Don't initialize with README** (we already have one)

### **Add Remote Origin**
```bash
# Replace with your actual repository URL
git remote add origin https://github.com/yourusername/markdown-notes-mcp.git
```

### **Push to Remote**
```bash
git push -u origin main
```

---

## 🔄 **Daily Git Workflow**

### **Check Status**
```bash
git status
```

### **Add Changes**
```bash
# Add specific file
git add filename.py

# Add all changes
git add .

# Add with interactive mode
git add -i
```

### **Commit Changes**
```bash
# Simple commit
git commit -m "Add new feature"

# Commit with detailed message
git commit -m "Add search export functionality

- Added --export flag to search command
- Support for JSON export format
- Fixed datetime serialization issues
- Added comprehensive error handling"
```

### **Push Changes**
```bash
# Push current branch
git push

# Push to specific remote/branch
git push origin main
```

### **Pull Latest Changes**
```bash
git pull origin main
```

---

## 🌿 **Branch Management**

### **Create New Branch**
```bash
git checkout -b feature/new-feature
```

### **Switch Branches**
```bash
git checkout main
git checkout feature/new-feature
```

### **List Branches**
```bash
git branch -a
```

### **Merge Branch**
```bash
git checkout main
git merge feature/new-feature
```

### **Delete Branch**
```bash
# Delete local branch
git branch -d feature/completed-feature

# Delete remote branch
git push origin --delete feature/completed-feature
```

---

## 🔍 **Advanced Git Commands**

### **View Detailed Log**
```bash
# Last 10 commits with details
git log --oneline -10

# Commits by author
git log --author="Your Name"

# Commits in date range
git log --since="2024-01-01" --until="2024-12-31"
```

### **View File Changes**
```bash
# Changes in working directory
git diff

# Changes in staging area
git diff --staged

# Changes between commits
git diff HEAD~1 HEAD

# Changes in specific file
git diff filename.py
```

### **Undo Changes**
```bash
# Unstage file
git reset HEAD filename.py

# Discard working directory changes
git checkout -- filename.py

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```

### **Stash Changes**
```bash
# Stash current changes
git stash

# List stashes
git stash list

# Apply latest stash
git stash pop

# Apply specific stash
git stash apply stash@{1}
```

---

## 📊 **Repository Statistics**

### **View Contributors**
```bash
git shortlog -sn
```

### **File Change Statistics**
```bash
git log --pretty=format: --numstat | awk 'NF==3 {plus+=$1; minus+=$2} END {print "Lines added:", plus, "Lines removed:", minus}'
```

### **Repository Size**
```bash
git count-objects -vH
```

---

## 🛠️ **Git Configuration**

### **Set User Information**
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### **Set Default Editor**
```bash
git config --global core.editor "code --wait"
```

### **Set Default Branch**
```bash
git config --global init.defaultBranch main
```

### **View Configuration**
```bash
git config --list
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

**"fatal: not a git repository"**
```bash
# Initialize git repository
git init
```

**"fatal: remote origin already exists"**
```bash
# Remove existing remote
git remote remove origin

# Add new remote
git remote add origin <new-url>
```

**"error: failed to push some refs"**
```bash
# Pull latest changes first
git pull origin main --rebase

# Then push
git push origin main
```

**"Changes not staged for commit"**
```bash
# Add all changes
git add .

# Or add specific files
git add filename.py
```

---

## 📋 **Git Best Practices**

### **Commit Messages**
- Use imperative mood: "Add feature" not "Added feature"
- Keep first line under 50 characters
- Add detailed description for complex changes
- Reference issue numbers: "Fix #123"

### **Branch Strategy**
- `main`/`master`: Production-ready code
- `develop`: Integration branch
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical fixes

### **Workflow**
1. Pull latest changes: `git pull`
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes and commit: `git add . && git commit -m "message"`
4. Push branch: `git push origin feature/new-feature`
5. Create Pull Request
6. Merge after review

---

## 🎯 **Quick Reference**

### **Most Used Commands**
```bash
git status              # Check status
git add .              # Add all changes
git commit -m "msg"    # Commit changes
git push               # Push to remote
git pull               # Pull from remote
git log --oneline      # View history
git diff               # View changes
```

### **Emergency Commands**
```bash
git reset --hard HEAD  # Discard all changes
git clean -fd          # Remove untracked files
git reflog             # View reference log
```

---

## 🎉 **Your Project is Now in Git!**

**All your files are now tracked in Git with:**
- ✅ Complete version history
- ✅ Professional .gitignore
- ✅ Ready for collaboration
- ✅ Backup and recovery capabilities
- ✅ Branch management for features
- ✅ Integration with GitHub/GitLab

**Next: Create a repository on GitHub and push your code!** 🚀

**Run: `python setup_git.py` to get started automatically!** 🎯