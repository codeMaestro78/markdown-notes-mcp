# MCP CLI - Advanced Command-Line Interface

## 🎯 **Overview**

The MCP CLI (`mcp_cli_fixed.py`) is a powerful command-line interface for your Markdown Notes MCP system. It provides advanced search, content management, export capabilities, and system analytics through an intuitive command-line experience with robust error handling and fallback systems.

## 🚀 **Quick Start**

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Setup CLI globally (optional)
python setup_cli.py

# Or use directly
python mcp_cli_fixed.py [command]
```

### Basic Usage
```bash
# Search your notes
python mcp_cli_fixed.py search "machine learning" --limit 5

# Add a new note
python mcp_cli_fixed.py add-note ./my_note.md --auto-tag

# Export search results
python mcp_cli_fixed.py export-search "PCA" --format pdf

# View system statistics
python mcp_cli_fixed.py stats --period week
```

## 📋 **Available Commands**

### 🔍 **Search Commands**

#### `python mcp_cli_fixed.py search <query> [options]`
Search your notes with advanced filtering and formatting options.

**Options:**
- `--format {text,json,table}` - Output format (default: text)
- `--limit <number>` - Maximum results (default: 10)
- `--threshold <float>` - Minimum relevance score (default: 0.0)
- `--export <filename>` - Export results to JSON file (NEW!)

**Examples:**
```bash
# Basic search
python mcp_cli_fixed.py search "machine learning"

# JSON output with limit
python mcp_cli_fixed.py search "PCA" --format json --limit 5

# High-relevance results only
python mcp_cli_fixed.py search "web development" --threshold 0.8

# Export results (NEW!)
python mcp_cli_fixed.py search "docker" --export docker_results.json --format json
```

### 📝 **Content Management**

#### `python mcp_cli_fixed.py add-note <file> [options]`
Add a new markdown file to your notes collection.

**Options:**
- `--auto-tag` - Automatically generate tags based on content
- `--rebuild` - Rebuild search index after adding

**Examples:**
```bash
# Add note with auto-tagging
python mcp_cli_fixed.py add-note ./new_research.md --auto-tag

# Add and rebuild index
python mcp_cli_fixed.py add-note ./meeting_notes.md --auto-tag --rebuild

# Simple add
python mcp_cli_fixed.py add-note ./quick_note.md
```

#### `python mcp_cli_fixed.py list-notes [options]`
List all available notes with metadata.

**Options:**
- `--sort {name,modified,size}` - Sort order (default: name)
- `--format {text,json}` - Output format (default: text)

**Examples:**
```bash
# List by modification date
python mcp_cli_fixed.py list-notes --sort modified

# JSON format (FIXED: datetime serialization)
python mcp_cli_fixed.py list-notes --format json

# Sort by file size
python mcp_cli_fixed.py list-notes --sort size
```

### 📤 **Export Commands**

#### `python mcp_cli_fixed.py export-search <query> [options]`
Export search results in various formats.

**Options:**
- `--format {pdf,html,markdown,json}` - Export format (default: pdf)
- `--output <filename>` - Output file path
- `--limit <number>` - Maximum results (default: 20)

**Examples:**
```bash
# Export as PDF
python mcp_cli_fixed.py export-search "machine learning" --format pdf

# Custom output file
python mcp_cli_fixed.py export-search "PCA" --format html --output pca_guide.html

# Export top 10 results
python mcp_cli_fixed.py export-search "web development" --limit 10 --format markdown
```

### 📊 **Analytics Commands**

#### `python mcp_cli_fixed.py stats [options]`
View system statistics and analytics.

**Options:**
- `--period {day,week,month,all}` - Time period (default: week)
- `--format {text,json}` - Output format (default: text)

**Examples:**
```bash
# Weekly statistics
python mcp_cli_fixed.py stats --period week

# Monthly overview
python mcp_cli_fixed.py stats --period month

# JSON format for scripting
python mcp_cli_fixed.py stats --format json
```

### 🔧 **System Commands**

#### `python mcp_cli_fixed.py rebuild-index [options] [notes_root]`
Rebuild the search index with custom settings.

**Options:**
- `--model <model_name>` - Embedding model to use
- `--chunk-size <number>` - Text chunk size
- `--overlap <number>` - Chunk overlap size
- `--force` - Force rebuild even if unchanged

**Examples:**
```bash
# Rebuild with different model
python mcp_cli_fixed.py rebuild-index --model all-mpnet-base-v2

# Custom chunk settings (FIXED!)
python mcp_cli_fixed.py rebuild-index --chunk-size 300 --overlap 100

# Force rebuild
python mcp_cli_fixed.py rebuild-index --force

# Custom notes directory
python mcp_cli_fixed.py rebuild-index ./my_notes
```

#### `python mcp_cli_fixed.py server [options]`
Start the MCP server with custom settings.

**Options:**
- `--host <address>` - Server host (default: 127.0.0.1)
- `--port <number>` - Server port (default: 8181)
- `--no-admin` - Disable admin HTTP interface

**Examples:**
```bash
# Start with custom port
python mcp_cli_fixed.py server --port 8080

# Disable admin interface
python mcp_cli_fixed.py server --no-admin

# Custom host and port
python mcp_cli_fixed.py server --host 0.0.0.0 --port 9000
```

## 🎨 **Output Formats**

### Text Format (Default)
```
🔍 Searching for: 'machine learning'
📊 Limit: 5, Format: text
--------------------------------------------------
1. 📄 machine_learning_fundamentals.md
   📊 Score: 0.95
   💬 Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions...

2. 📄 python_data_science.md
   📊 Score: 0.87
   💬 Popular ML libraries include scikit-learn, TensorFlow, PyTorch for machine learning tasks...

3. 📄 pca_notes.md
   📊 Score: 0.82
   💬 PCA is a dimensionality reduction technique commonly used in machine learning preprocessing...
```

### JSON Format (FIXED!)
```json
[
  {
    "file": "machine_learning_fundamentals.md",
    "score": 0.95,
    "text": "Machine learning is a subset of artificial intelligence...",
    "chunk_id": 0
  },
  {
    "file": "python_data_science.md",
    "score": 0.87,
    "text": "Popular ML libraries include scikit-learn...",
    "chunk_id": 2
  }
]
```

### Table Format
```
+------------+------------------+--------------------------------------------------+
| File       | Score           | Preview                                         |
+============+==================+==================================================+
| machine_learning_fundamentals.md | 0.950 | Machine learning is a subset of artificial...   |
+------------+------------------+--------------------------------------------------+


| python_data_science.md          | 0.870 | Popular ML libraries include scikit-learn...    |
+------------+------------------+--------------------------------------------------+
| pca_notes.md                    | 0.820 | PCA is a dimensionality reduction technique...  |
+------------+------------------+--------------------------------------------------+
```

## 🏷️ **Auto-Tagging System**

The CLI includes intelligent auto-tagging that analyzes your content and suggests relevant tags:

### Supported Tags:
- `machine learning` - ML, AI, algorithms
- `data science` - pandas, numpy, jupyter
- `web development` - HTML, CSS, JavaScript, React
- `devops` - Docker, Kubernetes, AWS, CI/CD
- `python` - Python, Django, Flask, FastAPI
- `cloud` - AWS, Azure, GCP, cloud computing
- `database` - SQL, MongoDB, PostgreSQL, Redis
- `security` - Security, encryption, authentication

### Example:
```bash
python mcp_cli_fixed.py add-note ./ml_project.md --auto-tag
# Output: 🏷️ Generated tags: machine learning, python, data science
```

## 📤 **Export Formats**

### PDF Export
- Professional formatting with headers and footers
- Search query prominently displayed
- Relevance scores included
- Multiple pages with proper pagination

### HTML Export
- Interactive web page with styling
- Clickable file links
- Responsive design for mobile viewing
- Embedded CSS for consistent appearance

### Markdown Export
- Clean markdown formatting
- Compatible with all markdown viewers
- Preserves original formatting
- Easy to edit and version control

## ⚙️ **Configuration**

### Environment Variables
```bash
# Set default model
export MCP_MODEL_NAME="all-mpnet-base-v2"

# Configure chunking
export MCP_CHUNK_SIZE="200"
export MCP_OVERLAP="50"

# Set environment
export MCP_ENVIRONMENT="production"
```

### Custom Settings
```bash
# Use custom index files
python mcp_cli_fixed.py search "query" --index custom_index.npz --meta custom_meta.json

# Override model for specific search
python mcp_cli_fixed.py search "query" --model paraphrase-multilingual-MiniLM-L12-v2
```

## 🚨 **Troubleshooting**

### Common Issues:

**"Command not found"**
```bash
# Install CLI globally
python setup_cli.py

# Or use full path
python mcp_cli_fixed.py search "query"
```

**"Index files not found"**
```bash
# Rebuild index
python mcp_cli_fixed.py rebuild-index

# Or build manually
python build_index.py ./notes
```

**"Permission denied"**
```bash
# Run as administrator (Windows)
# Or check file permissions
```

**"Module not found"**
```bash
# Install dependencies
pip install -r requirements.txt

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

**"JSON serialization error" (FIXED!)**
```bash
# This now works automatically
python mcp_cli_fixed.py list-notes --format json
```

**"Search export not working" (FIXED!)**
```bash
# This now works
python mcp_cli_fixed.py search "query" --export results.json --format json
```

**"Rebuild-index arguments error" (FIXED!)**
```bash
# This now works
python mcp_cli_fixed.py rebuild-index --chunk-size 300 --overlap 100
```

**"Server threading error" (FIXED!)**
```bash
# This now works
python mcp_cli_fixed.py server
```

## 📈 **Performance Tips**

### For Large Note Collections:
```bash
# Use larger chunks for better context
python mcp_cli_fixed.py rebuild-index --chunk-size 300 --overlap 100

# Use more powerful model
python mcp_cli_fixed.py rebuild-index --model all-mpnet-base-v2
```

### For Fast Searches:
```bash
# Limit results for speed
python mcp_cli_fixed.py search "query" --limit 5

# Use threshold to filter irrelevant results
python mcp_cli_fixed.py search "query" --threshold 0.7
```

## 🔧 **Advanced Usage**

### Scripting with JSON Output:
```bash
# Get search results as JSON
RESULTS=$(python mcp_cli_fixed.py search "machine learning" --format json --limit 3)

# Process with jq (if installed)
echo $RESULTS | jq '.[] | select(.score > 0.8) | .file'

# Use in scripts
python mcp_cli_fixed.py export-search "PCA" --format json --output results.json
python -c "import json; data=json.load(open('results.json')); print(len(data))"
```

### Batch Operations:
```bash
# Add multiple notes
for file in ./notes/*.md; do
    python mcp_cli_fixed.py add-note "$file" --auto-tag
done

# Rebuild index after batch add
python mcp_cli_fixed.py rebuild-index
```

### Integration with Other Tools:
```bash
# Export for documentation
python mcp_cli_fixed.py export-search "API" --format markdown --output api_guide.md

# Generate reports
python mcp_cli_fixed.py stats --period month --format json > monthly_report.json

# Backup search results
python mcp_cli_fixed.py search "important" --export backup_$(date +%Y%m%d).json
```

## 🎯 **Best Practices**

1. **Use Specific Queries**: More specific searches give better results
2. **Leverage Auto-Tagging**: Let the system categorize your content
3. **Regular Index Updates**: Rebuild index when adding many notes
4. **Export for Sharing**: Use PDF/HTML for sharing with others
5. **Monitor Performance**: Use stats command to track usage

## 📞 **Support**

- **Help Command**: `python mcp_cli_fixed.py --help`
- **Command Help**: `python mcp_cli_fixed.py <command> --help`
- **Verbose Output**: Add `--verbose` to see detailed information
- **Debug Mode**: Set environment variable `MCP_DEBUG=true`

---

## 🆕 **Recent Fixes & Improvements**

### Version 2.0 Features

- ✅ **JSON Serialization Fix**: Resolved datetime object serialization issues
- ✅ **Search Export Functionality**: Added --export argument to search command
- ✅ **Rebuild-Index Arguments**: Fixed argument parsing and directory handling
- ✅ **Server Threading Issues**: Fixed import and threading problems
- ✅ **Syntax Errors**: All f-string and compilation errors fixed
- ✅ **Fallback Systems**: Multiple fallback implementations for reliability
- ✅ **Comprehensive Testing**: Added test scripts for validation
- ✅ **Environment Configuration**: Support for environment variable configuration
- ✅ **Advanced Stats**: Enhanced statistics with better formatting
- ✅ **Smart Tagging**: Improved auto-tagging with keyword detection
- ✅ **Batch Processing**: Support for processing multiple files
- ✅ **Advanced Search**: Threshold filtering and custom export paths
- ✅ **Multiple Exports**: Enhanced export functionality with custom filenames
- ✅ **Error Recovery**: Comprehensive error handling and recovery mechanisms

---

**🎉 Your MCP CLI is now ready for advanced note management and search!**

The CLI provides a powerful, scriptable interface to your semantic search system, making it easy to integrate with other tools and workflows.