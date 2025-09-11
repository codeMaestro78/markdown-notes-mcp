# Markdown Notes MCP CLI

A powerful command-line interface for managing and searching through markdown notes using advanced semantic search powered by MCP (Model Context Protocol) and machine learning embeddings.

## 🚀 Features

- **Semantic Search**: Find notes using natural language queries with AI-powered relevance scoring
- **Question-Answering**: Ask questions about your notes and get AI-generated answers using Google Gemini
- **Note Generation**: Create new notes based on prompts or existing content
- **Multiple Output Formats**: Text, JSON, Table, PDF, HTML, and Markdown exports
- **Auto-Tagging**: Automatically generate relevant tags for new notes
- **MCP Server Integration**: Ready for GitHub Copilot integration
- **Advanced Statistics**: Comprehensive analytics with customizable time periods
- **Flexible Export**: Export search results in multiple formats with custom filenames
- **Robust Error Handling**: Graceful fallbacks and comprehensive error recovery
- **Batch Operations**: Process multiple notes efficiently
- **Advanced Search Options**: Threshold filtering, export functionality, and custom limits
- **Fallback Systems**: Multiple fallback implementations for maximum reliability
- **Environment Configuration**: Customizable settings via environment variables
- **Comprehensive Testing**: Built-in test scripts for validation

## 📋 Table of Contents

- Installation
- Quick Start
- Commands
- Configuration
- Examples
- Troubleshooting
- Testing
- Contributing
- License

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

1. **Clone or navigate to your project directory:**
   ```bash
   cd C:\Users\Devarshi\PycharmProjects\markdown-notes-mcp
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python mcp_cli_fixed.py --help
   ```

## 🚀 Quick Start

### Basic Usage

```bash
# Search for notes
python mcp_cli_fixed.py search "machine learning" --limit 5

# Add a new note with auto-tagging
python mcp_cli_fixed.py add-note ./my_note.md --auto-tag

# Export search results as PDF
python mcp_cli_fixed.py export-search "PCA" --format pdf

# View weekly statistics
python mcp_cli_fixed.py stats --period week

# Start MCP server for Copilot integration
python mcp_cli_fixed.py server

# Ask questions about your notes
python mcp_cli_fixed.py qa "Explain PCA from my notes"

# Generate new notes
python mcp_cli_fixed.py generate-note "Web development best practices"

# Generate with reference
python mcp_cli_fixed.py generate-note "Advanced ML" --base-note ml_notes.md
```

### Advanced Usage

```bash
# Search with export functionality
python mcp_cli_fixed.py search "cloud computing" --export results.json --format json

# List notes in JSON format (fixed datetime serialization)
python mcp_cli_fixed.py list-notes --format json

# Rebuild index with custom settings
python mcp_cli_fixed.py rebuild-index --chunk-size 300 --overlap 100

# Start server with custom configuration
python mcp_cli_fixed.py server --host 0.0.0.0 --port 8080
```

## 📖 Commands

### Search Command

Search through your markdown notes using semantic search with advanced options.

```bash
python mcp_cli_fixed.py search [OPTIONS] QUERY
```

**Options:**
- `--format {text,json,table}`: Output format (default: text)
- `--limit INTEGER`: Maximum number of results (default: 10)
- `--threshold FLOAT`: Minimum relevance threshold (default: 0.0)
- `--export PATH`: Export results to JSON file

**Examples:**
```bash
# Basic search
python mcp_cli_fixed.py search "machine learning"

# Search with JSON output
python mcp_cli_fixed.py search "python data science" --format json --limit 5

# Search with table format and threshold
python mcp_cli_fixed.py search "algorithms" --format table --threshold 0.8

# Search and export results (NEW!)
python mcp_cli_fixed.py search "cloud computing" --export search_results.json --format json
```

### Add Note Command

Add a new markdown note to your collection with advanced tagging options.

```bash
python mcp_cli_fixed.py add-note [OPTIONS] FILE
```

**Options:**
- `--auto-tag`: Automatically generate tags for the note
- `--rebuild`: Rebuild search index after adding note

**Examples:**
```bash
# Add note without tags
python mcp_cli_fixed.py add-note ./my_note.md

# Add note with auto-tagging
python mcp_cli_fixed.py add-note ./research.md --auto-tag

# Add note and rebuild index
python mcp_cli_fixed.py add-note ./new_findings.md --auto-tag --rebuild
```

### Export Search Command

Export search results in various formats with custom output paths.

```bash
python mcp_cli_fixed.py export-search [OPTIONS] QUERY
```

**Options:**
- `--format {pdf,html,markdown,json}`: Export format (default: pdf)
- `--output PATH`: Output file path
- `--limit INTEGER`: Maximum results to export (default: 20)

**Examples:**
```bash
# Export as PDF
python mcp_cli_fixed.py export-search "machine learning" --format pdf

# Export as HTML with custom output
python mcp_cli_fixed.py export-search "web development" --format html --output web_guide.html

# Export as Markdown
python mcp_cli_fixed.py export-search "docker" --format markdown --output docker_guide.md

# Export as JSON with limit
python mcp_cli_fixed.py export-search "python" --format json --limit 15 --output python_topics.json
```

### Statistics Command

View comprehensive system statistics with multiple time periods.

```bash
python mcp_cli_fixed.py stats [OPTIONS]
```

**Options:**
- `--period {day,week,month,all}`: Time period for statistics (default: week)
- `--format {text,json}`: Output format (default: text)

**Examples:**
```bash
# Weekly statistics
python mcp_cli_fixed.py stats --period week

# Monthly statistics in JSON
python mcp_cli_fixed.py stats --period month --format json

# Daily statistics
python mcp_cli_fixed.py stats --period day

# All-time statistics
python mcp_cli_fixed.py stats --period all
```

### List Notes Command

List all available notes with metadata and sorting options.

```bash
python mcp_cli_fixed.py list-notes [OPTIONS]
```

**Options:**
- `--sort {name,modified,size}`: Sort order (default: name)
- `--format {text,json}`: Output format (default: text)

**Examples:**
```bash
# List notes sorted by name
python mcp_cli_fixed.py list-notes

# List notes sorted by modification date
python mcp_cli_fixed.py list-notes --sort modified

# List notes sorted by size
python mcp_cli_fixed.py list-notes --sort size

# List notes in JSON format (FIXED: datetime serialization)
python mcp_cli_fixed.py list-notes --format json
```

### Rebuild Index Command

Rebuild the search index with advanced configuration options.

```bash
python mcp_cli_fixed.py rebuild-index [OPTIONS] [NOTES_ROOT]
```

**Options:**
- `--model TEXT`: Embedding model to use
- `--chunk-size INTEGER`: Text chunk size
- `--overlap INTEGER`: Chunk overlap size
- `--force`: Force rebuild even if files unchanged

**Examples:**
```bash
# Basic rebuild
python mcp_cli_fixed.py rebuild-index

# Rebuild with different model
python mcp_cli_fixed.py rebuild-index --model all-mpnet-base-v2

# Rebuild with custom chunk settings (FIXED!)
python mcp_cli_fixed.py rebuild-index --chunk-size 300 --overlap 100

# Force rebuild
python mcp_cli_fixed.py rebuild-index --force

# Rebuild specific directory
python mcp_cli_fixed.py rebuild-index ./my_notes
```

### Server Command

Start the MCP server for Copilot integration with improved error handling.

```bash
python mcp_cli_fixed.py server [OPTIONS]
```

**Options:**
- `--host TEXT`: Server host (default: 127.0.0.1)
- `--port INTEGER`: Server port (default: 8181)
- `--no-admin`: Disable admin HTTP interface

**Examples:**
```bash
# Start server with defaults
python mcp_cli_fixed.py server

# Start server on custom port
python mcp_cli_fixed.py server --port 8080

# Start server on custom host
python mcp_cli_fixed.py server --host 0.0.0.0

# Start server without admin interface
python mcp_cli_fixed.py server --no-admin

# Start server with custom host and port
python mcp_cli_fixed.py server --host 0.0.0.0 --port 9090
```

### QA Command (NEW!)

Ask questions about your notes using AI-powered question answering.

```bash
python mcp_cli_fixed.py qa [OPTIONS] QUESTION
```

**Options:**
- `--limit INTEGER`: Maximum context chunks (default: 5)
- `--model TEXT`: AI model to use (default: gemini-1.5-pro)

**Examples:**
```bash
# Ask a question
python mcp_cli_fixed.py qa "Explain PCA from my notes"

# Ask with more context
python mcp_cli_fixed.py qa "What are machine learning algorithms?" --limit 10
```

### Generate Note Command (NEW!)

Generate new notes based on prompts using AI.

```bash
python mcp_cli_fixed.py generate-note [OPTIONS] PROMPT
```

**Options:**
- `--base-note PATH`: Use existing note as reference
- `--output PATH`: Custom output filename
- `--model TEXT`: AI model to use (default: gemini-1.5-pro)

**Examples:**
```bash
# Generate a note
python mcp_cli_fixed.py generate-note "Web development best practices"

# Generate with reference
python mcp_cli_fixed.py generate-note "Advanced PCA" --base-note pca_notes.md

# Custom output
python mcp_cli_fixed.py generate-note "ML algorithms" --output ml_guide.md
```

## ⚙️ Configuration

### Environment Variables

Customize the behavior using environment variables:

```bash
# Set embedding model
set MCP_MODEL_NAME=all-mpnet-base-v2

# Set chunk size for text processing
set MCP_CHUNK_SIZE=200

# Set chunk overlap
set MCP_OVERLAP=50

# Set environment
set MCP_ENVIRONMENT=production
```

### Configuration File

The system uses [`config.py`](config.py ) for advanced configuration. You can modify:

- Embedding model settings
- Text chunking parameters
- File paths for index and metadata
- Server configuration

### Fallback Systems

The CLI includes multiple fallback systems for maximum reliability:

- **MCP Server Fallback**: Falls back to simple implementation if full MCP unavailable
- **Text Search Fallback**: Uses basic text matching if semantic search fails
- **Export Fallback**: Falls back to text format if PDF/HTML generation fails
- **Directory Creation**: Automatically creates notes directory if missing

## 📚 Examples

### Complete Workflow

```bash
# 1. Setup and check system
python mcp_cli_fixed.py list-notes
python mcp_cli_fixed.py stats

# 2. Add new research notes
echo "# Machine Learning Research
Recent advances in neural networks and deep learning" > research.md
python mcp_cli_fixed.py add-note ./research.md --auto-tag --rebuild

# 3. Search and analyze
python mcp_cli_fixed.py search "neural networks" --format table --limit 5

# 4. Export findings
python mcp_cli_fixed.py export-search "deep learning" --format pdf --output ml_research.pdf

# 5. View statistics
python mcp_cli_fixed.py stats --period week --format json

# 6. Start server for Copilot
python mcp_cli_fixed.py server
```

### Batch Processing

```bash
# Add multiple notes
for file in ./new_notes/*.md; do
    python mcp_cli_fixed.py add-note "$file" --auto-tag
done

# Rebuild index after batch
python mcp_cli_fixed.py rebuild-index

# Export multiple searches
python mcp_cli_fixed.py export-search "machine learning" --format pdf --output ml.pdf
python mcp_cli_fixed.py export-search "web development" --format html --output web.html
python mcp_cli_fixed.py export-search "docker" --format markdown --output docker.md
```

### Advanced Search

```bash
# High-precision search
python mcp_cli_fixed.py search "machine learning algorithms" --threshold 0.8 --limit 10

# Export search with custom filename
python mcp_cli_fixed.py search "data science" --export ds_results.json --format json

# Search with table output
python mcp_cli_fixed.py search "python libraries" --format table --limit 8
```

### Environment Configuration

```bash
# Set custom model
set MCP_MODEL_NAME=all-mpnet-base-v2
set MCP_CHUNK_SIZE=300
set MCP_OVERLAP=100

# Test with new settings
python mcp_cli_fixed.py rebuild-index
python mcp_cli_fixed.py search "machine learning" --limit 5
```

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors:**
```bash
# Check dependencies
pip install -r requirements.txt

# Verify Python path
python -c "import sys; print(sys.path)"
```

**2. Search Not Working:**
```bash
# Rebuild index
python mcp_cli_fixed.py rebuild-index

# Check if notes directory exists
dir notes\*.md
```

**3. JSON Serialization Errors (FIXED!):**
```bash
# This now works automatically
python mcp_cli_fixed.py list-notes --format json
```

**4. Export Functionality (NEW!):**
```bash
# Export search results
python mcp_cli_fixed.py search "query" --export results.json --format json
```

**5. Server Not Starting:**
```bash
# Check port availability
netstat -an | findstr :8181

# Try different port
python mcp_cli_fixed.py server --port 8080
```

**6. Rebuild Index Issues (FIXED!):**
```bash
# Now supports all arguments
python mcp_cli_fixed.py rebuild-index --chunk-size 300 --overlap 100
```

### Debug Mode

Enable debug output for troubleshooting:

```bash
# Run with verbose output
python mcp_cli_fixed.py search "test" --limit 3
```

### Recent Fixes

- ✅ **JSON datetime serialization** - Fixed datetime objects in JSON output
- ✅ **Search export functionality** - Added --export argument to search command
- ✅ **Rebuild-index arguments** - Fixed argument parsing and directory handling
- ✅ **Server threading issues** - Fixed import and threading problems
- ✅ **Syntax errors** - All f-string and compilation errors fixed
- ✅ **Fallback systems** - Multiple fallback implementations for reliability

## 🧪 Testing

### Test Scripts

Run comprehensive tests to verify all functionality:

```bash
# Run all diagnostic tests
python debug_cli.py
python quick_fix.py
python test_cli_simple.py
python test_cli_fixes.py
python test_all_fixes.py
python final_test.py
```

### Test Individual Components

```bash
# Test CLI help
python mcp_cli_fixed.py --help

# Test basic functionality
python mcp_cli_fixed.py list-notes
python mcp_cli_fixed.py stats

# Test search
python mcp_cli_fixed.py search "test" --limit 3

# Test export
python mcp_cli_fixed.py search "test" --export test.json --format json
```

### Test Fallback Systems

```bash
# Test without MCP server (uses fallback)
python mcp_cli_fixed.py search "test" --limit 3

# Test with missing dependencies (graceful degradation)
python mcp_cli_fixed.py export-search "test" --format pdf
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run tests: `python -m pytest`
6. Commit your changes: `git commit -am 'Add new feature'`
7. Push to the branch: `git push origin feature-name`
8. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-repo/markdown-notes-mcp.git
cd markdown-notes-mcp

# Create development environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python -m pytest

# Run linting
flake8 mcp_cli_fixed.py
black mcp_cli_fixed.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [sentence-transformers](https://github.com/UKPLab/sentence-transformers) for semantic search
- Uses [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- PDF generation powered by [ReportLab](https://www.reportlab.com/)
- Table formatting with [tabulate](https://github.com/astanin/python-tabulate)
- Robust error handling and fallback systems implemented

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review the examples for common use cases
- Run the test scripts for validation

---

**Happy note-taking with AI-powered search!** 🚀📝

*Last updated: September 2025*

## 🆕 Recent Updates

### Version 2.1 Features

- **🤖 AI Question-Answering**: Ask natural language questions about your notes
- **📝 AI Note Generation**: Create new notes based on prompts or existing content
- **🔄 Google Gemini Integration**: Advanced AI model for intelligent responses
- **⚙️ Enhanced Configuration**: Comprehensive AI settings and model selection
- **🛡️ Robust Error Handling**: Automatic retries and fallback systems for AI features
- **📊 Context-Aware Responses**: Smart retrieval of relevant note sections
- **🎯 Reference-Based Generation**: Use existing notes as context for new content
- **⚡ Performance Optimization**: Efficient API usage with rate limit handling
