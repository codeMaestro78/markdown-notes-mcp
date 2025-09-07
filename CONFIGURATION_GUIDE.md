# Configuration & Improvements Documentation

## 🚀 **Implemented Improvements**

### 1. **Centralized Configuration System** (`config.py`)

**What it does:**
- Centralized all configuration in one place
- Environment variable support for easy deployment
- Validation of configuration values
- Type hints and documentation

**Usage:**
```python
from config import config

# Access configuration values
model_name = config.model_name
chunk_size = config.chunk_size
batch_size = config.batch_size
```

**Environment Variables:**
```bash
# Model settings
export MCP_MODEL_NAME="all-MiniLM-L6-v2"
export MCP_CHUNK_SIZE="150"
export MCP_OVERLAP="30"
export MCP_BATCH_SIZE="64"

# File paths
export MCP_INDEX_FILE="notes_index.npz"
export MCP_META_FILE="notes_meta.json"
export MCP_NOTES_ROOT="./notes"

# Server settings
export MCP_HOST="127.0.0.1"
export MCP_PORT="8181"
export MCP_ADMIN_TOKEN="your-secret-token"

# Advanced settings
export MCP_MAX_FILE_SIZE="10485760"  # 10MB
export MCP_REQUEST_TIMEOUT="30"
```

### 2. **Enhanced Error Handling & Logging**

**What it does:**
- Comprehensive logging with timestamps
- Detailed error messages
- File logging in addition to console
- Graceful error recovery

**Log Files Created:**
- `build_index.log` - Index building logs
- `notes_mcp_server.log` - Server operation logs

**Log Levels:**
- `INFO`: General operations and progress
- `WARNING`: Non-critical issues
- `ERROR`: Critical errors that need attention
- `DEBUG`: Detailed debugging information

### 3. **Improved Command Line Interface**

**Build Index:**
```bash
# Basic usage (uses config defaults)
python build_index.py ./notes

# Custom settings
python build_index.py ./notes --index custom_index.npz --meta custom_meta.json --model "all-mpnet-base-v2"

# With environment variables
export MCP_CHUNK_SIZE="200"
export MCP_MODEL_NAME="all-mpnet-base-v2"
python build_index.py ./notes
```

**MCP Server:**
```bash
# Basic usage
python notes_mcp_server.py --index notes_index.npz --meta notes_meta.json

# Custom settings
python notes_mcp_server.py --index notes_index.npz --meta notes_meta.json --notes_root ./docs --model "all-mpnet-base-v2"
```

### 4. **Enhanced Admin API**

**New Endpoints:**
- `GET /health` - Health check
- `POST /reindex` - Rebuild index (with auth)
- `GET /metrics` - Detailed system metrics
- `GET /config` - View current configuration

**Usage Examples:**
```bash
# Health check
curl http://127.0.0.1:8181/health

# Get metrics
curl http://127.0.0.1:8181/metrics

# Reindex with auth
curl -X POST http://127.0.0.1:8181/reindex \
  -H "Content-Type: application/json" \
  -H "X-ADMIN-TOKEN: your-token" \
  -d '{"notes_root": "./notes"}'

# View configuration
curl http://127.0.0.1:8181/config
```

### 5. **Better File Processing**

**Improvements:**
- File size limits (configurable)
- Encoding error handling
- Progress tracking
- Better Markdown parsing
- Frontmatter extraction
- Heading-based chunking

**Supported File Types:**
- Markdown (.md) - Full support
- Text (.txt) - Basic support
- Future: PDF, DOCX support planned

### 6. **Performance Enhancements**

**Memory Management:**
- Configurable batch sizes
- Progress monitoring
- Timeout handling
- Resource cleanup

**Search Optimization:**
- FAISS integration (if available)
- Normalized embeddings
- Efficient metadata storage

## 📋 **Migration Guide**

### **For Existing Users:**

1. **No breaking changes** - your existing commands still work
2. **New features are opt-in** - use them when ready
3. **Environment variables** - override defaults as needed

### **Upgrading from Previous Version:**

```bash
# Your old commands still work
python build_index.py ./notes
python notes_mcp_server.py --index notes_index.npz --meta notes_meta.json

# New: Use environment variables for configuration
export MCP_MODEL_NAME="all-mpnet-base-v2"
export MCP_CHUNK_SIZE="200"
python build_index.py ./notes

# New: Better logging and error handling
python build_index.py ./notes  # Check build_index.log for details
```

## 🔧 **Troubleshooting**

### **Configuration Issues:**
```bash
# Check configuration
python -c "from config import config; print(config.to_dict())"

# Validate configuration
python -c "from config import config; print('Valid:', config.validate())"
```

### **Common Errors:**

**"Module 'config' not found":**
```bash
# Ensure you're in the project directory
cd /path/to/markdown-notes-mcp
python -c "import config"
```

**"Invalid configuration":**
```bash
# Check environment variables
env | grep MCP_

# Reset to defaults
unset MCP_MODEL_NAME MCP_CHUNK_SIZE  # etc.
```

**"Index build failed":**
```bash
# Check the log file
tail -f build_index.log

# Common issues:
# - Missing notes directory
# - Permission issues
# - Large files (>10MB by default)
```

### **Performance Tuning:**

```bash
# For faster indexing (uses more memory)
export MCP_BATCH_SIZE="128"
export MCP_MODEL_NAME="all-mpnet-base-v2"  # Better but slower

# For memory-constrained systems
export MCP_BATCH_SIZE="16"
export MCP_CHUNK_SIZE="100"
```

## 🎯 **Best Practices**

### **Development:**
1. Use environment variables for different environments
2. Check log files for debugging
3. Use the admin API for monitoring
4. Test with small datasets first

### **Production:**
1. Set strong admin tokens
2. Monitor log files regularly
3. Use absolute paths in configuration
4. Backup index files regularly
5. Set appropriate file size limits

### **Performance:**
1. Choose model based on accuracy vs speed needs
2. Adjust chunk size based on document types
3. Use FAISS for large indexes
4. Monitor memory usage during indexing

## 🚀 **Next Steps**

### **Immediate Benefits:**
- ✅ Better error messages and debugging
- ✅ Configurable settings without code changes
- ✅ Detailed progress tracking
- ✅ Production-ready logging

### **Future Enhancements:**
- 🔄 Incremental indexing
- 🔄 Multi-language support
- 🔄 Web-based admin interface
- 🔄 Plugin system for custom processors
- 🔄 Advanced search features

## 📞 **Support**

If you encounter issues:

1. Check the log files (`*.log`)
2. Verify your configuration with `python -c "from config import config; print(config.to_dict())"`
3. Test with minimal settings first
4. Check the admin API metrics endpoint

**All improvements are backward-compatible and opt-in!** 🎉
