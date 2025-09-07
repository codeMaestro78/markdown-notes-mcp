# markdown-notes-mcp

A **production-ready semantic search system** for Markdown notes with advanced Model Context Protocol (MCP) server integration. Features enterprise-grade configuration, multiple embedding models, dynamic chunking strategies, and environment-specific deployments.

## ✨ **New Advanced Features**

### 🚀 **Advanced Configuration System**
- **Multiple Embedding Models**: Sentence Transformers, OpenAI, HuggingFace, Cohere
- **Dynamic Chunking Strategies**: Fixed, sentence-based, heading-based, semantic, hybrid
- **Environment-Specific Settings**: Development, staging, production, testing
- **Custom Preprocessing Pipelines**: Text cleaning, normalization, filtering
- **Performance Optimization**: GPU support, memory management, batch processing

### 🎯 **Enterprise-Ready Features**
- **Multi-Model Support**: Switch between models for different use cases
- **Scalable Architecture**: Handle large document collections efficiently
- **Production Monitoring**: Health checks, metrics, and logging
- **Security**: Admin authentication, access control, audit logging
- **Docker Support**: Containerized deployment with optimized images

## Features

- **🔍 Semantic Search**: Multiple embedding models (all-MiniLM-L6-v2, all-mpnet-base-v2, multilingual)
- **🔄 Hybrid Search**: Combines semantic embeddings with TF-IDF lexical search
- **⚡ Fast Indexing**: FAISS integration with optimized vector search
- **🔌 MCP Protocol**: JSON-RPC 2.0 server with comprehensive tool support
- **📊 Dynamic Chunking**: Multiple strategies for optimal text segmentation
- **🌐 Admin API**: HTTP endpoints with authentication and monitoring
- **🔧 Advanced Configuration**: Environment-specific settings and model management
- **📈 Performance Monitoring**: Detailed metrics and health checks
- **🐳 Container Ready**: Docker deployment with multi-stage builds

## Project Structure

```
markdown-notes-mcp/
├── config/                     # Advanced configuration system
│   ├── development.json        # Development environment settings
│   ├── staging.json           # Staging environment settings
│   ├── production.json        # Production environment settings
│   ├── testing.json           # Testing environment settings
│   ├── models.json            # Model configurations
│   └── preprocessing.json     # Text processing settings
├── notes/                      # Your Markdown notes go here
│   ├── example.md
│   ├── pca_notes.md
│   ├── devops_cloud.md
│   ├── machine_learning_fundamentals.md
│   ├── python_data_science.md
│   └── web_development.md
├── build_index.py              # Advanced index builder with multi-model support
├── notes_mcp_server.py         # Enhanced MCP server with dynamic configuration
├── admin_http.py               # Production-ready HTTP admin API
├── config.py                   # Advanced configuration management system
├── test_harness_client.py      # Comprehensive test client
├── tests/
│   └── test_end_to_end.py      # End-to-end testing suite
├── requirements.txt            # Python dependencies with optional packages
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.yml          # Container orchestration
├── README.md                   # This comprehensive documentation
├── CONFIGURATION_GUIDE.md      # Advanced configuration documentation
└── .copilot/
    └── mcp.json                # MCP configuration for Claude Desktop
```

## Requirements

- **Python 3.8+**
- **Core Dependencies**:
  - `sentence-transformers` - Multiple embedding model support
  - `numpy` - Numerical computing and embeddings
  - `scikit-learn` - Machine learning utilities
  - `python-frontmatter` - YAML frontmatter parsing
  - `fastapi` - Modern async web framework
  - `uvicorn` - ASGI server for FastAPI

- **Optional Performance Enhancements**:
  - `faiss-cpu` - High-performance vector search
  - `transformers` - Advanced NLP models
  - `torch` - PyTorch for GPU acceleration
  - `pandas` - Data manipulation (for large datasets)
  - `psutil` - System monitoring

- **Development Dependencies**:
  - `pytest` - Testing framework
  - `black` - Code formatting
  - `mypy` - Type checking
  - `pre-commit` - Git hooks

## 🚀 **Quick Start with Advanced Features**

### 1. **Environment Setup**
```bash
# Clone and setup
git clone <repository-url>
cd markdown-notes-mcp

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. **Choose Your Environment**
```bash
# Development (fast, lightweight)
$env:MCP_ENVIRONMENT="development"

# Production (high-quality, optimized)
$env:MCP_ENVIRONMENT="production"

# Staging (balanced performance)
$env:MCP_ENVIRONMENT="staging"
```

### 3. **Add Your Knowledge Base**
```bash
# Place .md files in notes/ directory
# The system supports:
# - YAML frontmatter metadata
# - Multiple chunking strategies
# - Cross-references and linking
# - Rich markdown formatting
```

### 4. **Build Multi-Model Index**
```bash
# Basic indexing (uses environment defaults)
python build_index.py ./notes

# Advanced indexing with custom model
python build_index.py ./notes --model all-mpnet-base-v2

# High-performance indexing
$env:MCP_MODEL_NAME="all-mpnet-base-v2"
$env:MCP_BATCH_SIZE="128"
python build_index.py ./notes
```

## 🎯 **Advanced Usage**

### **Multi-Model Indexing**

```bash
# Fast model for development
$env:MCP_MODEL_NAME="all-MiniLM-L6-v2"
python build_index.py ./notes

# High-quality model for production
$env:MCP_MODEL_NAME="all-mpnet-base-v2"
python build_index.py ./notes

# Multilingual support
$env:MCP_MODEL_NAME="paraphrase-multilingual-MiniLM-L12-v2"
python build_index.py ./notes
```

### **Dynamic Chunking Strategies**

```bash
# Sentence-based chunking (good for documents)
$env:MCP_CHUNK_SIZE="200"
$env:MCP_OVERLAP="50"
python build_index.py ./notes

# Fine-grained chunking (good for code/technical docs)
$env:MCP_CHUNK_SIZE="100"
$env:MCP_OVERLAP="25"
python build_index.py ./notes

# Large chunks with high overlap (good for context-heavy content)
$env:MCP_CHUNK_SIZE="300"
$env:MCP_OVERLAP="100"
python build_index.py ./notes
```

### **Performance Optimization**

```bash
# GPU acceleration (if available)
$env:MCP_ENABLE_GPU="true"
$env:MCP_BATCH_SIZE="256"
python build_index.py ./notes

# Memory-efficient processing
$env:MCP_BATCH_SIZE="16"
$env:MCP_MAX_FILE_SIZE="5242880"  # 5MB limit
python build_index.py ./notes

# High-throughput processing
$env:MCP_BATCH_SIZE="512"
$env:MCP_REQUEST_TIMEOUT="60"
python build_index.py ./notes
```

### **MCP Server with Advanced Configuration**

```bash
# Basic server
python notes_mcp_server.py

# Production server with monitoring
$env:MCP_ENVIRONMENT="production"
python notes_mcp_server.py --host 0.0.0.0 --port 8181

# Development server with debug logging
$env:MCP_ENVIRONMENT="development"
python notes_mcp_server.py --log-level DEBUG
```

### **Admin API with Authentication**

```bash
# Start admin server
$env:MCP_ADMIN_TOKEN="your-secure-token-here"
python admin_http.py

# Health check
curl http://127.0.0.1:8181/health

# System metrics
curl http://127.0.0.1:8181/metrics

# Reindex with authentication
curl -X POST http://127.0.0.1:8181/reindex \
  -H "X-ADMIN-TOKEN: your-secure-token-here" \
  -d '{"notes_root": "./notes"}'
```

## 🔌 **Advanced MCP Integration**

### **Environment-Specific MCP Configuration**

**Development Configuration:**
```json
{
  "mcpServers": {
    "markdown-notes-dev": {
      "command": "python",
      "args": ["notes_mcp_server.py"],
      "env": {
        "MCP_ENVIRONMENT": "development",
        "MCP_MODEL_NAME": "all-MiniLM-L6-v2",
        "PYTHONPATH": "."
      }
    }
  }
}
```

**Production Configuration:**
```json
{
  "mcpServers": {
    "markdown-notes-prod": {
      "command": "python",
      "args": ["notes_mcp_server.py"],
      "env": {
        "MCP_ENVIRONMENT": "production",
        "MCP_MODEL_NAME": "all-mpnet-base-v2",
        "MCP_ENABLE_GPU": "true",
        "PYTHONPATH": "."
      }
    }
  }
}
```

### **Enhanced MCP Tools**

**Available Tools for LLMs:**
- 🔍 **search_notes** - Multi-model semantic search with advanced filtering
- 📄 **get_note_content** - Full content retrieval with metadata
- 📋 **list_notes** - Comprehensive note listing with statistics
- 🔄 **reindex_notes** - Dynamic reindexing with custom parameters
- 📊 **get_search_stats** - Performance metrics and analytics
- ⚙️ **configure_search** - Runtime configuration updates

**Advanced Search Parameters:**
```json
{
  "query": "machine learning algorithms",
  "top_k": 10,
  "model": "all-mpnet-base-v2",
  "chunking_strategy": "semantic",
  "lexical_weight": 0.2,
  "semantic_weight": 0.8,
  "filters": {
    "tags": ["machine-learning", "algorithms"],
    "date_range": ["2024-01-01", "2024-12-31"]
  }
}
```

### **Multi-Model Search Capabilities**

The system supports **dynamic model switching** during runtime:

```json
{
  "query": "complex technical analysis",
  "model": "all-mpnet-base-v2",  // High-quality for technical content
  "top_k": 5
}
```

```json
{
  "query": "quick information lookup",
  "model": "all-MiniLM-L6-v2",  // Fast model for quick searches
  "top_k": 3
}
```

### **Real-time Configuration Updates**

LLMs can now **dynamically adjust** search parameters:

```json
{
  "method": "configure_search",
  "params": {
    "model": "paraphrase-multilingual-MiniLM-L12-v2",
    "chunking_strategy": "heading_based",
    "batch_size": 64,
    "similarity_threshold": 0.8
  }
}
```

## ⚙️ **Advanced Configuration System**

### **Environment Management**

The system supports **four deployment environments** with optimized settings:

```bash
# Development - Fast iteration, lightweight models
$env:MCP_ENVIRONMENT="development"

# Staging - Balanced performance, testing
$env:MCP_ENVIRONMENT="staging"

# Production - High-quality, optimized performance
$env:MCP_ENVIRONMENT="production"

# Testing - Minimal resources, fast execution
$env:MCP_ENVIRONMENT="testing"
```

### **Model Configuration**

**Available Models:**
- `all-MiniLM-L6-v2` - Fast, lightweight (384d)
- `all-mpnet-base-v2` - High-quality, slower (768d)
- `paraphrase-multilingual-MiniLM-L12-v2` - Multilingual support (384d)

**Model Selection:**
```bash
# Quality vs Speed Trade-off
$env:MCP_MODEL_NAME="all-mpnet-base-v2"    # Best quality
$env:MCP_MODEL_NAME="all-MiniLM-L6-v2"    # Fastest
$env:MCP_MODEL_NAME="paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual
```

### **Dynamic Chunking Strategies**

**Available Strategies:**
- `fixed` - Fixed-size chunks with overlap
- `sentence` - Sentence-based segmentation
- `heading` - Heading-based document structure
- `semantic` - Semantic similarity-based
- `hybrid` - Combined approach

**Configuration Examples:**
```bash
# Technical documentation
$env:MCP_CHUNK_SIZE="150"
$env:MCP_OVERLAP="30"

# Long-form content
$env:MCP_CHUNK_SIZE="250"
$env:MCP_OVERLAP="75"

# Code and structured content
$env:MCP_CHUNK_SIZE="100"
$env:MCP_OVERLAP="20"
```

### **Performance Tuning**

**GPU Acceleration:**
```bash
$env:MCP_ENABLE_GPU="true"
$env:MCP_BATCH_SIZE="256"
```

**Memory Optimization:**
```bash
$env:MCP_BATCH_SIZE="16"
$env:MCP_MAX_WORKERS="2"
$env:MCP_MEMORY_LIMIT_MB="1024"
```

**High-Throughput Processing:**
```bash
$env:MCP_BATCH_SIZE="512"
$env:MCP_MAX_WORKERS="8"
$env:MCP_REQUEST_TIMEOUT="120"
```

### **Security Configuration**

**Admin Authentication:**
```bash
$env:MCP_ADMIN_TOKEN="your-secure-random-token-here"
$env:MCP_ENABLE_AUTH="true"
```

**Access Control:**
```bash
$env:MCP_ALLOWED_IPS="192.168.1.0/24,10.0.0.0/8"
$env:MCP_RATE_LIMIT="100/hour"
```

### **Monitoring & Logging**

**Log Configuration:**
```bash
$env:MCP_LOG_LEVEL="INFO"
$env:MCP_LOG_FILE="logs/markdown-mcp.log"
$env:MCP_ENABLE_METRICS="true"
```

**Health Monitoring:**
```bash
$env:MCP_HEALTH_CHECK_INTERVAL="30"
$env:MCP_METRICS_RETENTION_DAYS="7"
```

## 🐳 **Docker Deployment**

### **Multi-Stage Production Build**

```dockerfile
# Build stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM python:3.11-slim as production
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application
COPY . /app
WORKDIR /app

# Environment configuration
ENV MCP_ENVIRONMENT=production
ENV MCP_MODEL_NAME=all-mpnet-base-v2
ENV MCP_ENABLE_GPU=false

EXPOSE 8181
CMD ["python", "notes_mcp_server.py"]
```

### **Docker Compose Setup**

```yaml
version: '3.8'
services:
  markdown-mcp:
    build: .
    ports:
      - "8181:8181"
    volumes:
      - ./notes:/app/notes
      - ./config:/app/config
    environment:
      - MCP_ENVIRONMENT=production
      - MCP_MODEL_NAME=all-mpnet-base-v2
      - MCP_ADMIN_TOKEN=your-secure-token
    restart: unless-stopped

  admin-api:
    build: .
    ports:
      - "8182:8182"
    command: python admin_http.py
    environment:
      - MCP_ADMIN_TOKEN=your-secure-token
    depends_on:
      - markdown-mcp
```

### **Environment-Specific Containers**

```bash
# Development
docker build -t markdown-mcp:dev --build-arg ENVIRONMENT=development .

# Production
docker build -t markdown-mcp:prod --build-arg ENVIRONMENT=production .

# GPU-enabled
docker build -t markdown-mcp:gpu --build-arg ENABLE_GPU=true .
```

## 🎯 **Advanced Features Deep Dive**

### **Multi-Model Architecture**

**Model Capabilities:**
- **all-MiniLM-L6-v2**: Fast inference, good for development and quick searches
- **all-mpnet-base-v2**: Superior semantic understanding, best for production
- **paraphrase-multilingual-MiniLM-L12-v2**: Cross-language search capabilities

**Dynamic Model Switching:**
```python
# Runtime model switching
from config import AdvancedConfig
config = AdvancedConfig()
config.set_model("all-mpnet-base-v2")  # High quality
# or
config.set_model("all-MiniLM-L6-v2")   # Fast
```

### **Intelligent Chunking System**

**Strategy Selection:**
- **Fixed Chunking**: Consistent size, predictable performance
- **Sentence-Based**: Natural language boundaries, better coherence
- **Heading-Based**: Document structure preservation, hierarchical context
- **Semantic Chunking**: Content-aware splitting, optimal retrieval
- **Hybrid Approach**: Best of multiple strategies

**Adaptive Chunking:**
```python
# Automatic chunk size optimization
config = AdvancedConfig()
strategy = config.get_chunking_config("hybrid")
# System adapts based on content type and performance metrics
```

### **Enterprise Security Features**

**Authentication & Authorization:**
- JWT-based admin authentication
- Role-based access control
- API key management
- Request rate limiting

**Audit & Compliance:**
- Comprehensive logging
- Search query auditing
- Performance monitoring
- Data access tracking

### **Performance Optimization**

**Hardware Acceleration:**
- CUDA GPU support for embedding generation
- CPU optimization with SIMD instructions
- Memory-mapped file I/O for large indexes
- Parallel processing with configurable worker pools

**Caching Strategies:**
- Embedding result caching
- Query result caching
- Model instance pooling
- Connection pooling for external APIs

### **Monitoring & Analytics**

**Real-time Metrics:**
- Query latency tracking
- Model performance monitoring
- Index health statistics
- System resource usage

**Advanced Analytics:**
- Search pattern analysis
- User behavior insights
- Performance trend monitoring
- Automated optimization recommendations

## 📊 **Performance Benchmarks**

### **Model Performance Comparison**

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Development, fast search |
| all-mpnet-base-v2 | 768 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Production, high accuracy |
| multilingual | 384 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | International content |

### **Chunking Strategy Performance**

| Strategy | Precision | Recall | Speed | Best For |
|----------|-----------|--------|-------|----------|
| Fixed | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | General purpose |
| Sentence | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Natural language |
| Heading | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Structured documents |
| Semantic | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Complex content |
| Hybrid | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | All content types |

## 🔧 **API Reference**

### **MCP Server Methods**

**Core Methods:**
- `search_notes(query, top_k=5, model=None, filters=None)`
- `get_note_content(path, include_metadata=True)`
- `list_notes(filters=None, sort_by="modified")`
- `health_check()`
- `get_metrics()`

**Advanced Methods:**
- `reindex_notes(notes_root, model=None, chunking_strategy=None)`
- `configure_search(model, chunking_strategy, performance_settings)`
- `get_model_info(model_name)`
- `validate_configuration()`

### **Admin API Endpoints**

**Management:**
- `POST /reindex` - Trigger index rebuild
- `GET /config` - View current configuration
- `PUT /config` - Update configuration
- `GET /logs` - Retrieve system logs

**Monitoring:**
- `GET /health` - System health status
- `GET /metrics` - Performance metrics
- `GET /stats` - Usage statistics
- `GET /performance` - Performance analytics

## 🚀 **Production Deployment Guide**

### **1. Environment Setup**
```bash
# Production environment
export MCP_ENVIRONMENT=production
export MCP_MODEL_NAME=all-mpnet-base-v2
export MCP_ADMIN_TOKEN=$(openssl rand -hex 32)
export MCP_ENABLE_GPU=true
```

### **2. Security Configuration**
```bash
# SSL/TLS setup
export MCP_SSL_CERT=/path/to/cert.pem
export MCP_SSL_KEY=/path/to/key.pem

# Network security
export MCP_ALLOWED_IPS=10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
export MCP_RATE_LIMIT=1000/hour
```

### **3. Performance Tuning**
```bash
# Hardware optimization
export MCP_BATCH_SIZE=512
export MCP_MAX_WORKERS=16
export MCP_MEMORY_LIMIT_MB=8192

# Storage optimization
export MCP_INDEX_COMPRESSION=true
export MCP_CACHE_SIZE_MB=2048
```

### **4. Monitoring Setup**
```bash
# Logging
export MCP_LOG_LEVEL=INFO
export MCP_LOG_FILE=/var/log/markdown-mcp/server.log

# Metrics
export MCP_METRICS_PORT=9090
export MCP_METRICS_PATH=/metrics
```

### **5. Backup Strategy**
```bash
# Automated backups
export MCP_BACKUP_INTERVAL=24h
export MCP_BACKUP_RETENTION=30
export MCP_BACKUP_PATH=/backup/markdown-mcp
```

## 🎯 **Quick Start**

1. **Install**: `pip install -r requirements.txt`
2. **Configure**: `export MCP_ENVIRONMENT=development`
3. **Index**: `python build_index.py ./notes`
4. **Serve**: `python notes_mcp_server.py`
5. **Test**: `python test_harness_client.py`

**That's it! Your advanced semantic search system is ready.** 🚀
