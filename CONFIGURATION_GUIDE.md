# Advanced Configuration Guide

## 🚀 **Enterprise-Grade Configuration System**

This comprehensive guide covers the **advanced configuration system** with multi-model support, dynamic chunking strategies, environment-specific settings, and production-ready features.

## 📋 **System Architecture**

### **Core Components**

1. **AdvancedConfig Class** (`config.py`)
   - Centralized configuration management
   - Environment detection and switching
   - Model and chunking strategy management
   - Performance optimization settings

2. **Environment-Specific Files** (`config/*.json`)
   - `development.json` - Fast, lightweight settings
   - `staging.json` - Balanced performance settings
   - `production.json` - Optimized, high-performance settings
   - `testing.json` - Minimal resource settings

3. **Model Configuration** (`config/models.json`)
   - Multiple embedding model definitions
   - Provider-specific settings
   - Dimension and performance specifications

4. **Preprocessing Pipeline** (`config/preprocessing.json`)
   - Text cleaning and normalization
   - Custom filtering rules
   - Chunking strategy definitions

## 🎯 **Environment Management**

### **Supported Environments**

#### **Development Environment**
```json
{
  "model_name": "all-MiniLM-L6-v2",
  "chunk_size": 100,
  "overlap": 20,
  "batch_size": 32,
  "max_file_size": 5242880,
  "request_timeout": 10,
  "host": "127.0.0.1",
  "port": 8181,
  "admin_token": "dev_token_123"
}
```
**Use Case**: Fast iteration, lightweight models, local development
**Performance**: Speed-optimized, minimal resource usage

#### **Staging Environment**
```json
{
  "model_name": "all-mpnet-base-v2",
  "chunk_size": 150,
  "overlap": 30,
  "batch_size": 64,
  "max_file_size": 10485760,
  "request_timeout": 15,
  "host": "0.0.0.0",
  "port": 8181,
  "admin_token": "staging_token_456"
}
```
**Use Case**: Pre-production testing, performance validation
**Performance**: Balanced speed and quality

#### **Production Environment**
```json
{
  "model_name": "all-mpnet-base-v2",
  "chunk_size": 200,
  "overlap": 50,
  "batch_size": 128,
  "max_file_size": 20971520,
  "request_timeout": 30,
  "host": "0.0.0.0",
  "port": 8181,
  "admin_token": "prod_token_789"
}
```
**Use Case**: Live production deployment, high availability
**Performance**: Quality-optimized, maximum performance

#### **Testing Environment**
```json
{
  "model_name": "all-MiniLM-L6-v2",
  "chunk_size": 50,
  "overlap": 10,
  "batch_size": 16,
  "max_file_size": 1048576,
  "request_timeout": 5,
  "host": "127.0.0.1",
  "port": 8182,
  "admin_token": "test_token_000"
}
```
**Use Case**: Automated testing, CI/CD pipelines
**Performance**: Minimal resources, fast execution

### **Environment Switching**

#### **Command Line**
```bash
# Development
$env:MCP_ENVIRONMENT="development"
python build_index.py ./notes

# Production
$env:MCP_ENVIRONMENT="production"
python notes_mcp_server.py

# Testing
$env:MCP_ENVIRONMENT="testing"
pytest tests/
```

#### **Runtime Switching**
```python
from config import AdvancedConfig

# Load specific environment
config = AdvancedConfig()
config.load_environment("production")

# Or detect automatically
config = AdvancedConfig()  # Uses MCP_ENVIRONMENT variable
```

## 🤖 **Multi-Model Architecture**

### **Available Models**

#### **Sentence Transformers Models**
```json
{
  "all-MiniLM-L6-v2": {
    "provider": "sentence_transformers",
    "model_name": "all-MiniLM-L6-v2",
    "dimensions": 384,
    "description": "Fast and efficient for general purpose semantic search"
  },
  "all-mpnet-base-v2": {
    "provider": "sentence_transformers",
    "model_name": "all-mpnet-base-v2",
    "dimensions": 768,
    "description": "High-quality embeddings for complex semantic understanding"
  },
  "paraphrase-multilingual-MiniLM-L12-v2": {
    "provider": "sentence_transformers",
    "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
    "dimensions": 384,
    "description": "Multilingual support for international content"
  }
}
```

#### **Model Selection Guide**

| Model | Dimensions | Speed | Quality | Memory | Use Case |
|-------|------------|-------|---------|--------|----------|
| all-MiniLM-L6-v2 | 384 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Low | Development, fast search |
| all-mpnet-base-v2 | 768 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High | Production, accuracy-critical |
| multilingual | 384 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | International content |

### **Dynamic Model Switching**

#### **Configuration-Based Switching**
```python
from config import AdvancedConfig

config = AdvancedConfig()

# Switch to high-quality model
config.set_model("all-mpnet-base-v2")

# Switch to fast model
config.set_model("all-MiniLM-L6-v2")

# Switch to multilingual
config.set_model("paraphrase-multilingual-MiniLM-L12-v2")
```

#### **Runtime Model Switching**
```python
# During indexing
python build_index.py ./notes --model all-mpnet-base-v2

# During search
search_request = {
  "query": "machine learning",
  "model": "all-mpnet-base-v2",
  "top_k": 5
}
```

## 📊 **Dynamic Chunking Strategies**

### **Available Strategies**

#### **Fixed Size Chunking**
```json
{
  "method": "fixed",
  "size": 200,
  "overlap": 50
}
```
**Best For**: General purpose, predictable performance
**Pros**: Consistent, fast processing
**Cons**: May break semantic units

#### **Sentence-Based Chunking**
```json
{
  "method": "sentence",
  "sentence_splitter": "nltk",
  "min_sentences": 2,
  "max_sentences": 5
}
```
**Best For**: Natural language content
**Pros**: Respects semantic boundaries
**Cons**: Variable chunk sizes

#### **Heading-Based Chunking**
```json
{
  "method": "heading",
  "preserve_headings": true,
  "custom_splitters": ["\n## ", "\n### ", "\n#### "]
}
```
**Best For**: Structured documents, technical documentation
**Pros**: Maintains document hierarchy
**Cons**: May create very large chunks

#### **Semantic Chunking**
```json
{
  "method": "semantic",
  "similarity_threshold": 0.7,
  "min_chunk_size": 100,
  "max_chunk_size": 500
}
```
**Best For**: Complex content, research papers
**Pros**: Optimal semantic coherence
**Cons**: Computationally intensive

#### **Hybrid Chunking**
```json
{
  "method": "hybrid",
  "chunk_size": 150,
  "overlap": 30,
  "preserve_headings": true,
  "semantic_threshold": 0.7
}
```
**Best For**: All content types, production use
**Pros**: Best of all strategies
**Cons**: Most complex configuration

### **Strategy Selection Guide**

| Content Type | Recommended Strategy | Configuration |
|-------------|---------------------|---------------|
| Technical Docs | heading_based | preserve_headings: true |
| Research Papers | semantic | threshold: 0.8 |
| Blog Posts | sentence_based | min_sentences: 3 |
| Code Documentation | fixed | size: 100, overlap: 20 |
| Mixed Content | hybrid | chunk_size: 150, overlap: 30 |

## 🔧 **Advanced Configuration Options**

### **Performance Tuning**

#### **GPU Acceleration**
```bash
# Enable GPU processing
$env:MCP_ENABLE_GPU="true"
$env:MCP_BATCH_SIZE="256"

# CPU optimization
$env:MCP_MAX_WORKERS="8"
$env:MCP_MEMORY_LIMIT_MB="4096"
```

#### **Memory Management**
```bash
# Large dataset handling
$env:MCP_BATCH_SIZE="512"
$env:MCP_MAX_FILE_SIZE="536870912"  # 512MB

# Memory-constrained environments
$env:MCP_BATCH_SIZE="16"
$env:MCP_MEMORY_LIMIT_MB="1024"
```

#### **I/O Optimization**
```bash
# High-throughput processing
$env:MCP_REQUEST_TIMEOUT="120"
$env:MCP_MAX_WORKERS="16"

# Low-latency requirements
$env:MCP_REQUEST_TIMEOUT="5"
$env:MCP_MAX_WORKERS="2"
```

### **Security Configuration**

#### **Authentication**
```bash
# Admin token (generate securely)
$env:MCP_ADMIN_TOKEN="$(openssl rand -hex 32)"

# API key authentication
$env:MCP_API_KEYS="key1,key2,key3"
```

#### **Access Control**
```bash
# IP restrictions
$env:MCP_ALLOWED_IPS="192.168.1.0/24,10.0.0.0/8"

# Rate limiting
$env:MCP_RATE_LIMIT="1000/hour"
```

#### **Encryption**
```bash
# SSL/TLS configuration
$env:MCP_SSL_CERT="/path/to/cert.pem"
$env:MCP_SSL_KEY="/path/to/key.pem"

# Data encryption
$env:MCP_ENCRYPT_DATA="true"
```

### **Monitoring & Logging**

#### **Log Configuration**
```bash
# Log levels
$env:MCP_LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR

# Log files
$env:MCP_LOG_FILE="/var/log/markdown-mcp/server.log"
$env:MCP_LOG_MAX_SIZE="100MB"
$env:MCP_LOG_BACKUP_COUNT="5"
```

#### **Metrics Collection**
```bash
# Enable metrics
$env:MCP_ENABLE_METRICS="true"
$env:MCP_METRICS_PORT="9090"
$env:MCP_METRICS_PATH="/metrics"

# Performance monitoring
$env:MCP_PERFORMANCE_MONITORING="true"
$env:MCP_METRICS_RETENTION_DAYS="30"
```

## 🔄 **Custom Preprocessing Pipelines**

### **Text Processing Configuration**

```json
{
  "preprocessing": {
    "remove_headers": true,
    "normalize_whitespace": true,
    "remove_code_blocks": false,
    "extract_links": true,
    "custom_filters": [
      {
        "pattern": "\\[\\[.*?\\]\\]",
        "replacement": "",
        "description": "Remove Obsidian-style links"
      },
      {
        "pattern": "#+\\s*",
        "replacement": "",
        "description": "Remove markdown headers for better chunking"
      }
    ]
  }
}
```

### **Available Filters**

#### **Built-in Filters**
- **remove_frontmatter**: Strip YAML frontmatter
- **normalize_unicode**: Unicode normalization
- **remove_code_blocks**: Strip code sections
- **extract_links**: Convert links to text
- **remove_tables**: Strip table formatting
- **clean_html**: Remove HTML tags

#### **Custom Filters**
```json
{
  "custom_filters": [
    {
      "pattern": "\\[\\[([^\\]]+)\\]\\]",
      "replacement": "\\1",
      "description": "Convert Obsidian links to plain text"
    },
    {
      "pattern": "```.*?```",
      "replacement": "[CODE_BLOCK]",
      "description": "Replace code blocks with placeholder"
    }
  ]
}
```

## 📈 **Performance Optimization**

### **Hardware-Specific Tuning**

#### **GPU Optimization**
```bash
# NVIDIA GPU settings
$env:MCP_ENABLE_GPU="true"
$env:MCP_GPU_DEVICE="0"
$env:MCP_GPU_MEMORY_FRACTION="0.8"

# Multi-GPU support
$env:MCP_MULTI_GPU="true"
$env:MCP_GPU_DEVICES="0,1,2"
```

#### **CPU Optimization**
```bash
# Multi-core processing
$env:MCP_MAX_WORKERS="16"
$env:MCP_THREAD_POOL_SIZE="32"

# Memory optimization
$env:MCP_MEMORY_LIMIT_MB="8192"
$env:MCP_ENABLE_MEMORY_POOL="true"
```

### **Index Optimization**

#### **FAISS Configuration**
```bash
# Index types
$env:MCP_FAISS_INDEX_TYPE="IndexIVFFlat"
$env:MCP_FAISS_NLIST="1024"
$env:MCP_FAISS_NPROBE="10"

# Quantization
$env:MCP_ENABLE_QUANTIZATION="true"
$env:MCP_QUANTIZATION_BITS="8"
```

#### **Compression Settings**
```bash
# Index compression
$env:MCP_INDEX_COMPRESSION="true"
$env:MCP_COMPRESSION_LEVEL="6"

# Metadata compression
$env:MCP_METADATA_COMPRESSION="true"
```

## 🔒 **Security Best Practices**

### **Production Security**

#### **Network Security**
```bash
# Firewall configuration
$env:MCP_ALLOWED_IPS="10.0.0.0/8,172.16.0.0/12"
$env:MCP_BLOCKED_IPS="192.168.1.100"

# SSL/TLS
$env:MCP_SSL_CERT="/etc/ssl/certs/markdown-mcp.crt"
$env:MCP_SSL_KEY="/etc/ssl/private/markdown-mcp.key"
$env:MCP_SSL_VERIFY_CLIENT="true"
```

#### **Application Security**
```bash
# Authentication
$env:MCP_ADMIN_TOKEN="$(openssl rand -hex 32)"
$env:MCP_JWT_SECRET="$(openssl rand -hex 64)"

# Authorization
$env:MCP_ENABLE_RBAC="true"
$env:MCP_DEFAULT_ROLE="read"

# Input validation
$env:MCP_MAX_QUERY_LENGTH="1000"
$env:MCP_SANITIZE_INPUT="true"
```

### **Data Protection**

#### **Encryption at Rest**
```bash
# Index encryption
$env:MCP_ENCRYPT_INDEX="true"
$env:MCP_ENCRYPTION_KEY_PATH="/etc/markdown-mcp/keys"

# Metadata encryption
$env:MCP_ENCRYPT_METADATA="true"
```

#### **Access Logging**
```bash
# Audit logging
$env:MCP_ENABLE_AUDIT_LOG="true"
$env:MCP_AUDIT_LOG_PATH="/var/log/markdown-mcp/audit.log"

# Search logging
$env:MCP_LOG_SEARCH_QUERIES="true"
$env:MCP_LOG_SEARCH_RESULTS="false"
```

## 📊 **Monitoring & Analytics**

### **System Metrics**

#### **Performance Metrics**
```bash
# Query performance
$env:MCP_TRACK_QUERY_LATENCY="true"
$env:MCP_QUERY_LATENCY_BUCKETS="0.1,0.5,1.0,5.0"

# Model performance
$env:MCP_TRACK_MODEL_METRICS="true"
$env:MCP_MODEL_METRICS_INTERVAL="60"
```

#### **Resource Metrics**
```bash
# System resources
$env:MCP_TRACK_CPU_USAGE="true"
$env:MCP_TRACK_MEMORY_USAGE="true"
$env:MCP_TRACK_DISK_IO="true"

# Index metrics
$env:MCP_TRACK_INDEX_SIZE="true"
$env:MCP_TRACK_INDEX_UPDATES="true"
```

### **Business Metrics**

#### **Usage Analytics**
```bash
# Search analytics
$env:MCP_TRACK_SEARCH_PATTERNS="true"
$env:MCP_POPULAR_QUERIES_RETENTION="90"

# User analytics
$env:MCP_TRACK_USER_SESSIONS="true"
$env:MCP_ANONYMIZE_USER_DATA="true"
```

## 🚀 **Advanced Deployment Scenarios**

### **Multi-Region Deployment**

```bash
# Region-specific configuration
$env:MCP_REGION="us-west-2"
$env:MCP_REPLICA_COUNT="3"
$env:MCP_LOAD_BALANCER="true"

# Cross-region synchronization
$env:MCP_ENABLE_REPLICATION="true"
$env:MCP_REPLICATION_INTERVAL="300"
```

### **High Availability Setup**

```bash
# Load balancing
$env:MCP_LOAD_BALANCER="nginx"
$env:MCP_HEALTH_CHECK_ENDPOINT="/health"

# Auto-scaling
$env:MCP_AUTO_SCALE="true"
$env:MCP_MIN_INSTANCES="2"
$env:MCP_MAX_INSTANCES="10"
```

### **Container Orchestration**

```bash
# Kubernetes configuration
$env:MCP_K8S_NAMESPACE="markdown-mcp"
$env:MCP_K8S_REPLICAS="3"
$env:MCP_K8S_SERVICE_TYPE="LoadBalancer"

# Docker Swarm
$env:MCP_SWARM_MODE="true"
$env:MCP_SWARM_REPLICAS="5"
```

## 🔧 **Troubleshooting Advanced Configurations**

### **Configuration Validation**

```python
from config import AdvancedConfig

# Validate configuration
config = AdvancedConfig()
is_valid, errors = config.validate()

if not is_valid:
    for error in errors:
        print(f"Configuration error: {error}")
```

### **Performance Debugging**

```python
# Enable debug logging
$env:MCP_LOG_LEVEL="DEBUG"
$env:MCP_ENABLE_PERFORMANCE_LOGGING="true"

# Profile execution
python -m cProfile build_index.py ./notes
```

### **Model Debugging**

```python
# Test model loading
from config import AdvancedConfig
config = AdvancedConfig()

try:
    model = config.get_model_config("all-mpnet-base-v2")
    print(f"Model loaded: {model.model_name}")
except Exception as e:
    print(f"Model loading failed: {e}")
```

## 📚 **Best Practices**

### **Development Workflow**

1. **Use development environment** for local development
2. **Test configurations** in staging before production
3. **Monitor performance metrics** during development
4. **Document custom configurations** for team sharing

### **Production Deployment**

1. **Use production environment** settings
2. **Enable security features** (SSL, authentication)
3. **Configure monitoring** and alerting
4. **Set up automated backups**
5. **Plan for scaling** and high availability

### **Performance Optimization**

1. **Choose appropriate models** based on use case
2. **Tune chunking strategies** for content type
3. **Optimize hardware usage** (GPU, CPU, memory)
4. **Monitor and adjust** based on metrics
5. **Implement caching** for frequently accessed data

### **Security Implementation**

1. **Use strong authentication** tokens
2. **Implement network restrictions**
3. **Enable encryption** for sensitive data
4. **Regular security audits** and updates
5. **Monitor access logs** for suspicious activity

This advanced configuration system provides enterprise-grade flexibility while maintaining ease of use for basic deployments. 🚀

## **AI Features Configuration**

#### **Google Gemini Setup**
```bash
# Required for QA and note generation
export GOOGLE_API_KEY="your_google_api_key_here"

# Get API key from: https://makersuite.google.com/app/apikey
```

#### **AI Model Configuration**
```bash
# Default model for AI features
export MCP_AI_MODEL="gemini-1.5-pro"

# Alternative models
export MCP_AI_MODEL="gemini-pro"  # Older model
```

#### **AI Request Settings**
```bash
# Maximum tokens for AI responses
export MCP_AI_MAX_TOKENS="1000"

# Temperature for response creativity (0.0-1.0)
export MCP_AI_TEMPERATURE="0.7"

# Retry settings for API failures
export MCP_AI_MAX_RETRIES="3"
export MCP_AI_RETRY_DELAY="2"
```

#### **Context Configuration**
```bash
# Maximum context chunks for QA
export MCP_QA_MAX_CHUNKS="5"

# Context relevance threshold
export MCP_QA_THRESHOLD="0.7"
```

#### **Note Generation Settings**
```bash
# Default output directory for generated notes
export MCP_GENERATE_OUTPUT_DIR="./notes"

# Auto-save generated notes
export MCP_GENERATE_AUTO_SAVE="true"

# Include timestamps in filenames
export MCP_GENERATE_TIMESTAMP="true"
```
