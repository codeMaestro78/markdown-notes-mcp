# markdown-notes-mcp

A semantic search system for Markdown notes with Model Context Protocol (MCP) server integration. Build searchable embeddings indexes from your Markdown files and expose them via JSON-RPC 2.0 over stdio for LLM integrations.

## Features

- **Semantic Search**: Uses sentence-transformers (all-MiniLM-L6-v2) for embedding-based similarity search
- **Hybrid Search**: Combines semantic embeddings with TF-IDF lexical search
- **Fast Indexing**: Optional FAISS integration for efficient vector search
- **MCP Protocol**: JSON-RPC 2.0 server over stdio with Content-Length framing
- **Chunking**: Intelligent text chunking with configurable overlap for better search granularity
- **Admin API**: HTTP endpoints for health checks and reindexing

## Project Structure

```
markdown-notes-mcp/
├── notes/                      # Your Markdown notes go here
│   ├── example.md
│   └── pca_notes.md
├── build_index.py              # Build embeddings index from notes
├── notes_mcp_server.py         # Main MCP server (JSON-RPC over stdio)
├── test_harness_client.py      # Test client for MCP server
├── admin_http.py               # Optional HTTP admin API
├── tests/
│   └── test_end_to_end.py      # End-to-end tests
├── requirements.txt            # Python dependencies
├── README.md
└── .copilot/
    └── mcp.json                # MCP configuration
```

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:
  - sentence-transformers
  - numpy
  - scikit-learn
  - python-frontmatter
  - fastapi (for admin API)
  - uvicorn (for admin API)
  - faiss-cpu (optional, for faster search)

## Setup

1. **Clone and setup environment:**
```bash
cd C:\Users\Devarshi\PycharmProjects\markdown-notes-mcp
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. **Add your notes:**
```bash
# Place your .md files in the notes/ directory
# Supports frontmatter and standard Markdown
```

3. **Build the search index:**
```bash
python build_index.py ./notes
# Creates: notes_index.npz and notes_meta.json
```

## Usage

### Building the Index

```bash
# Basic usage
python build_index.py ./notes

# Custom output paths
python build_index.py ./notes --index custom_index.npz --meta custom_meta.json
```

The indexing process:
- Extracts text from Markdown files (strips frontmatter)
- Chunks text by headings and word count (150 words/chunk, 30 word overlap)
- Generates embeddings using all-MiniLM-L6-v2
- Saves compressed index (.npz) and metadata (.json)

### MCP Server

Start the JSON-RPC server over stdio:

```bash
python notes_mcp_server.py --index notes_index.npz --meta notes_meta.json --notes_root ./notes
```

**Available RPC Methods:**
- `list_notes` - Get list of all indexed files
- `get_note_content` - Get full content of a note by path
- `search_notes` - Hybrid semantic + lexical search
- `health_check` - Server health status

**Search Parameters:**
```json
{
  "query": "your search query",
  "top_k": 5,
  "lexical_weight": 0.3
}
```

### Test Client

Test the MCP server functionality:

```bash
python test_harness_client.py
```

This will:
1. Start the MCP server as subprocess
2. Send list_notes request
3. Fetch content of first note
4. Perform semantic search for "PCA"

### Admin HTTP API

Optional web interface for administration:

```bash
python admin_http.py
# Server starts on http://127.0.0.1:8181
```

**Endpoints:**
- `GET /health` - Health check
- `GET /metrics` - Index statistics  
- `POST /reindex` - Rebuild index (requires X-ADMIN-TOKEN header)

Set admin token via environment:
```bash
set NOTES_ADMIN_TOKEN=your-secret-token
python admin_http.py
```

## MCP Integration

### Claude Desktop Configuration

Add to your MCP settings (`.copilot/mcp.json`):

```json
{
  "mcpServers": {
    "markdown-notes": {
      "command": "python",
      "args": [
        "C:\\Users\\Devarshi\\PycharmProjects\\markdown-notes-mcp\\notes_mcp_server.py",
        "--index", "notes_index.npz",
        "--meta", "notes_meta.json", 
        "--notes_root", "./notes"
      ],
      "env": {
        "PYTHONPATH": "C:\\Users\\Devarshi\\PycharmProjects\\markdown-notes-mcp"
      }
    }
  }
}
```

### Available Tools for LLMs

When integrated, LLMs can:
- **Search your notes** semantically for relevant content
- **List all available** notes and documents
- **Read full content** of specific notes
- **Combine results** from multiple searches

## Configuration

### Indexing Parameters

Edit `build_index.py` constants:
- `DEFAULT_CHUNK_WORD = 150` - Words per chunk
- `DEFAULT_OVERLAP = 30` - Overlap between chunks
- `BATCH_SIZE = 64` - Encoding batch size
- `MODEL_NAME = "all-MiniLM-L6-v2"` - Embedding model

### Search Tuning

Adjust hybrid search weights in `search_notes`:
- `lexical_weight=0.3` - Balance between semantic (0.7) and lexical (0.3) search
- Higher lexical_weight favors exact keyword matches
- Lower lexical_weight favors semantic similarity

## Performance

- **FAISS**: Install `faiss-cpu` for faster similarity search on large indexes
- **Chunking**: Smaller chunks = more precise results, larger chunks = more context
- **Batch Size**: Increase for faster indexing on GPUs (if available)

## Testing

Run the end-to-end test:

```bash
pytest tests/test_end_to_end.py -v
```

Or run individual components:

```bash
# Test MCP server manually
python test_harness_client.py

# Test admin API
curl http://127.0.0.1:8181/health
```

## Troubleshooting

**Index files missing:**
```bash
# Rebuild the index
python build_index.py ./notes
```

**Import errors:**
```bash
# Install missing dependencies
pip install sentence-transformers python-frontmatter
```

**MCP server not responding:**
- Check that index and meta files exist
- Verify notes_root path is correct
- Test with `test_harness_client.py` first

**Search returns no results:**
- Verify your notes contain text content
- Try different search queries
- Check if index was built successfully

## License

MIT License - feel free to modify and distribute.

---

**Quick Start:**
1. `pip install -r requirements.txt`
2. Add `.md` files to `notes/` folder  
3. `python build_index.py ./notes`
4. `python notes_mcp_server.py --index notes_index.npz --meta notes_meta.json --notes_root ./notes`
