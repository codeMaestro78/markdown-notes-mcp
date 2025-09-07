"""
Lightweight admin API for managing the markdown-notes-mcp system:
 - GET /health - Health check endpoint
 - POST /reindex - Rebuild search index (requires auth token)
 - GET /metrics - Index statistics and system metrics
"""

import subprocess
import sys
import logging
import json
from pathlib import Path
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Import configuration
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("admin-api")

app = FastAPI(
    title="Markdown Notes MCP Admin API",
    description="Administrative API for managing the notes search system",
    version="1.0.0"
)


class ReindexRequest(BaseModel):
    """Request model for reindexing operation."""
    notes_root: str
    index: Optional[str] = None
    meta: Optional[str] = None


@app.get("/health")
def health():
    return {"status": 'ok', "time": time.time()}


@app.post("/reindex")
def reindex(payload: ReindexRequest, x_admin_token: Optional[str] = Header(None)):
    """
    Rebuild the search index from notes.

    Requires X-ADMIN-TOKEN header for authentication.
    """
    if x_admin_token != config.admin_token:
        logger.warning("Unauthorized reindex attempt")
        raise HTTPException(status_code=403, detail="Invalid admin token")

    notes_root = payload.notes_root
    index = payload.index or config.index_file
    meta = payload.meta or config.meta_file

    logger.info(f"Starting reindex: notes_root={notes_root}, index={index}, meta={meta}")

    # Use the virtual environment Python if available
    python_exe = sys.executable

    cmd = [python_exe, "build_index.py", notes_root, "--index", index, "--meta", meta]

    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        p = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)  # 5 min timeout

        logger.info("Reindex completed successfully")
        return {
            "status": "ok",
            "stdout": p.stdout,
            "stderr": p.stderr,
            "returncode": p.returncode
        }

    except subprocess.TimeoutExpired:
        logger.error("Reindex operation timed out")
        raise HTTPException(status_code=408, detail="Reindex operation timed out")
    except subprocess.CalledProcessError as e:
        logger.error(f"Reindex failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Reindex failed",
                "stdout": e.stdout,
                "stderr": e.stderr,
                "returncode": e.returncode
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error during reindex: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/metrics")
def metrics():
    """
    Get system metrics and index statistics.
    """
    try:
        idx_path = Path(config.index_file)
        meta_path = Path(config.meta_file)

        if not idx_path.exists() or not meta_path.exists():
            return {
                "indexed": False,
                "message": "Index files not found",
                "expected_index": str(idx_path),
                "expected_meta": str(meta_path)
            }

        # Load embeddings
        arr = np.load(str(idx_path))
        embeddings = arr['embeddings']

        # Load metadata
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        # Calculate statistics
        num_files = len(set(m['file'] for m in meta))
        avg_chunk_length = sum(len(m['text']) for m in meta) / len(meta) if meta else 0

        return {
            "indexed": True,
            "num_vectors": int(embeddings.shape[0]),
            "vector_dimensions": int(embeddings.shape[1]),
            "num_files": num_files,
            "total_chunks": len(meta),
            "avg_chunk_length": round(avg_chunk_length, 1),
            "index_file_size_mb": round(idx_path.stat().st_size / (1024 * 1024), 2),
            "meta_file_size_mb": round(meta_path.stat().st_size / (1024 * 1024), 2),
            "last_modified": time.ctime(max(idx_path.stat().st_mtime, meta_path.stat().st_mtime))
        }

    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.get("/config")
def get_config():
    """
    Get current configuration (without sensitive data).
    """
    return {
        "model_name": config.model_name,
        "chunk_size": config.chunk_size,
        "overlap": config.overlap,
        "batch_size": config.batch_size,
        "index_file": config.index_file,
        "meta_file": config.meta_file,
        "notes_root": str(config.notes_root),
        "host": config.host,
        "port": config.port,
        "max_file_size": config.max_file_size
    }


if __name__ == "__main__":
    logger.info(f"Starting admin API server on {config.host}:{config.port}")
    logger.info(f"Admin token: {config.admin_token[:4]}****")  # Show first 4 chars only
    uvicorn.run(app, host=config.host, port=config.port)
