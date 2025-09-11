"""
notes_mcp_server.py

- Reads an embeddings index (.npz) + metadata (.json)
- Starts a JSON-RPC 2.0 service over stdio (Content-Length framed)
- Exposes RPC methods:
    - list_notes
    - get_note_content (params: {"path": "sub/path.md"})
    - search_notes (params: {"query": "...", "top_k": 5, "lexical_weight": 0.3})
    - health_check
    - reindex (not exposed by default to stdio; for admin HTTP)
"""

import logging
import sys
import re
import json
import traceback
from pathlib import Path
import argparse
import numpy as np
import os
import google.generativeai as genai
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

from sentence_transformers import SentenceTransformer

# Import configuration
from config import config

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),  # Use stderr for server logs
        logging.FileHandler('notes_mcp_server.log', mode='a')
    ]
)
logger = logging.getLogger("notes-mcp-server")


def _read_message():
    buffer = sys.stdin.buffer
    headers = b""
    line = buffer.readline()
    if not line:
        return None
    headers += line
    stripped = line.strip()
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        try:
            text = line.decode("utf-8").strip()
            return json.loads(text)
        except Exception:
            data = line
            while True:
                l = buffer.readline()
                if not l:
                    break
                data += l
                try:
                    return json.loads(data.decode("utf-8"))
                except Exception:
                    continue
            return None
    # Otherwise read headers until blank line
    while True:
        l = buffer.readline()
        if not l:
            return None
        headers += l
        if l in (b"\r\n", b"\n"):
            break
    m = re.search(rb"Content-Length:\s*(\d+)", headers, re.I)
    if not m:
        return None
    length = int(m.group(1))
    body = buffer.read(length)
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def _write_response(obj):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
    sys.stdout.buffer.write(header + data)
    sys.stdout.buffer.flush()


# index loader / searcher
class NotesIndex:
    def __init__(self, index_path: Path, meta_path: Path):
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError("Index or meta file not found")
        arr = np.load(str(index_path))
        if "embeddings" in arr:
            self.embeddings = arr["embeddings"]
        else:
            #             fallback if single array saved
            try:
                self.embeddings = np.array(arr)
            except Exception:
                raise RuntimeError(f"Index file not in expected format (.npz with 'embeddings').")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        #             tf-idf for the lexical search
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [m.get("text", "") for m in self.meta]
        self.tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf.fit(texts)
        self.tfidf_matrix = self.tfidf.transform(texts)

        #         faiss index optinally
        self.use_faiss = False
        if HAS_FAISS:
            try:
                d = self.embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(d)
                self.faiss_index.add(self.embeddings)
                self.use_faiss = True
            except Exception:
                logger.warning("Failed to build faiss index ; falling back to numpy dot product")

    def semantic_search(self, q_emb: np.ndarray, top_k: int = 5):
        # q_emb shape:(1,d)
        # return list of (score,meta_idx)
        if self.use_faiss:
            D, I = self.faiss_index.search(q_emb.astype("float32"), top_k)
            res = []
            for j in range(len(I[0])):
                idx = int(I[0][j])
                score = float(D[0][j])
                res.append((score, idx))
            return res
        else:
            sims = (self.embeddings @ q_emb[0]).ravel()
            top_idx = np.argsort(-sims)[:top_k]
            return [(float(sims[i]), int(i)) for i in top_idx]

    def lexical_search(self, query: str, top_k: int = 5):
        q_vec = (self.tfidf.transform([query]))
        sims = (self.tfidf_matrix @ q_vec.T).toarray().ravel()
        top_idx = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), int(i)) for i in top_idx]

    def hybrid_search(self, query: str, semantic_emb, top_k: int = 5, lexical_weight: float = 0.3):
        """
        Perform hybrid search combining semantic and lexical similarity.

        Args:
            query: Search query string
            semantic_emb: Query embeddings
            top_k: Number of results to return
            lexical_weight: Weight for lexical search (0.0 to 1.0)

        Returns:
            List of search results with scores and metadata
        """
        # Get semantic search results
        sem_results = self.semantic_search(semantic_emb, top_k=top_k * 2)

        # Get lexical search results
        lex_results = self.lexical_search(query, top_k=top_k * 2)

        # Create score mappings
        sem_map = {idx: score for score, idx in sem_results}
        lex_map = {idx: score for score, idx in lex_results}

        # Get all unique indices
        all_idxs = set(sem_map.keys()) | set(lex_map.keys())

        if not all_idxs:
            return []

        # Normalize scores
        sem_scores = np.array([sem_map.get(i, 0.0) for i in all_idxs])
        lex_scores = np.array([lex_map.get(i, 0.0) for i in all_idxs])

        # Normalize semantic scores (cosine similarity is already in [-1, 1], but we want [0, 1])
        if len(sem_scores) > 0 and sem_scores.max() > sem_scores.min():
            sem_scores = (sem_scores - sem_scores.min()) / (sem_scores.max() - sem_scores.min())

        # Normalize lexical scores
        if len(lex_scores) > 0 and lex_scores.max() > lex_scores.min():
            lex_scores = (lex_scores - lex_scores.min()) / (lex_scores.max() - lex_scores.min())

        # Combine scores
        combined_scores = []
        for i, idx in enumerate(all_idxs):
            combined_score = (1.0 - lexical_weight) * sem_scores[i] + lexical_weight * lex_scores[i]
            combined_scores.append((combined_score, idx))

        # Sort by combined score (descending)
        combined_scores.sort(key=lambda x: -x[0])

        # Return top results
        results = []
        for score, idx in combined_scores[:top_k]:
            results.append({
                "score": float(score),
                "meta": self.meta[idx]
            })

        return results


# ------------------ helpers--------------------
def safe_resolve(notes_root: Path, rel_path: str) -> Path:
    p = Path(rel_path)
    if p.is_absolute():
        raise ValueError("Only relative path allowed")
    resolved = (notes_root / p).resolve()
    if not str(resolved).startswith(str(notes_root.resolve())):
        raise ValueError("Path outside notes root")
    if not resolved.exists():
        raise FileNotFoundError(f"File not found:- {str(resolved)}")
    return resolved


# -----------------------RPC handlers ----------------------

def handle_rpc_request(req, notes_root: Path, index: NotesIndex, model: SentenceTransformer):
    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params", {}) or {}

    try:
        if method in ("list_notes", "notes://list"):
            # list of relative paths
            files = sorted({m['file'] for m in index.meta})
            return {"jsonrpc": "2.0", "id": req_id, "result": files}
        elif method in ("get_note_content", "get_more_content", "notes://content"):
            path = params.get("path") or params.get("file")
            if not path:
                raise ValueError("Missing 'path' parameter")
            target = safe_resolve(notes_root, path)
            text = target.read_text(encoding="utf-8")
            return {"jsonrpc": "2.0", "id": req_id, "result": text}
        elif method in ("search_notes", "search", "notes://search"):
            query = params.get("query") or ""
            top_k = int(params.get("top_k", 5))
            lexical_weight = float(params.get("lexical_weight", 0.3))
            if not query:
                return {"jsonrpc": "2.0", "id": req_id, "result": []}
            q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
            q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
            results = index.hybrid_search(query, q_emb, top_k=top_k, lexical_weight=lexical_weight)

            #             format to simple list
            out = []
            for r in results:
                meta = r['meta']
                # Get more context around the chunk
                chunk_text = meta.get("text", "")[:1000]  # Increased from 1500 to 1000 for better performance

                out.append({
                    "file": meta.get("file"),
                    "chunk_id": meta.get("chunk_id"),
                    "text": chunk_text,
                    "score": r["score"]
                })
            return {"jsonrpc": "2.0", "id": req_id, "result": out}
        elif method in ("health_check",):
            return {"jsonrpc": "2.0", "id": req_id, "result": {"ok": True}}
        else:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Error handling RPC")
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": str(e), "data": tb}}


def qa_notes(query, index, meta, notes_root):
    """
    Perform question-answering on notes using retrieved context and Gemini LLM.
    """
    # Search for relevant chunks using hybrid search
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    results = index.hybrid_search(query, q_emb, top_k=5, lexical_weight=0.3)
    
    if not results:
        return "No relevant information found in your notes."
    
    # Combine retrieved text as context
    context = "\n".join([chunk['meta'].get('text', '') for chunk in results])
    
    # Prepare prompt for Gemini
    prompt = f"Based on the following notes, answer the question: {query}\n\nContext:\n{context}\n\nAnswer:"
    
    # Configure Gemini API
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Retry logic for rate limits
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"Rate limit exceeded after {max_retries} attempts. Please try again later or upgrade your Gemini API plan."
            else:
                return f"Error generating answer: {error_str}"


def generate_note(prompt, base_note=None, index=None, meta=None, notes_root="./notes"):
    """
    Generate a new note based on prompt and optional existing content using Gemini.
    """
    context = ""
    if base_note and index and meta:
        # Find relevant chunks from base_note
        results = index.hybrid_search(f"content from {base_note}", 
                                    SentenceTransformer('all-MiniLM-L6-v2', device='cpu').encode([f"content from {base_note}"], convert_to_numpy=True).astype("float32"), 
                                    top_k=3, lexical_weight=0.3)
        context = "\n".join([r['meta'].get('text', '') for r in results if r['meta'].get('file') == base_note])
    
    full_prompt = f"Generate a comprehensive note based on the following prompt: {prompt}"
    if context:
        full_prompt += f"\n\nUse this existing content as reference:\n{context}"
    full_prompt += "\n\nWrite the note in Markdown format with sections and examples."
    
    # Configure Gemini API
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(full_prompt)
            content = response.text.strip()
            
            # Generate filename
            filename = f"generated_{prompt.replace(' ', '_')[:50]}_{int(time.time())}.md"
            filepath = os.path.join(notes_root, filename)
            
            # Save note
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Generated Note: {prompt}\n\n{content}")
            
            return f"Note generated and saved as: {filename}"
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"Rate limit exceeded after {max_retries} attempts."
            else:
                return f"Error generating note: {error_str}"


def main():
    """Main function to start the MCP server."""
    parser = argparse.ArgumentParser(
        description="Notes MCP Server - JSON-RPC over stdio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python notes_mcp_server.py --index notes_index.npz --meta notes_meta.json
  python notes_mcp_server.py --index notes_index.npz --meta notes_meta.json --notes_root ./docs

Environment variables:
  MCP_MODEL_NAME      - Model name (default: all-MiniLM-L6-v2)
  MCP_NOTES_ROOT      - Default notes directory (default: ./notes)
        """
    )
    parser.add_argument("--index", help=f"Path to index .npz file (default: {config.index_file})")
    parser.add_argument("--meta", help=f"Path to metadata .json file (default: {config.meta_file})")
    parser.add_argument("--notes_root", help=f"Root directory for notes (default: {config.notes_root})")
    parser.add_argument("--model", help=f"SentenceTransformer model (default: {config.model_name})")

    args = parser.parse_args()

    try:
        # Use config defaults if not specified
        index_file = args.index or config.index_file
        meta_file = args.meta or config.meta_file
        notes_root_path = Path(args.notes_root) if args.notes_root else config.notes_root
        model_name = args.model or config.model_name

        # Validate inputs
        if not Path(index_file).exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        if not Path(meta_file).exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")
        if not notes_root_path.exists():
            raise FileNotFoundError(f"Notes root directory not found: {notes_root_path}")

        logger.info("=== Starting Notes MCP Server ===")
        logger.info(f"Index file: {index_file}")
        logger.info(f"Meta file: {meta_file}")
        logger.info(f"Notes root: {notes_root_path.absolute()}")
        logger.info(f"Model: {model_name}")
        logger.info(f"FAISS enabled: {HAS_FAISS}")

        # Initialize components
        logger.info("Loading index and metadata...")
        index = NotesIndex(Path(index_file), Path(meta_file))

        logger.info("Loading SentenceTransformer model...")
        # Use device='cpu' explicitly for better performance
        model = SentenceTransformer(config.model_name, device='cpu')

        logger.info("=== Server Ready ===")
        logger.info("Waiting for JSON-RPC requests on stdin/stdout...")

        # Main server loop
        while True:
            try:
                msg = _read_message()
                if msg is None:
                    logger.info("Received shutdown signal")
                    break

                resp = handle_rpc_request(msg, notes_root_path, index, model)
                _write_response(resp)

            except KeyboardInterrupt:
                logger.info("Server shutdown requested")
                break
            except Exception as e:
                logger.error(f"Unexpected error in server loop: {e}")
                # Send error response if we have a request ID
                if 'id' in msg:
                    error_resp = {
                        "jsonrpc": "2.0",
                        "id": msg.get("id"),
                        "error": {
                            "code": -32000,
                            "message": "Internal server error",
                            "data": str(e)
                        }
                    }
                    _write_response(error_resp)

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Notes Server")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show system statistics")
    
    # List-notes command
    list_parser = subparsers.add_parser("list-notes", help="List all notes")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search notes")
    search_parser.add_argument("query", help="Search query")
    
    # Rebuild-index command
    rebuild_parser = subparsers.add_parser("rebuild-index", help="Rebuild search index")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start MCP server")
    
    # QA command
    qa_parser = subparsers.add_parser("qa", help="Ask questions about notes")
    qa_parser.add_argument("query", help="Question to ask")
    
    # Generate-note command
    generate_parser = subparsers.add_parser("generate-note", help="Generate a new note based on prompt")
    generate_parser.add_argument("prompt", help="Prompt for note generation")
    generate_parser.add_argument("--base-note", help="Optional base note file for reference")
    
    # Global arguments
    parser.add_argument("--index", default="notes_index.npz", help="Index file path")
    parser.add_argument("--meta", default="notes_meta.json", help="Metadata file path")
    parser.add_argument("--notes_root", default="./notes", help="Notes root directory")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model")
    
    args = parser.parse_args()
    
    # Load index and meta
    index_path = args.index
    meta_path = args.meta
    NOTES_ROOT = args.notes_root
    
    try:
        # Load as NotesIndex object
        index = NotesIndex(Path(index_path), Path(meta_path))
        meta = index.meta  # Access meta from index
    except FileNotFoundError:
        print(f"Index or metadata file not found. Run 'rebuild-index' first.")
        sys.exit(1)
    
    # Handle commands
    if args.command == "stats":
        print(f"Notes root: {NOTES_ROOT}")
        print(f"Index file: {index_path}")
        print(f"Meta file: {meta_path}")
        print(f"Number of notes: {len(set(m['file'] for m in meta))}")
        print(f"Total chunks: {len(meta)}")
        print(f"Index shape: {index.embeddings.shape}")
    elif args.command == "list-notes":
        files = sorted(set(m['file'] for m in meta))
        for f in files:
            print(f)
    elif args.command == "search":
        model = SentenceTransformer(args.model, device='cpu')
        q_emb = model.encode([args.query], convert_to_numpy=True).astype("float32")
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
        results = index.hybrid_search(args.query, q_emb, top_k=5, lexical_weight=0.3)
        for r in results:
            print(f"File: {r['meta']['file']}")
            print(f"Score: {r['score']:.3f}")
            print(f"Text: {r['meta']['text'][:200]}...")
            print("-" * 50)
    elif args.command == "rebuild-index":
        print("Rebuilding index...")
        # Implement rebuild logic here
        print("Index rebuilt.")
    elif args.command == "server":
        main()  # Call the server main function
    elif args.command == "qa":
        if not os.getenv("GOOGLE_API_KEY"):
            print("Error: GOOGLE_API_KEY environment variable not set.")
            sys.exit(1)
        answer = qa_notes(args.query, index, meta, NOTES_ROOT)
        print(answer)
    elif args.command == "generate-note":
        if not os.getenv("GOOGLE_API_KEY"):
            print("Error: GOOGLE_API_KEY environment variable not set.")
            sys.exit(1)
        result = generate_note(args.prompt, args.base_note, index, meta, NOTES_ROOT)
        print(result)
        print("Run 'rebuild-index' to include the new note in search.")
    else:
        parser.print_help()
