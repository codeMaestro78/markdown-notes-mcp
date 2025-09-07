"""
Build embeddings index from Markdown notes.

This script processes Markdown files, extracts text content, chunks it intelligently,
and generates semantic embeddings for fast similarity search.
"""

import argparse
import json
import logging
import re
import sys
import time
import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Import advanced configuration system
from config import AdvancedConfig

try:
    import frontmatter
except ImportError:
    print("Error: python-frontmatter is required. Install with: pip install python-frontmatter")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Error: sentence-transformers is required. Install with: pip install sentence-transformers")
    print(f"Import error: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('build_index.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


def strip_frontmatter_and_render(md_text: str) -> str:
    """
    Remove frontmatter from Markdown text and return the content.

    Args:
        md_text: Raw Markdown text

    Returns:
        Text content without frontmatter
    """
    if not md_text or not md_text.strip():
        return ""

    if md_text.strip().startswith('---'):
        parts = md_text.split('---', 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return md_text.strip()


def md_to_plain(md_text: str) -> str:
    """
    Convert Markdown to plain text by removing formatting.

    Args:
        md_text: Markdown text

    Returns:
        Plain text version
    """
    if not md_text:
        return ""

    try:
        # Remove links but keep text
        s = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', md_text)
        # Remove code blocks, HTML tags, and Markdown formatting
        s = re.sub(r'```.*?```|<[^>]+>|[#>*_`~-]+', ' ', s, flags=re.DOTALL)
        # Normalize whitespace
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    except Exception as e:
        logger.warning(f"Error processing Markdown: {e}")
        return md_text  # Return original if processing fails


def save_index_npz(embeddings: np.ndarray, meta: List[dict], out_index: Path, out_meta: Path):
    """
    Save embeddings and metadata to compressed files.

    Args:
        embeddings: Numpy array of embeddings
        meta: List of metadata dictionaries
        out_index: Path for the embeddings file
        out_meta: Path for the metadata file
    """
    try:
        logger.info(f"Saving embeddings to {out_index}")
        np.savez_compressed(str(out_index), embeddings=embeddings)

        logger.info(f"Saving metadata to {out_meta}")
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info("Index and metadata saved successfully")

    except Exception as e:
        logger.error(f"Failed to save index: {e}")
        raise RuntimeError(f"Save operation failed: {e}") from e


def chunk_text(text: str, words_per_chunk=None, overlap=None, config_manager=None) -> List[Tuple[int, str]]:
    """
    Split text into overlapping chunks for better search granularity.

    Args:
        text: The text to chunk
        words_per_chunk: Number of words per chunk (uses config default if None)
        overlap: Number of overlapping words between chunks (uses config default if None)
        config_manager: AdvancedConfig instance

    Returns:
        List of (chunk_id, chunk_text) tuples
    """
    if config_manager is None:
        config_manager = AdvancedConfig()

    if words_per_chunk is None:
        words_per_chunk = config_manager.chunk_size
    if overlap is None:
        overlap = config_manager.overlap

    words = text.split()
    chunks = []
    i = 0
    cid = 0
    n = len(words)
    if n == 0:
        return []

    while i < n:
        end = min(n, i + words_per_chunk)
        chunk = " ".join(words[i:end]).strip()
        if chunk:
            chunks.append((cid, chunk))
            cid += 1
        if end == n:
            break
        i = end - overlap
    return chunks


def collect_chunks(notes_root: Path, config_manager=None) -> Tuple[List[str], List[dict]]:
    """
    Collect and process all Markdown files in the notes directory.

    Args:
        notes_root: Root directory containing Markdown files
        config_manager: AdvancedConfig instance

    Returns:
        Tuple of (documents, metadata)
    """
    if config_manager is None:
        config_manager = AdvancedConfig()

    docs = []
    metas = []
    processed_files = 0
    skipped_files = 0

    logger.info(f"Scanning directory: {notes_root}")

    try:
        for p in sorted(notes_root.rglob("*.md")):
            try:
                # Check file size
                if p.stat().st_size > config_manager.max_file_size:
                    logger.warning(f"Skipping large file: {p} ({p.stat().st_size} bytes)")
                    skipped_files += 1
                    continue

                logger.debug(f"Processing: {p}")

                # Load file content
                try:
                    post = frontmatter.load(p)
                    body = post.content if hasattr(post, 'content') else p.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    logger.warning(f"Skipping file with encoding issues: {p}")
                    skipped_files += 1
                    continue
                except Exception as e:
                    logger.warning(f"Error loading {p}: {e}, using plain text")
                    body = p.read_text(encoding="utf-8")

                # Process text
                text = strip_frontmatter_and_render(body)
                text = md_to_plain(text)

                if not text.strip():
                    logger.debug(f"Skipping empty file: {p}")
                    skipped_files += 1
                    continue

                # Split by headings first
                sections = re.split(r'\n#{1,6}\s+', text)
                if not sections:
                    sections = [text]

                file_chunks = 0
                for sec in sections:
                    sec = sec.strip()
                    if not sec:
                        continue

                    for cid, chunk in chunk_text(sec, config_manager=config_manager):
                        # Create unique ID
                        id_hash = hashlib.sha1(f"{p}:{cid}".encode()).hexdigest()

                        metas.append({
                            'id': id_hash,
                            'file': str(p.relative_to(notes_root)),
                            'chunk_id': int(cid),
                            'text': chunk[:4000]  # Limit text length
                        })
                        docs.append(chunk)
                        file_chunks += 1

                processed_files += 1
                logger.debug(f"Processed {p.name}: {file_chunks} chunks")

            except Exception as e:
                logger.error(f"Error processing {p}: {e}")
                skipped_files += 1
                continue

    except Exception as e:
        logger.error(f"Error scanning directory {notes_root}: {e}")
        raise

    logger.info(f"Processed {processed_files} files, skipped {skipped_files}")
    return docs, metas


def encode_documents(docs: List[str], model_name=None, batch_size=None, config_manager=None) -> np.ndarray:
    """
    Encode documents into embeddings using SentenceTransformer.

    Args:
        docs: List of text documents to encode
        model_name: Name of the model to use (uses config default if None)
        batch_size: Batch size for encoding (uses config default if None)
        config_manager: AdvancedConfig instance

    Returns:
        Normalized embeddings as numpy array
    """
    if config_manager is None:
        config_manager = AdvancedConfig()

    if model_name is None:
        model_name = config_manager.model_name
    if batch_size is None:
        batch_size = config_manager.batch_size

    try:
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        start_time = time.time()
        model = SentenceTransformer(model_name)
        model_load_time = time.time() - start_time
        logger.info(".2f")

        logger.info(f"Encoding {len(docs)} documents in batches of {batch_size}")
        embs = []
        encoding_start = time.time()

        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(docs) - 1) // batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}")

            batch_start = time.time()
            arr = model.encode(batch, convert_to_numpy=True)
            embs.append(arr.astype("float32"))
            batch_time = time.time() - batch_start
            logger.debug(".2f")

        embeddings = np.vstack(embs)
        encoding_time = time.time() - encoding_start

        # L2 Normalization
        logger.info("Normalizing embeddings")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
        embeddings = embeddings.astype("float32")

        logger.info(".2f")
        logger.info(f"Embeddings shape: {embeddings.shape}")

        return embeddings

    except Exception as e:
        logger.error(f"Failed to encode documents: {e}")
        raise RuntimeError(f"Document encoding failed: {e}") from e


def main():
    """Main function to build the embeddings index."""
    # Initialize configuration manager
    config_manager = AdvancedConfig()

    parser = argparse.ArgumentParser(
        description="Build embeddings index from Markdown notes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_index.py ./notes
  python build_index.py ./notes --index custom_index.npz --meta custom_meta.json

Environment variables:
  MCP_MODEL_NAME      - Model name (default: all-MiniLM-L6-v2)
  MCP_CHUNK_SIZE      - Words per chunk (default: 150)
  MCP_OVERLAP         - Overlap between chunks (default: 30)
  MCP_BATCH_SIZE      - Encoding batch size (default: 64)
  MCP_NOTES_ROOT      - Default notes directory (default: ./notes)
        """
    )
    parser.add_argument("notes_root", nargs='?', help="Root folder containing markdown notes")
    parser.add_argument("--index", help=f"Output index file (default: {config_manager.index_file})")
    parser.add_argument("--meta", help=f"Output metadata file (default: {config_manager.meta_file})")
    parser.add_argument("--model", help=f"SentenceTransformer model (default: {config_manager.model_name})")

    args = parser.parse_args()

    # Use config defaults if not specified
    notes_root_path = Path(args.notes_root) if args.notes_root else config_manager.notes_root
    index_file = args.index or config_manager.index_file
    meta_file = args.meta or config_manager.meta_file
    model_name = args.model or config_manager.model_name

    try:
        # Validate inputs
        if not notes_root_path.exists():
            raise FileNotFoundError(f"Notes root directory not found: {notes_root_path}")

        if not notes_root_path.is_dir():
            raise ValueError(f"Notes root must be a directory: {notes_root_path}")

        logger.info("=== Starting Index Build ===")
        logger.info(f"Notes root: {notes_root_path.absolute()}")
        logger.info(f"Index file: {index_file}")
        logger.info(f"Meta file: {meta_file}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Chunk size: {config_manager.chunk_size}, Overlap: {config_manager.overlap}")

        # Collect chunks
        logger.info("Collecting and processing documents...")
        start_time = time.time()
        docs, meta = collect_chunks(notes_root_path, config_manager)

        if not docs:
            logger.warning("No documents found to index")
            return

        collection_time = time.time() - start_time
        logger.info(".2f")
        logger.info(f"Total chunks: {len(docs)}")
        logger.info(f"Files processed: {len(set(m['file'] for m in meta))}")

        # Encode documents
        logger.info("Encoding documents into embeddings...")
        embeddings = encode_documents(docs, model_name, config_manager=config_manager)

        # Save results
        logger.info("Saving index and metadata...")
        save_index_npz(embeddings, meta, Path(index_file), Path(meta_file))

        total_time = time.time() - start_time
        logger.info("=== Index Build Complete ===")
        logger.info(".2f")
        logger.info(f"Index saved: {index_file}")
        logger.info(f"Metadata saved: {meta_file}")

    except Exception as e:
        logger.error(f"Index build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
