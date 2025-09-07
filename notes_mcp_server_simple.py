#!/usr/bin/env python3
"""
Simple NotesMCPServer implementation for CLI fallback
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

class NotesMCPServer:
    """Simple MCP server implementation for CLI"""

    def __init__(self, index_file: str = "notes_index.npz",
                 meta_file: str = "notes_meta.json",
                 notes_root: str = "./notes"):
        self.index_file = Path(index_file)
        self.meta_file = Path(meta_file)
        self.notes_root = Path(notes_root)

    def search_notes(self, query: str, top_k: int = 10, threshold: float = 0.0) -> List[Dict]:
        """Search notes using simple text matching"""
        results = []
        notes_dir = self.notes_root

        if not notes_dir.exists():
            return results

        query_lower = query.lower()

        for file_path in notes_dir.glob("*.md"):
            try:
                content = file_path.read_text(encoding='utf-8')
                content_lower = content.lower()

                # Simple relevance scoring
                if query_lower in content_lower:
                    # Count occurrences for scoring
                    score = content_lower.count(query_lower) / len(content.split())

                    # Find context around the query
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if query_lower in line.lower():
                            # Get surrounding context
                            start = max(0, i - 2)
                            end = min(len(lines), i + 3)
                            context = '\n'.join(lines[start:end])
                            break
                    else:
                        context = content[:200] + "..."

                    results.append({
                        'file': file_path.name,
                        'score': min(score, 1.0),  # Cap at 1.0
                        'text': context,
                        'chunk_id': 0
                    })

            except Exception as e:
                print(f"⚠️  Error reading {file_path}: {e}")
                continue

        # Sort by score and limit results
        results.sort(key=lambda x: x['score'], reverse=True)

        # Apply threshold
        if threshold > 0:
            results = [r for r in results if r['score'] >= threshold]

        return results[:top_k]

    def run(self, host: str = "127.0.0.1", port: int = 8181):
        """Run the MCP server"""
        print(f"🚀 Simple MCP Server running on {host}:{port}")
        print("💡 This is a basic implementation for CLI testing")
        print("💡 Press Ctrl+C to stop")

        try:
            # Simple server loop
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")

# Simple admin app for compatibility
class SimpleAdminApp:
    def run(self, host: str = "127.0.0.1", port: int = 8181, debug: bool = False):
        print(f"🌐 Simple Admin Interface on {host}:{port}")

def create_admin_app(server):
    """Create admin app for compatibility"""
    return SimpleAdminApp()