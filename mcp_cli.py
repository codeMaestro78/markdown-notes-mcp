#!/usr/bin/env python3
"""
mcp_cli.py - Advanced Command-Line Interface for Markdown Notes MCP

A powerful CLI tool for interacting with your markdown notes through the MCP server.
Provides search, content management, export, and analytics capabilities.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess
import time
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from config import AdvancedConfig
    # Fix the import - use the correct function name from build_index.py
    from build_index import main as build_index_main

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Try using: python mcp_cli_fixed.py instead")
    sys.exit(1)

class MCPCLI:
    """Advanced CLI for Markdown Notes MCP Server"""

    def __init__(self):
        self.config = AdvancedConfig()
        self.project_root = project_root

    def run(self):
        """Main CLI entry point"""
        parser = argparse.ArgumentParser(
            description="Advanced CLI for Markdown Notes MCP Server",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python mcp_cli.py search "machine learning" --format json --limit 5
  python mcp_cli.py add-note ./new_note.md --auto-tag
  python mcp_cli.py export-search "PCA" --format pdf
  python mcp_cli.py stats --period week
  python mcp_cli.py list-notes --sort modified
  python mcp_cli.py rebuild-index --model all-mpnet-base-v2
            """
        )

        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Search command
        search_parser = subparsers.add_parser('search', help='Search notes with advanced options')
        search_parser.add_argument('query', help='Search query')
        search_parser.add_argument('--format', choices=['text', 'json', 'table'], default='text',
                                 help='Output format (default: text)')
        search_parser.add_argument('--limit', type=int, default=10,
                                 help='Maximum number of results (default: 10)')
        search_parser.add_argument('--threshold', type=float, default=0.0,
                                 help='Minimum relevance threshold (default: 0.0)')
        search_parser.add_argument('--export', help='Export results to file (JSON format)')

        # Add note command
        add_parser = subparsers.add_parser('add-note', help='Add a new note to the collection')
        add_parser.add_argument('file', help='Path to markdown file to add')
        add_parser.add_argument('--auto-tag', action='store_true',
                              help='Automatically generate tags for the note')
        add_parser.add_argument('--rebuild', action='store_true',
                              help='Rebuild index after adding note')

        # Export search command
        export_parser = subparsers.add_parser('export-search', help='Export search results')
        export_parser.add_argument('query', help='Search query to export')
        export_parser.add_argument('--format', choices=['pdf', 'html', 'markdown', 'json'],
                                 default='pdf', help='Export format (default: pdf)')
        export_parser.add_argument('--output', help='Output file path')
        export_parser.add_argument('--limit', type=int, default=20,
                                 help='Maximum results to export (default: 20)')

        # Stats command
        stats_parser = subparsers.add_parser('stats', help='Show system statistics')
        stats_parser.add_argument('--period', choices=['day', 'week', 'month', 'all'],
                                default='week', help='Time period for statistics (default: week)')
        stats_parser.add_argument('--format', choices=['text', 'json'], default='text',
                                help='Output format (default: text)')

        # List notes command
        list_parser = subparsers.add_parser('list-notes', help='List all available notes')
        list_parser.add_argument('--sort', choices=['name', 'modified', 'size'],
                               default='name', help='Sort order (default: name)')
        list_parser.add_argument('--format', choices=['text', 'json'], default='text',
                               help='Output format (default: text)')

        # Rebuild index command
        rebuild_parser = subparsers.add_parser('rebuild-index', help='Rebuild the search index')
        rebuild_parser.add_argument('--model', help='Embedding model to use')
        rebuild_parser.add_argument('--chunk-size', type=int, help='Text chunk size')
        rebuild_parser.add_argument('--overlap', type=int, help='Chunk overlap size')
        rebuild_parser.add_argument('--force', action='store_true',
                                  help='Force rebuild even if files unchanged')
        rebuild_parser.add_argument('notes_root', nargs='?', default='./notes',
                                  help='Path to notes directory (default: ./notes)')

        # Server command
        server_parser = subparsers.add_parser('server', help='Start MCP server')
        server_parser.add_argument('--host', default='127.0.0.1', help='Server host')
        server_parser.add_argument('--port', type=int, default=8181, help='Server port')
        server_parser.add_argument('--no-admin', action='store_true',
                                 help='Disable admin HTTP interface')

        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return

        # Execute the appropriate command
        try:
            command_method = getattr(self, f'cmd_{args.command.replace("-", "_")}')
            command_method(args)
        except AttributeError:
            print(f"❌ Unknown command: {args.command}")
        except Exception as e:
            print(f"❌ Command execution failed: {e}")

    def cmd_search(self, args):
        """Handle search command"""
        print(f"🔍 Searching for: '{args.query}'")
        print(f"📊 Limit: {args.limit}, Format: {args.format}")
        print("-" * 50)

        try:
            # Try to import NotesMCPServer, fallback to simple implementation
            try:
                from notes_mcp_server import NotesMCPServer
                use_mcp = True
            except ImportError:
                try:
                    from notes_mcp_server_simple import NotesMCPServer
                    print("⚠️  Using simple MCP server implementation")
                    use_mcp = True
                except ImportError:
                    print("⚠️  MCP server not available, using simple text search")
                    use_mcp = False

            if use_mcp:
                # Use MCP server for search
                server = NotesMCPServer(
                    index_file=self.config.index_file,
                    meta_file=self.config.meta_file,
                    notes_root=self.config.notes_root
                )

                results = server.search_notes(args.query, top_k=args.limit)
            else:
                # Fallback to simple text search
                results = self._simple_text_search(args.query, args.limit)

            if not results:
                print("❌ No results found.")
                return

            # Filter by threshold
            if args.threshold > 0:
                results = [r for r in results if r.get('score', 0) >= args.threshold]

            # Format and display results
            if args.format == 'json':
                self._output_json(results)
            elif args.format == 'table':
                self._output_table(results)
            else:
                self._output_text(results)

            # Export results if requested
            if hasattr(args, 'export') and args.export:
                self._export_json(results, Path(args.export))
                print(f"📤 Exported {len(results)} results to {args.export}")

        except Exception as e:
            print(f"❌ Search error: {e}")
            import traceback
            traceback.print_exc()

    def cmd_add_note(self, args):
        """Handle add-note command"""
        file_path = Path(args.file)

        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return

        if file_path.suffix.lower() != '.md':
            print(f"❌ Not a markdown file: {file_path}")
            return

        print(f"📝 Adding note: {file_path}")

        try:
            # Copy file to notes directory
            notes_dir = self.config.notes_root
            notes_dir.mkdir(exist_ok=True)

            dest_file = notes_dir / file_path.name
            import shutil
            shutil.copy2(file_path, dest_file)

            print(f"✅ Copied to: {dest_file}")

            # Auto-tag if requested
            if args.auto_tag:
                print("🏷️  Generating tags...")
                tags = self._generate_tags(dest_file)
                if tags:
                    print(f"🏷️  Generated tags: {', '.join(tags)}")
                    self._add_tags_to_file(dest_file, tags)

            # Rebuild index if requested
            if args.rebuild:
                print("🔄 Rebuilding index...")
                self._rebuild_index()

            print("✅ Note added successfully!")

        except Exception as e:
            print(f"❌ Error adding note: {e}")
            import traceback
            traceback.print_exc()

    def cmd_export_search(self, args):
        """Handle export-search command"""
        print(f"📤 Exporting search results for: '{args.query}'")
        print(f"📄 Format: {args.format}, Limit: {args.limit}")

        try:
            # Try to use MCP server, fallback to simple search
            try:
                from notes_mcp_server import NotesMCPServer
                server = NotesMCPServer(
                    index_file=self.config.index_file,
                    meta_file=self.config.meta_file,
                    notes_root=self.config.notes_root
                )
                results = server.search_notes(args.query, top_k=args.limit)
            except ImportError:
                try:
                    from notes_mcp_server_simple import NotesMCPServer
                    server = NotesMCPServer()
                    results = server.search_notes(args.query, top_k=args.limit)
                    print("⚠️  Using simple MCP server implementation")
                except ImportError:
                    print("⚠️  MCP server not available, using simple text search")
                    results = self._simple_text_search(args.query, args.limit)

            if not results:
                print("❌ No results to export.")
                return

            # Determine output file
            if args.output:
                output_file = Path(args.output)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = Path(f"search_results_{timestamp}.{args.format}")

            # Export based on format
            if args.format == 'json':
                self._export_json(results, output_file)
            elif args.format == 'pdf':
                self._export_pdf(results, args.query, output_file)
            elif args.format == 'html':
                self._export_html(results, args.query, output_file)
            elif args.format == 'markdown':
                self._export_markdown(results, args.query, output_file)

            print(f"✅ Exported to: {output_file}")

        except Exception as e:
            print(f"❌ Export error: {e}")
            import traceback
            traceback.print_exc()

    def cmd_stats(self, args):
        """Handle stats command"""
        print(f"📊 System Statistics ({args.period})")
        print("-" * 50)

        try:
            stats = self._get_system_stats(args.period)

            if args.format == 'json':
                self._output_json(stats)
            else:
                self._output_stats_text(stats)

        except Exception as e:
            print(f"❌ Stats error: {e}")
            import traceback
            traceback.print_exc()

    def cmd_list_notes(self, args):
        """Handle list-notes command"""
        print("📋 Available Notes")
        print("-" * 50)

        try:
            notes = self._get_notes_list()

            # Sort notes
            if args.sort == 'modified':
                notes.sort(key=lambda x: x['modified'], reverse=True)
            elif args.sort == 'size':
                notes.sort(key=lambda x: x['size'], reverse=True)
            else:  # name
                notes.sort(key=lambda x: x['name'])

            if args.format == 'json':
                self._output_json(notes)
            else:
                self._output_notes_text(notes)

        except Exception as e:
            print(f"❌ List error: {e}")
            import traceback
            traceback.print_exc()

    def cmd_rebuild_index(self, args):
        """Handle rebuild-index command"""
        print("🔄 Rebuilding Search Index")
        print("-" * 50)

        # Set environment variables if provided
        if args.model:
            os.environ['MCP_MODEL_NAME'] = args.model
        if args.chunk_size:
            os.environ['MCP_CHUNK_SIZE'] = str(args.chunk_size)
        if args.overlap:
            os.environ['MCP_OVERLAP'] = str(args.overlap)

        # Ensure notes directory exists
        notes_path = Path(args.notes_root)
        if not notes_path.exists():
            print(f"❌ Notes directory not found: {args.notes_root}")
            print("💡 Creating notes directory...")
            notes_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {args.notes_root}")
            print("💡 Add some .md files to the directory and try again.")
            return

        try:
            # Import and run build_index
            from build_index import main as build_index_main
            build_index_main()
            print("✅ Index rebuilt successfully!")

        except Exception as e:
            print(f"❌ Rebuild error: {e}")
            import traceback
            traceback.print_exc()

    def cmd_server(self, args):
        """Handle server command"""
        print("🚀 Starting MCP Server")
        print("-" * 50)
        print(f"🌐 Host: {args.host}")
        print(f"🔌 Port: {args.port}")
        print(f"📁 Notes: {self.config.notes_root}")
        print(f"📊 Index: {self.config.index_file}")
        print()

        try:
            # Try to import and start server
            try:
                from notes_mcp_server import NotesMCPServer
                from admin_http import create_admin_app

                server = NotesMCPServer(
                    index_file=self.config.index_file,
                    meta_file=self.config.meta_file,
                    notes_root=self.config.notes_root
                )

                if not args.no_admin:
                    # Start admin server in background
                    import threading
                    admin_app = create_admin_app(server)
                    admin_thread = threading.Thread(
                        target=lambda: admin_app.run(host=args.host, port=args.port, debug=False)
                    )
                    admin_thread.daemon = True
                    admin_thread.start()
                    print(f"🌐 Admin interface: http://{args.host}:{args.port}")

                # Start MCP server
                print("💡 MCP server is running and ready for Copilot integration!")
                print("💡 Press Ctrl+C to stop")
                server.run()

            except ImportError:
                try:
                    from notes_mcp_server_simple import NotesMCPServer, create_admin_app
                    server = NotesMCPServer()
                    print("⚠️  Using simple MCP server implementation")

                    if not args.no_admin:
                        import threading
                        admin_app = create_admin_app(server)
                        admin_thread = threading.Thread(
                            target=lambda: admin_app.run(host=args.host, port=args.port, debug=False)
                        )
                        admin_thread.daemon = True
                        admin_thread.start()
                        print(f"🌐 Admin interface: http://{args.host}:{args.port}")

                    print("💡 Simple MCP server is running!")
                    print("💡 Press Ctrl+C to stop")
                    server.run(host=args.host, port=args.port)

                except ImportError:
                    print("❌ MCP server components not available")
                    print("💡 Try running: python notes_mcp_server_simple.py directly")
                    return

        except Exception as e:
            print(f"❌ Server error: {e}")
            import traceback
            traceback.print_exc()

    # Helper methods
    def _output_text(self, results):
        """Output results in text format"""
        for i, result in enumerate(results, 1):
            print(f"{i}. 📄 {result['file']}")
            print(f"   📊 Score: {result['score']:.3f}")
            print(f"   💬 {result['text'][:200]}...")
            print()

    def _output_table(self, results):
        """Output results in table format"""
        try:
            from tabulate import tabulate

            table_data = []
            for result in results:
                table_data.append([
                    result['file'],
                    ".3f",
                    result['text'][:100] + "..."
                ])

            headers = ["File", "Score", "Preview"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        except ImportError:
            print("⚠️  tabulate not installed, falling back to text format")
            self._output_text(results)

    def _output_json(self, data):
        """Output data in JSON format"""
        print(json.dumps(data, indent=2, ensure_ascii=False))

    def _output_stats_text(self, stats):
        """Output statistics in text format"""
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"📊 {key.replace('_', ' ').title()}:")
                for sub_key, sub_value in value.items():
                    print(f"   • {sub_key}: {sub_value}")
            else:
                print(f"📊 {key}: {value}")
            print()

    def _output_notes_text(self, notes):
        """Output notes list in text format"""
        for note in notes:
            print(f"📄 {note['name']}")
            print(f"   📅 Modified: {note['modified'].strftime('%Y-%m-%d %H:%M')}")
            print(f"   📏 Size: {note['size']} bytes")
            print()

    def _export_results(self, results, filename, format_type):
        """Export search results to file"""
        output_file = Path(filename)

        if format_type == 'json':
            self._export_json(results, output_file)
        elif format_type == 'text':
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Search Results\n{'='*50}\n\n")
                for i, result in enumerate(results, 1):
                    f.write(f"{i}. {result['file']}\n")
                    f.write(f"   Score: {result['score']:.3f}\n")
                    f.write(f"   {result['text']}\n\n")

        print(f"📤 Exported {len(results)} results to {output_file}")

    def _export_json(self, data, output_file):
        """Export data as JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _export_pdf(self, results, query, output_file):
        """Export results as PDF"""
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

            doc = SimpleDocTemplate(str(output_file), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title = Paragraph(f"Search Results for: {query}", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))

            # Results
            for i, result in enumerate(results, 1):
                result_text = f"<b>{i}. {result['file']}</b><br/>"
                result_text += f"Score: {result['score']:.3f}<br/>"
                result_text += f"{result['text'][:500]}..."

                para = Paragraph(result_text, styles['Normal'])
                story.append(para)
                story.append(Spacer(1, 12))

            doc.build(story)

        except ImportError:
            print("⚠️  PDF export requires reportlab: pip install reportlab")
            # Fallback to text export
            self._export_results(results, str(output_file).replace('.pdf', '.txt'), 'text')

    def _export_html(self, results, query, output_file):
        """Export results as HTML"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Search Results for: {query}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .result {{ margin-bottom: 20px; border-bottom: 1px solid #ccc; padding-bottom: 10px; }}
                .score {{ color: #666; font-size: 0.9em; }}
                .file {{ font-weight: bold; color: #2c5aa0; }}
            </style>
        </head>
        <body>
            <h1>Search Results for: {query}</h1>
            <p>Found {len(results)} results</p>
        """

        for i, result in enumerate(results, 1):
            html_content += f"""
            <div class="result">
                <h3 class="file">{i}. {result['file']}</h3>
                <div class="score">Score: {result['score']:.3f}</div>
                <p>{result['text']}</p>
            </div>
            """

        html_content += "</body></html>"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _export_markdown(self, results, query, output_file):
        """Export results as Markdown"""
        md_content = f"# Search Results for: {query}\n\n"
        md_content += f"**Found {len(results)} results**\n\n"

        for i, result in enumerate(results, 1):
            md_content += f"## {i}. {result['file']}\n\n"
            md_content += f"**Score:** {result['score']:.3f}\n\n"
            md_content += f"{result['text']}\n\n"
            md_content += "---\n\n"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)

    def _generate_tags(self, file_path):
        """Generate automatic tags for a markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple keyword-based tagging
            tags = []
            content_lower = content.lower()

            tag_keywords = {
                'machine learning': ['machine learning', 'ml', 'artificial intelligence', 'ai'],
                'data science': ['data science', 'pandas', 'numpy', 'jupyter'],
                'web development': ['html', 'css', 'javascript', 'react', 'vue', 'angular'],
                'devops': ['docker', 'kubernetes', 'aws', 'ci/cd', 'jenkins'],
                'python': ['python', 'django', 'flask', 'fastapi'],
                'cloud': ['aws', 'azure', 'gcp', 'cloud computing'],
                'database': ['sql', 'mongodb', 'postgresql', 'redis'],
                'security': ['security', 'encryption', 'authentication', 'oauth']
            }

            for tag, keywords in tag_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    tags.append(tag)

            return tags[:5]  # Limit to 5 tags

        except Exception as e:
            print(f"⚠️  Tag generation error: {e}")
            return []

    def _add_tags_to_file(self, file_path, tags):
        """Add tags to markdown file frontmatter"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check if frontmatter exists
            if content.startswith('---'):
                # Find end of frontmatter
                end_idx = content.find('---', 3)
                if end_idx != -1:
                    frontmatter = content[3:end_idx]
                    body = content[end_idx + 3:]

                    # Add tags to frontmatter
                    if 'tags:' not in frontmatter:
                        frontmatter += f"tags: {tags}\n"

                    new_content = f"---\n{frontmatter}---{body}"
                else:
                    new_content = content
            else:
                # Add frontmatter
                new_content = f"---\ntags: {tags}\n---\n\n{content}"

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

        except Exception as e:
            print(f"⚠️  Error adding tags: {e}")

    def _rebuild_index(self):
        """Rebuild the search index"""
        try:
            build_index_main()
        except Exception as e:
            raise Exception(f"Index rebuild failed: {e}")

    def _get_system_stats(self, period):
        """Get system statistics"""
        # This is a simplified version - you could expand this
        # to read from actual log files or database

        notes_dir = self.config.notes_root
        total_files = len(list(notes_dir.glob("*.md")))

        # Calculate period start
        now = datetime.now()
        if period == 'day':
            start_date = now - timedelta(days=1)
        elif period == 'week':
            start_date = now - timedelta(weeks=1)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        else:
            start_date = datetime.min

        # Get file stats
        recent_files = []
        total_size = 0

        for file_path in notes_dir.glob("*.md"):
            stat = file_path.stat()
            modified = datetime.fromtimestamp(stat.st_mtime)
            size = stat.st_size
            total_size += size

            if modified >= start_date:
                recent_files.append({
                    'name': file_path.name,
                    'modified': modified,
                    'size': size
                })

        return {
            'total_files': total_files,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'recent_files_count': len(recent_files),
            'period': period,
            'index_file_exists': Path(self.config.index_file).exists(),
            'meta_file_exists': Path(self.config.meta_file).exists(),
            'configuration': {
                'model': self.config.model_name,
                'chunk_size': self.config.chunk_size,
                'overlap': self.config.overlap,
                'environment': self.config.environment
            }
        }

    def _get_notes_list(self):
        """Get list of all notes with metadata"""
        notes = []
        notes_dir = self.config.notes_root

        for file_path in notes_dir.glob("*.md"):
            stat = file_path.stat()
            notes.append({
                'name': file_path.name,
                'path': str(file_path),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'size': stat.st_size,
                'size_kb': round(stat.st_size / 1024, 1)
            })

        return notes

    def _simple_text_search(self, query: str, limit: int = 10):
        """Simple text-based search as fallback"""
        results = []
        notes_dir = Path("./notes")

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
        return results[:limit]


def main():
    """Main entry point"""
    try:
        cli = MCPCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()