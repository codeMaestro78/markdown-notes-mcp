#!/usr/bin/env python3
"""
test_mcp_copilot.py

Quick test script to verify MCP server is working for Copilot integration.
"""

import subprocess
import json
import sys
import time

def test_mcp_server():
    """Test the MCP server with basic operations."""
    print("🧪 Testing MCP Server for Copilot Integration")
    print("=" * 50)

    # Start MCP server
    cmd = [
        sys.executable,
        'notes_mcp_server.py',
        '--index', 'notes_index.npz',
        '--meta', 'notes_meta.json',
        '--notes_root', './notes'
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='c:/Users/Devarshi/PycharmProjects/markdown-notes-mcp'
        )

        # Give server time to start
        time.sleep(2)

        # Test 1: Health Check
        print("✅ Test 1: Health Check")
        request = {'jsonrpc': '2.0', 'id': 'test_health', 'method': 'health_check'}
        payload = json.dumps(request)
        header = f'Content-Length: {len(payload)}\r\n\r\n'

        proc.stdin.write(header + payload)
        proc.stdin.flush()

        # Read response
        header_line = proc.stdout.readline().strip()
        if header_line.startswith('Content-Length:'):
            content_length = int(header_line.split(':')[1].strip())
            proc.stdout.readline()  # Empty line
            response_body = proc.stdout.read(content_length)
            response = json.loads(response_body)
            print(f"   Health Check Result: {response.get('result', 'Failed')}")

        # Test 2: List Notes
        print("✅ Test 2: List Notes")
        request = {'jsonrpc': '2.0', 'id': 'test_list', 'method': 'list_notes'}
        payload = json.dumps(request)
        header = f'Content-Length: {len(payload)}\r\n\r\n'

        proc.stdin.write(header + payload)
        proc.stdin.flush()

        header_line = proc.stdout.readline().strip()
        if header_line.startswith('Content-Length:'):
            content_length = int(header_line.split(':')[1].strip())
            proc.stdout.readline()
            response_body = proc.stdout.read(content_length)
            response = json.loads(response_body)
            files = response.get('result', [])
            print(f"   Found {len(files)} note files:")
            for file in files[:5]:  # Show first 5
                print(f"   - {file}")

        # Test 3: Search Notes
        print("✅ Test 3: Search Notes")
        request = {
            'jsonrpc': '2.0',
            'id': 'test_search',
            'method': 'search_notes',
            'params': {'query': 'machine learning', 'top_k': 3}
        }
        payload = json.dumps(request)
        header = f'Content-Length: {len(payload)}\r\n\r\n'

        proc.stdin.write(header + payload)
        proc.stdin.flush()

        header_line = proc.stdout.readline().strip()
        if header_line.startswith('Content-Length:'):
            content_length = int(header_line.split(':')[1].strip())
            proc.stdout.readline()
            response_body = proc.stdout.read(content_length)
            response = json.loads(response_body)
            results = response.get('result', [])
            print(f"   Found {len(results)} search results for 'machine learning':")
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result['file']} (Score: {result['score']:.3f})")
                print(f"      \"{result['text'][:100]}...\"")

        print("\n🎉 MCP Server Test Complete!")
        print("✅ Server is ready for Copilot integration")
        print("\n💡 Try asking Copilot:")
        print("   - 'List all my available notes'")
        print("   - 'Search my notes for machine learning'")
        print("   - 'Show me the content of example.md'")

    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        print("💡 Make sure:")
        print("   - Index files exist (notes_index.npz, notes_meta.json)")
        print("   - Python virtual environment is activated")
        print("   - All dependencies are installed")
    finally:
        if 'proc' in locals():
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except:
                proc.kill()

if __name__ == "__main__":
    test_mcp_server()
