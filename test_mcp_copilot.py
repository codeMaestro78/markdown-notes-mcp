#!/usr/bin/env python3
"""
Test script to verify MCP server functionality for Copilot integration.
"""

import json
import subprocess
import sys
from pathlib import Path

def test_mcp_server():
    """Test the MCP server with sample requests."""

    # Test requests
    test_requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "list_notes"
        },
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "search_notes",
            "params": {
                "query": "machine learning",
                "top_k": 3
            }
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "health_check"
        }
    ]

    # MCP server command
    cmd = [
        "C:/Users/Devarshi/PycharmProjects/markdown-notes-mcp/.venv/Scripts/python.exe",
        "notes_mcp_server.py",
        "--index", "notes_index.npz",
        "--meta", "notes_meta.json",
        "--notes_root", "./notes"
    ]

    # Environment variables
    env = {
        "PYTHONPATH": "C:/Users/Devarshi/PycharmProjects/markdown-notes-mcp",
        "MCP_ENVIRONMENT": "development",
        "MCP_MODEL_NAME": "all-MiniLM-L6-v2",
        "MCP_NOTES_ROOT": "./notes",
        "MCP_INDEX_FILE": "notes_index.npz",
        "MCP_META_FILE": "notes_meta.json",
        "MCP_CONFIG_DIR": "./config"
    }

    print("🧪 Testing MCP Server for Copilot Integration")
    print("=" * 50)

    try:
        # Start MCP server process
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**process.env, **env} if hasattr(subprocess, 'env') else env,
            cwd="C:/Users/Devarshi/PycharmProjects/markdown-notes-mcp"
        )

        results = []

        for i, request in enumerate(test_requests):
            print(f"\n📤 Sending request {i+1}: {request['method']}")

            # Send request
            request_json = json.dumps(request) + "\n"
            process.stdin.write(request_json)
            process.stdin.flush()

            # Read response
            response_line = process.stdout.readline().strip()
            if response_line:
                try:
                    response = json.loads(response_line)
                    print(f"📥 Response: {json.dumps(response, indent=2)[:200]}...")
                    results.append(True)
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON response: {e}")
                    results.append(False)
            else:
                print("❌ No response received")
                results.append(False)

        # Terminate server
        process.terminate()
        process.wait(timeout=5)

        # Summary
        success_count = sum(results)
        print("\n🎯 Test Results:")
        print(f"✅ Successful requests: {success_count}/{len(test_requests)}")

        if success_count == len(test_requests):
            print("🎉 MCP Server is ready for Copilot integration!")
            return True
        else:
            print("⚠️  Some tests failed. Check server logs for details.")
            return False

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_mcp_server()
    sys.exit(0 if success else 1)
