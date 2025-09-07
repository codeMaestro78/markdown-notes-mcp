#!/usr/bin/env python3
"""
test_cli_simple.py - Simple test script for MCP CLI

A basic test to check if the CLI is working at all.
"""

import subprocess
import sys
import os
from pathlib import Path

def test_cli_simple():
    """Simple test of CLI functionality"""
    print("🧪 Simple MCP CLI Test")
    print("=" * 40)

    project_root = Path(__file__).parent
    cli_script = project_root / "mcp_cli.py"

    print(f"📁 Project root: {project_root}")
    print(f"📄 CLI script: {cli_script}")

    # Check if CLI script exists
    if not cli_script.exists():
        print("❌ mcp_cli.py not found!")
        return False

    print("✅ CLI script found")

    # Get Python executable
    python_exe = sys.executable
    print(f"🐍 Python: {python_exe}")

    # Test 1: Can we run Python?
    try:
        result = subprocess.run([python_exe, "--version"],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Python is working")
        else:
            print("❌ Python execution failed")
            return False
    except Exception as e:
        print(f"❌ Python test failed: {e}")
        return False

    # Test 2: Can we import the CLI module?
    try:
        # Add project to path
        sys.path.insert(0, str(project_root))

        # Try importing
        import mcp_cli
        print("✅ CLI module can be imported")
    except ImportError as e:
        print(f"❌ CLI import failed: {e}")
        print("   This suggests missing dependencies or path issues")
        return False
    except Exception as e:
        print(f"⚠️  CLI import warning: {e}")

    # Test 3: Can we run the CLI help?
    try:
        cmd = [python_exe, str(cli_script), "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

        if result.returncode == 0:
            print("✅ CLI help command works")
            print(f"   Output length: {len(result.stdout)} characters")
            return True
        else:
            print("❌ CLI help command failed")
            print(f"   Return code: {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()[:200]}...")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ CLI help command timed out")
        return False
    except Exception as e:
        print(f"💥 CLI test exception: {e}")
        return False

def main():
    """Main function"""
    success = test_cli_simple()

    if success:
        print("\n🎉 CLI is working! Try these commands:")
        print("   python mcp_cli.py --help")
        print("   python mcp_cli.py list-notes")
        print("   python mcp_cli.py stats")
        print("   python mcp_cli.py search 'machine learning' --limit 3")
    else:
        print("\n❌ CLI has issues. Try:")
        print("1. pip install -r requirements.txt")
        print("2. python test_cli_fixed.py (for detailed diagnostics)")
        print("3. Check that you're in the project directory")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())