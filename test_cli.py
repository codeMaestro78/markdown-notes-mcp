#!/usr/bin/env python3
"""
test_cli.py - Test script for MCP CLI functionality

This script tests the basic functionality of the MCP CLI to ensure everything is working correctly.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"🧪 Testing: {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print(f"✅ PASSED: {description}")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()[:100]}...")
            return True
        else:
            print(f"❌ FAILED: {description}")
            print(f"   Error: {result.stderr.strip()}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"💥 ERROR: {description} - {e}")
        return False

def test_cli():
    """Test the MCP CLI functionality"""

    print("🧪 MCP CLI Test Suite")
    print("=" * 50)

    project_root = Path(__file__).parent
    cli_script = project_root / "mcp_cli.py"

    if not cli_script.exists():
        print("❌ mcp_cli.py not found!")
        return False

    # Check if virtual environment exists
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = project_root / ".venv" / "bin" / "python"
        if not venv_python.exists():
            print("⚠️  Virtual environment not found, using system Python")

    python_cmd = str(venv_python) if venv_python.exists() else "python"

    tests_passed = 0
    total_tests = 0

    # Test 1: CLI help
    total_tests += 1
    cmd = f'{python_cmd} "{cli_script}" --help'
    if run_command(cmd, "CLI help command"):
        tests_passed += 1

    # Test 2: List notes command
    total_tests += 1
    cmd = f'{python_cmd} "{cli_script}" list-notes --format json'
    if run_command(cmd, "List notes command"):
        tests_passed += 1

    # Test 3: Stats command
    total_tests += 1
    cmd = f'{python_cmd} "{cli_script}" stats --format json'
    if run_command(cmd, "Stats command"):
        tests_passed += 1

    # Test 4: Search command (if index exists)
    index_file = project_root / "notes_index.npz"
    if index_file.exists():
        total_tests += 1
        cmd = f'{python_cmd} "{cli_script}" search "test" --limit 3 --format json'
        if run_command(cmd, "Search command"):
            tests_passed += 1
    else:
        print("⚠️  Skipping search test - index file not found")
        print("   Run: python build_index.py ./notes")

    # Test 5: Rebuild index command
    total_tests += 1
    cmd = f'{python_cmd} "{cli_script}" rebuild-index --help'
    if run_command(cmd, "Rebuild index help"):
        tests_passed += 1

    # Test 6: Export search help
    total_tests += 1
    cmd = f'{python_cmd} "{cli_script}" export-search --help'
    if run_command(cmd, "Export search help"):
        tests_passed += 1

    # Test 7: Add note help
    total_tests += 1
    cmd = f'{python_cmd} "{cli_script}" add-note --help'
    if run_command(cmd, "Add note help"):
        tests_passed += 1

    # Test 8: Server help
    total_tests += 1
    cmd = f'{python_cmd} "{cli_script}" server --help'
    if run_command(cmd, "Server help"):
        tests_passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! CLI is working correctly.")
        return True
    else:
        print(f"⚠️  {total_tests - tests_passed} test(s) failed. Check the output above.")
        return False

def main():
    """Main test function"""
    try:
        success = test_cli()

        if success:
            print("\n💡 Next Steps:")
            print("1. Try: python mcp_cli.py search 'machine learning' --limit 3")
            print("2. Try: python mcp_cli.py list-notes")
            print("3. Try: python mcp_cli.py stats")
            print("4. For global install: python setup_cli.py")
            print("\n📖 Full documentation: CLI_DOCUMENTATION.md")

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
        return 1
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())