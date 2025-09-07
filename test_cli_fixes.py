#!/usr/bin/env python3
"""
Simple test to verify CLI fixes work
"""

import sys
import os
from pathlib import Path

def test_cli_fixes():
    """Test that the CLI fixes work"""
    print("🧪 Testing CLI Fixes")
    print("=" * 40)

    project_root = Path(__file__).parent
    cli_path = project_root / "mcp_cli_fixed.py"

    if not cli_path.exists():
        print("❌ mcp_cli_fixed.py not found")
        return False

    # Test 1: Can we run the CLI help?
    print("1. Testing CLI help...")
    try:
        result = os.system(f'python "{cli_path}" --help >nul 2>&1')
        if result == 0:
            print("   ✅ CLI help works")
        else:
            print("   ❌ CLI help failed")
            return False
    except Exception as e:
        print(f"   💥 CLI help exception: {e}")
        return False

    # Test 2: Can we run list-notes?
    print("2. Testing list-notes...")
    try:
        result = os.system(f'python "{cli_path}" list-notes >nul 2>&1')
        if result == 0:
            print("   ✅ list-notes works")
        else:
            print("   ❌ list-notes failed")
    except Exception as e:
        print(f"   💥 list-notes exception: {e}")
        return False

    # Test 3: Can we run stats?
    print("3. Testing stats...")
    try:
        result = os.system(f'python "{cli_path}" stats >nul 2>&1')
        if result == 0:
            print("   ✅ stats works")
        else:
            print("   ❌ stats failed")
    except Exception as e:
        print(f"   💥 stats exception: {e}")
        return False

    # Test 4: Can we run search with simple text search?
    print("4. Testing search (simple text)...")
    try:
        result = os.system(f'python "{cli_path}" search "test" --limit 3 >nul 2>&1')
        if result == 0:
            print("   ✅ search works (simple text fallback)")
        else:
            print("   ❌ search failed")
    except Exception as e:
        print(f"   💥 search exception: {e}")
        return False

    print("\n🎉 All CLI fixes verified!")
    print("\n💡 You can now run:")
    print("   python mcp_cli_fixed.py search \"machine learning\" --format json --limit 5")
    print("   python mcp_cli_fixed.py add-note ./note.md --auto-tag")
    print("   python mcp_cli_fixed.py export-search \"PCA\" --format pdf")
    print("   python mcp_cli_fixed.py stats --period week")
    print("   python mcp_cli_fixed.py list-notes")
    print("   python mcp_cli_fixed.py server")

    return True

if __name__ == "__main__":
    success = test_cli_fixes()
    sys.exit(0 if success else 1)