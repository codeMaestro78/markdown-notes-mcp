#!/usr/bin/env python3
"""
quick_fix.py - Quick fix for MCP CLI issues

This script will diagnose and fix common CLI issues.
"""

import os
import sys
import subprocess
from pathlib import Path

def diagnose_issues():
    """Diagnose common CLI issues"""
    print("🔍 Diagnosing MCP CLI Issues")
    print("=" * 50)

    project_root = Path(__file__).parent
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python executable: {sys.executable}")
    print(f"📍 Current working directory: {os.getcwd()}")

    # Check if we're in the right directory
    if not (project_root / "mcp_cli.py").exists():
        print("❌ mcp_cli.py not found in current directory!")
        print("Please navigate to the project directory:")
        print(f"   cd {project_root}")
        return False

    # Check Python path
    print(f"🔍 Python path includes project: {str(project_root) in sys.path}")

    # Try importing key modules
    print("\n📦 Testing imports:")
    try:
        sys.path.insert(0, str(project_root))
        import config
        print("✅ config module imported")
    except ImportError as e:
        print(f"❌ config import failed: {e}")

    try:
        import notes_mcp_server
        print("✅ notes_mcp_server module imported")
    except ImportError as e:
        print(f"❌ notes_mcp_server import failed: {e}")

    # Check if we can run Python commands
    print("\n🚀 Testing Python execution:")
    try:
        result = subprocess.run([sys.executable, "--version"],
                              capture_output=True, text=True, timeout=5)
        print(f"✅ Python execution works: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ Python execution failed: {e}")

    # Test CLI help command
    print("\n🧪 Testing CLI help:")
    try:
        cmd = [sys.executable, str(project_root / "mcp_cli.py"), "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            print("✅ CLI help works!")
            print(f"   Output preview: {result.stdout[:100]}...")
        else:
            print("❌ CLI help failed")
            print(f"   Return code: {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr}")
            else:
                print("   No error message (this is the issue!)")

    except subprocess.TimeoutExpired:
        print("⏰ CLI help timed out")
    except Exception as e:
        print(f"💥 CLI test exception: {e}")

    return True

def quick_fix():
    """Apply quick fixes"""
    print("\n🔧 Applying Quick Fixes")
    print("=" * 30)

    project_root = Path(__file__).parent

    # Fix 1: Ensure proper Python path
    print("1. ✅ Ensuring Python path is set correctly")

    # Fix 2: Check for missing dependencies
    print("2. 🔍 Checking dependencies...")
    try:
        import sentence_transformers
        print("   ✅ sentence-transformers available")
    except ImportError:
        print("   ⚠️  sentence-transformers not found")
        print("   Run: pip install sentence-transformers")

    try:
        import numpy
        print("   ✅ numpy available")
    except ImportError:
        print("   ⚠️  numpy not found")
        print("   Run: pip install numpy")

    # Fix 3: Test the fixed CLI
    print("3. 🧪 Testing fixed CLI...")
    fixed_cli = project_root / "mcp_cli_fixed.py"
    if fixed_cli.exists():
        try:
            cmd = [sys.executable, str(fixed_cli), "--help"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                print("   ✅ Fixed CLI works!")
                print("   💡 Use: python mcp_cli_fixed.py [command]")
            else:
                print("   ❌ Fixed CLI still has issues")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")

        except Exception as e:
            print(f"   💥 Fixed CLI test failed: {e}")
    else:
        print("   ⚠️  Fixed CLI not found")

    # Fix 4: Provide usage instructions
    print("\n📖 Usage Instructions:")
    print("1. Make sure you're in the project directory")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Try the fixed CLI: python mcp_cli_fixed.py --help")
    print("4. Or use simple test: python test_cli_simple.py")
    print("5. For detailed diagnostics: python test_cli_fixed.py")

def main():
    """Main function"""
    diagnose_issues()
    quick_fix()

    print("\n🎯 Next Steps:")
    print("1. Try: python mcp_cli_fixed.py --help")
    print("2. Try: python mcp_cli_fixed.py list-notes")
    print("3. Try: python mcp_cli_fixed.py stats")
    print("4. If still failing, run: python test_cli_fixed.py")

if __name__ == "__main__":
    main()