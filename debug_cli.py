#!/usr/bin/env python3
"""
debug_cli.py - Debug the CLI issues step by step
"""

import os
import sys
import subprocess
from pathlib import Path

def debug_step_by_step():
    """Debug CLI issues step by step"""
    print("🐛 CLI Debug Session")
    print("=" * 50)

    project_root = Path(__file__).parent
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python: {sys.executable}")
    print(f"📍 CWD: {os.getcwd()}")

    # Step 1: Check file existence
    print("\n1️⃣  Checking file existence:")
    files_to_check = [
        'mcp_cli.py',
        'config.py',
        'notes_mcp_server.py',
        'build_index.py',
        'requirements.txt'
    ]

    for file in files_to_check:
        path = project_root / file
        if path.exists():
            print(f"   ✅ {file} exists")
        else:
            print(f"   ❌ {file} missing")

    # Step 2: Check Python path
    print("\n2️⃣  Checking Python path:")
    print(f"   Project in path: {str(project_root) in sys.path}")

    # Step 3: Test basic Python execution
    print("\n3️⃣  Testing basic Python:")
    try:
        result = subprocess.run([sys.executable, "-c", "print('Hello from Python')"],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("   ✅ Basic Python works")
        else:
            print(f"   ❌ Basic Python failed: {result.stderr}")
    except Exception as e:
        print(f"   💥 Basic Python exception: {e}")

    # Step 4: Test imports individually
    print("\n4️⃣  Testing imports:")
    sys.path.insert(0, str(project_root))

    test_imports = [
        'os',
        'sys',
        'pathlib',
        'json',
        'config',
        'notes_mcp_server',
        'build_index'
    ]

    for module in test_imports:
        try:
            __import__(module)
            print(f"   ✅ {module} imported")
        except ImportError as e:
            print(f"   ❌ {module} import failed: {e}")
        except Exception as e:
            print(f"   ⚠️  {module} import warning: {e}")

    # Step 5: Test CLI script execution directly
    print("\n5️⃣  Testing CLI script execution:")
    cli_path = project_root / "mcp_cli.py"

    if cli_path.exists():
        print("   CLI script found, testing execution...")

        # Test 1: Just run the script without arguments
        try:
            cmd = [sys.executable, str(cli_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            print(f"   Return code: {result.returncode}")
            if result.stdout:
                print(f"   STDOUT: {result.stdout[:200]}...")
            if result.stderr:
                print(f"   STDERR: {result.stderr[:200]}...")
            else:
                print("   STDERR: (empty)")

        except subprocess.TimeoutExpired:
            print("   ⏰ Script timed out")
        except Exception as e:
            print(f"   💥 Script execution failed: {e}")

        # Test 2: Try with --help
        print("   Testing --help argument...")
        try:
            cmd = [sys.executable, str(cli_path), "--help"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            print(f"   Help return code: {result.returncode}")
            if result.returncode == 0:
                print("   ✅ Help command works!")
            else:
                print("   ❌ Help command failed")
                if result.stderr:
                    print(f"   Help error: {result.stderr[:300]}...")
                else:
                    print("   Help error: (empty stderr - this is the issue!)")

        except Exception as e:
            print(f"   💥 Help test failed: {e}")

    else:
        print("   ❌ CLI script not found")

    # Step 6: Check environment
    print("\n6️⃣  Environment check:")
    print(f"   Python version: {sys.version}")
    print(f"   Platform: {sys.platform}")

    # Check for common issues
    print("\n🔍 Common issue detection:")

    # Check if we're in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"   In virtual environment: {in_venv}")

    # Check for missing dependencies
    try:
        import sentence_transformers
        print("   ✅ sentence-transformers available")
    except ImportError:
        print("   ❌ sentence-transformers missing - run: pip install sentence-transformers")

    try:
        import numpy
        print("   ✅ numpy available")
    except ImportError:
        print("   ❌ numpy missing - run: pip install numpy")

def main():
    """Main debug function"""
    debug_step_by_step()

    print("\n🎯 Debug Summary:")
    print("If you're seeing empty error messages, it usually means:")
    print("1. Import errors in the CLI script")
    print("2. Missing dependencies")
    print("3. Path issues")
    print("4. Virtual environment problems")

    print("\n💡 Try these solutions:")
    print("1. Run: python quick_fix.py")
    print("2. Run: python test_cli_fixed.py")
    print("3. Install deps: pip install -r requirements.txt")
    print("4. Use fixed CLI: python mcp_cli_fixed.py --help")

if __name__ == "__main__":
    main()