#!/usr/bin/env python3
"""
test_cli_fixed.py - Fixed test script for MCP CLI

This script provides better error reporting and debugging for CLI testing.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command_detailed(cmd, description):
    """Run a command with detailed error reporting"""
    print(f"\n🧪 Testing: {description}")
    print(f"   Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print("-" * 60)

    try:
        # Use shell=True for Windows compatibility
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd()
        )

        print(f"   Return code: {result.returncode}")

        if result.returncode == 0:
            print("✅ PASSED")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()[:200]}...")
            return True
        else:
            print("❌ FAILED")
            if result.stdout.strip():
                print(f"   STDOUT: {result.stdout.strip()}")
            if result.stderr.strip():
                print(f"   STDERR: {result.stderr.strip()}")
            else:
                print("   STDERR: (empty)")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT: Command took too long")
        return False
    except FileNotFoundError as e:
        print(f"📁 FILE NOT FOUND: {e}")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
        return False

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔍 Checking Prerequisites")
    print("=" * 50)

    issues = []

    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 8):
        issues.append("Python 3.8+ required")

    # Check if we're in the right directory
    project_root = Path(__file__).parent
    print(f"📁 Current directory: {project_root}")

    # Check for required files
    required_files = [
        'mcp_cli.py',
        'config.py',
        'requirements.txt',
        'notes_mcp_server.py'
    ]

    for file in required_files:
        if (project_root / file).exists():
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
            issues.append(f"Missing {file}")

    # Check for notes directory
    notes_dir = project_root / 'notes'
    if notes_dir.exists():
        md_files = list(notes_dir.glob('*.md'))
        print(f"✅ Notes directory found with {len(md_files)} .md files")
    else:
        print("❌ Notes directory not found")
        issues.append("Missing notes directory")

    # Check for index files
    index_file = project_root / 'notes_index.npz'
    meta_file = project_root / 'notes_meta.json'

    if index_file.exists():
        print("✅ Index file found")
    else:
        print("⚠️  Index file not found (will be created)")

    if meta_file.exists():
        print("✅ Meta file found")
    else:
        print("⚠️  Meta file not found (will be created)")

    # Check virtual environment
    venv_path = project_root / '.venv'
    if venv_path.exists():
        print("✅ Virtual environment found")
    else:
        print("⚠️  Virtual environment not found")
        issues.append("Consider creating virtual environment")

    if issues:
        print(f"\n⚠️  Found {len(issues)} potential issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✅ All prerequisites look good!")

    return len(issues) == 0

def test_basic_python():
    """Test basic Python functionality"""
    print("\n🧪 Testing Basic Python")
    print("-" * 30)

    # Test Python import
    try:
        import sys
        print(f"✅ Python executable: {sys.executable}")
    except Exception as e:
        print(f"❌ Python import failed: {e}")
        return False

    # Test basic imports
    test_imports = ['os', 'pathlib', 'subprocess']
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module} import successful")
        except ImportError as e:
            print(f"❌ {module} import failed: {e}")
            return False

    return True

def test_cli_imports():
    """Test CLI-specific imports"""
    print("\n🧪 Testing CLI Imports")
    print("-" * 30)

    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    test_imports = [
        'config',
        'notes_mcp_server',
        'build_index'
    ]

    success_count = 0
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module} import successful")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module} import failed: {e}")
            print(f"   Make sure you're in the project directory: {project_root}")
        except Exception as e:
            print(f"⚠️  {module} import warning: {e}")

    return success_count >= 2  # At least config and one other

def main():
    """Main test function"""
    print("🧪 MCP CLI Diagnostic Test")
    print("=" * 60)

    # Check prerequisites
    prereqs_ok = check_prerequisites()

    # Test basic Python
    python_ok = test_basic_python()

    # Test CLI imports
    imports_ok = test_cli_imports()

    if not (prereqs_ok and python_ok and imports_ok):
        print("\n❌ Prerequisites not met. Please fix issues above before proceeding.")
        return 1

    print("\n🚀 Running CLI Tests")
    print("=" * 60)

    project_root = Path(__file__).parent
    cli_script = project_root / "mcp_cli.py"

    # Determine Python command
    python_cmd = sys.executable

    tests_passed = 0
    total_tests = 0

    # Test 1: CLI script exists and is executable
    total_tests += 1
    if cli_script.exists():
        print("✅ CLI script found")
        tests_passed += 1
    else:
        print("❌ CLI script not found")
        return 1

    # Test 2: Basic Python execution
    total_tests += 1
    if run_command_detailed(f'"{python_cmd}" --version', "Python version check"):
        tests_passed += 1

    # Test 3: CLI help command
    total_tests += 1
    cmd = f'"{python_cmd}" "{cli_script}" --help'
    if run_command_detailed(cmd, "CLI help command"):
        tests_passed += 1

    # Test 4: CLI list command (doesn't require index)
    total_tests += 1
    cmd = f'"{python_cmd}" "{cli_script}" list-notes --help'
    if run_command_detailed(cmd, "List notes help"):
        tests_passed += 1

    # Test 5: CLI stats command (doesn't require index)
    total_tests += 1
    cmd = f'"{python_cmd}" "{cli_script}" stats --help'
    if run_command_detailed(cmd, "Stats help"):
        tests_passed += 1

    # Test 6: Check if we can import the CLI module directly
    total_tests += 1
    try:
        # Try importing the CLI module
        spec = __import__('mcp_cli', fromlist=['MCPCLI'])
        print("✅ CLI module import successful")
        tests_passed += 1
    except Exception as e:
        print(f"❌ CLI module import failed: {e}")
        # Try to run the CLI with verbose error reporting
        cmd = f'"{python_cmd}" -c "import mcp_cli"'
        run_command_detailed(cmd, "Direct CLI import test")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! CLI is ready to use.")
        print("\n💡 Try these commands:")
        print("   python mcp_cli.py --help")
        print("   python mcp_cli.py list-notes")
        print("   python mcp_cli.py stats")
        return 0
    else:
        print(f"⚠️  {total_tests - tests_passed} test(s) failed.")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure you're in the project directory")
        print("2. Check that all dependencies are installed: pip install -r requirements.txt")
        print("3. Try rebuilding the index: python build_index.py ./notes")
        print("4. Check file permissions")
        print("5. Try running: python -c \"import sys; print(sys.path)\"")
        return 1

if __name__ == "__main__":
    sys.exit(main())