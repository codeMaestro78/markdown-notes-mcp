#!/usr/bin/env python3
"""
setup_cli.py - Setup script for MCP CLI

This script helps install the MCP CLI tool globally and sets up the necessary environment.
"""

import os
import sys
import shutil
from pathlib import Path

def setup_cli():
    """Setup the MCP CLI for global use"""

    print("🚀 Setting up MCP CLI")
    print("=" * 50)

    project_root = Path(__file__).parent
    cli_script = project_root / "mcp_cli.py"

    if not cli_script.exists():
        print("❌ mcp_cli.py not found in current directory")
        return False

    # Determine Python executable
    python_exe = sys.executable
    print(f"📍 Python executable: {python_exe}")

    # Create batch file content
    batch_content = f'@echo off\nREM MCP CLI Launcher\npython "{cli_script}" %*\n'

    # Try to install in common locations
    install_locations = [
        Path(os.environ.get('USERPROFILE', '')) / 'bin',
        Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'mcp',
        Path.home() / 'bin',
        Path.home() / '.local' / 'bin',
    ]

    # Also check PATH directories
    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    for path_dir in path_dirs:
        if path_dir and Path(path_dir).exists():
            install_locations.append(Path(path_dir))

    installed = False
    for location in install_locations:
        try:
            location.mkdir(parents=True, exist_ok=True)
            batch_file = location / 'mcp.bat'

            with open(batch_file, 'w') as f:
                f.write(batch_content)

            print(f"✅ Installed MCP CLI to: {batch_file}")

            # Test the installation
            test_result = os.system(f'"{batch_file}" --help >nul 2>&1')
            if test_result == 0:
                print("✅ CLI installation verified!")
                installed = True
                break
            else:
                # Clean up failed installation
                batch_file.unlink(missing_ok=True)

        except (OSError, PermissionError) as e:
            print(f"⚠️  Could not install to {location}: {e}")
            continue

    if not installed:
        print("❌ Could not install CLI to any standard location")
        print("\n📋 Manual Installation Checking:-  ")
        print("1. Copy the following content to a file named 'mcp.bat'")
        print("2. Place it in a directory in your PATH")
        print("3. Or run: python mcp_cli.py [command]")
        print()
        print("mcp.bat content:")
        print("-" * 20)
        print(batch_content)
        return False

    print("\n🎉 MCP CLI Setup Complete!")
    print("\n📖 Usage Examples:")
    print("  mcp search 'machine learning' --format json --limit 5")
    print("  mcp add-note ./new_note.md --auto-tag")
    print("  mcp export-search 'PCA' --format pdf")
    print("  mcp stats --period week")
    print("  mcp list-notes --sort modified")
    print("  mcp rebuild-index --model all-mpnet-base-v2")
    print("  mcp server --host 127.0.0.1 --port 8181")

    print("\n💡 Test the installation:")
    print("  mcp --help")

    return True

def uninstall_cli():
    """Uninstall the MCP CLI"""

    print("🗑️  Uninstalling MCP CLI")
    print("=" * 50)

    # Find and remove CLI installations
    removed = []

    # Check common locations
    locations = [
        Path(os.environ.get('USERPROFILE', '')) / 'bin' / 'mcp.bat',
        Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'mcp' / 'mcp.bat',
        Path.home() / 'bin' / 'mcp.bat',
        Path.home() / '.local' / 'bin' / 'mcp.bat',
    ]

    # Check PATH directories
    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    for path_dir in path_dirs:
        if path_dir:
            locations.append(Path(path_dir) / 'mcp.bat')

    for location in locations:
        if location.exists():
            try:
                location.unlink()
                removed.append(str(location))
                print(f"✅ Removed: {location}")
            except OSError as e:
                print(f"⚠️  Could not remove {location}: {e}")

    if removed:
        print(f"\n✅ Successfully removed {len(removed)} CLI installation(s)")
    else:
        print("\n⚠️  No CLI installations found")

def main():
    """Main setup function"""

    if len(sys.argv) > 1 and sys.argv[1] == '--uninstall':
        uninstall_cli()
        return

    print("MCP CLI Setup Tool")
    print("==================")
    print()
    print("This tool will install the MCP CLI for global use.")
    print("You'll be able to run 'mcp' commands from anywhere.")
    print()

    if len(sys.argv) > 1 and sys.argv[1] == '--yes':
        confirm = True
    else:
        response = input("Continue with installation? (y/N): ").strip().lower()
        confirm = response in ['y', 'yes']

    if confirm:
        success = setup_cli()
        if success:
            print("\n🎊 Installation successful!")
        else:
            print("\n❌ Installation failed!")
            sys.exit(1)
    else:
        print("Installation cancelled.")

if __name__ == "__main__":
    main()
