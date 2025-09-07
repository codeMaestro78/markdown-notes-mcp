#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def run_git_status():
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, cwd=Path(__file__).parent)
        if result.returncode == 0:
            if result.stdout.strip():
                print("📝 Uncommitted changes:")
                print(result.stdout)
            else:
                print("✅ Working directory is clean")
        else:
            print("❌ Git status failed")
            print(result.stderr)
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    run_git_status()
