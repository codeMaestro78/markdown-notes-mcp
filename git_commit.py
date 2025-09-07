#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def commit_changes(message=None):
    if not message:
        print("❌ Commit message required")
        print("Usage: python git_commit.py 'Your commit message'")
        sys.exit(1)

    try:
        # Add all changes
        subprocess.run(['git', 'add', '.'], cwd=Path(__file__).parent, check=True)

        # Commit with message
        subprocess.run(['git', 'commit', '-m', message], cwd=Path(__file__).parent, check=True)

        print(f"✅ Committed: {message}")

        # Show status
        result = subprocess.run(['git', 'status', '--short'], capture_output=True, text=True, cwd=Path(__file__).parent)
        print("\n📊 Current status:")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"❌ Git command failed: {e}")
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        commit_changes(sys.argv[1])
    else:
        commit_changes()
