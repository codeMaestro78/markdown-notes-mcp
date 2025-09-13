#!/usr/bin/env python3
"""
Direct runner for Smart Collections Manager
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from smart_collections.manager import main

if __name__ == "__main__":
    main()
