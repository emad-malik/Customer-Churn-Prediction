"""Thin wrapper — run directly with: python scripts/train.py"""
import sys
from pathlib import Path

# Make sure the src package is importable when running as a plain script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from churn.cli import train_main

if __name__ == "__main__":
    train_main()
