"""Pytest discovery: add project root to sys.path so `from edge.* import ...` works."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
