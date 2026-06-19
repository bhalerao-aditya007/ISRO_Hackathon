"""
HuggingFace Spaces entry point — imports and re-exports the FastAPI app.
HF Spaces looks for app.py with a variable named `app`.
"""
import os
import sys
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).parent))

# Set env vars before importing backend
os.environ.setdefault("PRISM_ROOT", str(Path(__file__).parent))
os.environ.setdefault("PRISM_OUTPUT_DIR", "/data/prism_outputs")
os.environ.setdefault("PRISM_CONFIG", str(Path(__file__).parent / "config" / "pipeline_config.json"))
os.environ.setdefault("DATA_CACHE_DIR", "/data/prism_data")

from backend import app  # noqa: F401 — HF Spaces picks up `app`
