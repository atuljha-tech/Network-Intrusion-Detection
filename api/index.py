"""
api/index.py — Serverless function entrypoint for Vercel deployment.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path so src and dashboard packages resolve cleanly
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dashboard.server import app

# Vercel WSGI / ASGI interface
app = app

