"""
app.py — Root Production & Vercel Entrypoint for AI-NIDS SOC Dashboard.

Enables Vercel Zero-Configuration Flask Framework Preset detection.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dashboard.server import app

# Expose WSGI application instance for Vercel
app = app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=3000, debug=False)

