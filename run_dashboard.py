"""
run_dashboard.py — CLI Launcher for the AI-NIDS Unified Single-Page SOC Dashboard.

Usage:
    python3 run_dashboard.py [port]

Default Port:
    8501 (opens at http://127.0.0.1:8501)
"""

from __future__ import annotations

import sys
from dashboard.server import run_server

if __name__ == "__main__":
    port = 8501
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Warning: Invalid port '{sys.argv[1]}'; using default {port}")
    run_server(host="127.0.0.1", port=port)

