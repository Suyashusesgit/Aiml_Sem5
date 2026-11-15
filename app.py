#!/usr/bin/env python3
"""
Root entry point / launcher for Project Triathlon Web Application.
Implementation lives in apps/web/app.py.
"""

import os
import sys

# Add project root and apps/web to path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "apps", "web"))
sys.path.insert(0, REPO_ROOT)

from apps.web.app import main

if __name__ == "__main__":
    main()
