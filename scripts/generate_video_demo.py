#!/usr/bin/env python3
"""
Forwarder script to extensions.video.generate_video_demo.
Implementation lives in extensions/video/generate_video_demo.py.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from extensions.video.generate_video_demo import main

if __name__ == "__main__":
    main()
