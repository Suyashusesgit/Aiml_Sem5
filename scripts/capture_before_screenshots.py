#!/usr/bin/env python3
"""
Captures before screenshots for UI audit across resolutions and tabs.
"""
import os
import subprocess
import time

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BASE_URL = "http://localhost:8501"
OUTPUT_DIR = os.path.abspath("docs/ui_audit/before")
PROFILE_DIR = "/tmp/chrome_audit_profile"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROFILE_DIR, exist_ok=True)

targets = [
    # First load
    ("first_load_1440x900.png", 1440, 900, f"{BASE_URL}/"),
    ("first_load_390x844.png", 390, 844, f"{BASE_URL}/"),
    ("first_load_0_5s.png", 1440, 900, f"{BASE_URL}/"),
    ("first_load_3s.png", 1440, 900, f"{BASE_URL}/"),
    ("first_load_10s.png", 1440, 900, f"{BASE_URL}/"),
]

for filename, w, h, url in targets:
    out_path = os.path.join(OUTPUT_DIR, filename)
    cmd = [
        CHROME_BIN,
        "--headless=new",
        "--disable-gpu",
        f"--user-data-dir={PROFILE_DIR}",
        f"--screenshot={out_path}",
        f"--window-size={w},{h}",
        "--hide-scrollbars",
        url
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        if os.path.exists(out_path):
            print(f"Captured: {filename} ({os.path.getsize(out_path)} bytes)")
        else:
            print(f"Failed to capture: {filename}")
    except Exception as e:
        print(f"Error capturing {filename}: {e}")
