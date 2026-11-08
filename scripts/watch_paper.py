#!/usr/bin/env python3
"""
Live Watcher & Automatic PDF Compiler for IEEE Paper
Monitors report/IEEE_Paper_Image_Enhancement_Selection.tex and automatically
recompiles to report/IEEE_Paper_Image_Enhancement_Selection.pdf on every edit.

Usage:
    python3 scripts/watch_paper.py
    # or:
    make watch_paper
"""

import os
import sys
import time
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEX_FILE = os.path.join(PROJECT_ROOT, "report", "IEEE_Paper_Image_Enhancement_Selection.tex")
PDF_FILE = os.path.join(PROJECT_ROOT, "report", "IEEE_Paper_Image_Enhancement_Selection.pdf")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")


def compile_pdf() -> bool:
    t0 = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] Detected change in LaTeX file. Compiling PDF...")
    try:
        res = subprocess.run(
            ["tectonic", "IEEE_Paper_Image_Enhancement_Selection.tex"],
            cwd=REPORT_DIR,
            capture_output=True,
            text=True,
            check=True
        )
        dt = time.time() - t0
        pdf_size_mb = os.path.getsize(PDF_FILE) / (1024 * 1024)
        print(f"[{time.strftime('%H:%M:%S')}] SUCCESS: PDF recompiled in {dt:.2f}s ({pdf_size_mb:.2f} MB)")
        print(f"[{time.strftime('%H:%M:%S')}] Saved to: {PDF_FILE}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR during compilation:")
        print(e.stderr or e.stdout)
        return False
    except FileNotFoundError:
        print("ERROR: 'tectonic' is not installed or not in PATH.")
        return False


def main():
    if not os.path.exists(TEX_FILE):
        print(f"Error: LaTeX file not found at {TEX_FILE}")
        sys.exit(1)

    print("=" * 70)
    print("Project Triathlon: LaTeX Live PDF Auto-Compiler")
    print(f"Watching: report/IEEE_Paper_Image_Enhancement_Selection.tex")
    print(f"Target:   report/IEEE_Paper_Image_Enhancement_Selection.pdf")
    print("Every time you save the LaTeX file, it automatically compiles into PDF.")
    print("Press Ctrl+C to stop watching.")
    print("=" * 70)

    last_mtime = os.path.getmtime(TEX_FILE)
    # Perform initial compilation if PDF doesn't exist
    if not os.path.exists(PDF_FILE):
        compile_pdf()

    try:
        while True:
            time.sleep(0.5)
            try:
                current_mtime = os.path.getmtime(TEX_FILE)
                if current_mtime != last_mtime:
                    last_mtime = current_mtime
                    # Debounce brief write bursts
                    time.sleep(0.2)
                    compile_pdf()
            except OSError:
                pass
    except KeyboardInterrupt:
        print("\nWatcher stopped.")


if __name__ == "__main__":
    main()
