"""
Verification test for Project Triathlon Interactive Web Application.
Starts web server, verifies HTTP 200 response and API readiness, then cleanly terminates.
"""

import subprocess
import time
import urllib.request
import sys


def main():
    print("Testing Interactive Web App (apps/web/app.py)...")
    port = 8599
    proc = subprocess.Popen(
        [sys.executable, "apps/web/app.py", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Wait up to 5 seconds for server to start
        url = f"http://127.0.0.1:{port}"
        connected = False
        for _ in range(10):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    status = response.status
                    content = response.read().decode("utf-8")
                    if status == 200 and "Project Triathlon" in content:
                        print(f"  [OK] Successfully connected to {url} (HTTP {status})")
                        print(f"  [OK] HTML payload received ({len(content)} characters)")
                        connected = True
                        break
            except Exception:
                continue

        if not connected:
            stdout, stderr = proc.communicate(timeout=2)
            print(f"Server stdout: {stdout}")
            print(f"Server stderr: {stderr}")
            raise RuntimeError("Failed to connect to web app within timeout.")

        print("Interactive Web App verification test PASSED!\n")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
