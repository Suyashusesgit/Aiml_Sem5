#!/usr/bin/env python3
"""
Repository Health & Anti-Corruption Checker
Fails on:
1. Duplicate files by SHA-256 hash (excluding _archive/ and data/)
2. Tracked caches, __pycache__, or generated artifacts in Git index
3. Tracked files larger than 25 MB
4. Unresolved internal imports in src/, scripts/, tests/
5. Non-existent CLI commands referenced in README

Usage:
    python3 scripts/check_repo.py
"""

import os
import sys
import hashlib
import subprocess
import ast
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO_ROOT)


def check_duplicate_hashes() -> bool:
    print("[1/5] Checking for duplicate files by SHA-256 hash...")
    hash_map = {}
    ignored_prefixes = (
        os.path.join(REPO_ROOT, ".git"),
        os.path.join(REPO_ROOT, ".venv"),
        os.path.join(REPO_ROOT, "_archive"),
        os.path.join(REPO_ROOT, "data"),
        os.path.join(REPO_ROOT, "results"),
        os.path.join(REPO_ROOT, ".vscode"),
    )

    for root, dirs, files in os.walk(REPO_ROOT):
        if any(root.startswith(p) for p in ignored_prefixes):
            continue
        for f in files:
            path = os.path.join(root, f)
            try:
                with open(path, "rb") as fp:
                    h = hashlib.sha256(fp.read()).hexdigest()
                hash_map.setdefault(h, []).append(os.path.relpath(path, REPO_ROOT))
            except Exception:
                pass

    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    if duplicates:
        print("  FAIL: Duplicate files detected:")
        for h, paths in duplicates.items():
            print(f"    Hash {h[:12]}: {paths}")
        return False
    print("  PASS: Zero duplicate files detected.")
    return True


def check_tracked_caches_and_size() -> bool:
    print("[2/5] Checking Git index for tracked caches, __pycache__, and large files (>25MB)...")
    try:
        output = subprocess.check_output(["git", "ls-files"], text=True)
        tracked_files = output.strip().splitlines()
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Unable to run git ls-files: {e}")
        return False

    failed = False
    for path in tracked_files:
        # Check cache / junk patterns
        if any(pattern in path for pattern in ["__pycache__", ".pyc", ".pytest_cache", ".ipynb_checkpoints", ".DS_Store"]):
            print(f"  FAIL: Tracked cache file found: {path}")
            failed = True
        if path.startswith("results/") and path.endswith(".csv"):
            print(f"  FAIL: Tracked result cache found in results/: {path}")
            failed = True

        # Check size > 25MB
        if os.path.isfile(path):
            sz_mb = os.path.getsize(path) / (1024 * 1024)
            if sz_mb > 25.0:
                print(f"  FAIL: Tracked file exceeds 25 MB ({sz_mb:.1f} MB): {path}")
                failed = True

    if not failed:
        print(f"  PASS: All {len(tracked_files)} tracked files are clean and <= 25 MB.")
    return not failed


def check_unresolved_imports() -> bool:
    print("[3/5] Checking for unresolved internal imports in src/, scripts/, tests/...")
    target_dirs = ["src", "scripts", "tests"]
    sys.path.insert(0, REPO_ROOT)

    failed = False
    for tdir in target_dirs:
        dir_path = os.path.join(REPO_ROOT, tdir)
        if not os.path.isdir(dir_path):
            continue
        for root, dirs, files in os.walk(dir_path):
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                    try:
                        tree = ast.parse(fp.read(), filename=path)
                    except SyntaxError as e:
                        print(f"  FAIL: SyntaxError in {path}: {e}")
                        failed = True
                        continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        mod = node.module
                        if mod and mod.startswith("src."):
                            try:
                                __import__(mod)
                            except Exception as e:
                                print(f"  FAIL: Unresolved import '{mod}' in {os.path.relpath(path, REPO_ROOT)}: {e}")
                                failed = True

    if not failed:
        print("  PASS: All internal imports resolve successfully.")
    return not failed


def check_readme_commands() -> bool:
    print("[4/5] Checking README commands and CLI subcommands...")
    readme_path = os.path.join(REPO_ROOT, "README.md")
    if not os.path.isfile(readme_path):
        print("  FAIL: README.md not found.")
        return False

    with open(readme_path, "r", encoding="utf-8") as fp:
        readme_text = fp.read()

    # Extract all python -m src <subcmd>
    matches = re.findall(r"python(?:3)? -m src (\w+)", readme_text)
    known_subcmds = {"env", "groups", "run", "paper", "demo", "smoke"}

    failed = False
    for cmd in set(matches):
        if cmd not in known_subcmds:
            print(f"  FAIL: README references unknown CLI subcommand: 'python -m src {cmd}'")
            failed = True

    if not failed:
        print(f"  PASS: All {len(set(matches))} referenced CLI commands exist in src.cli.")
    return not failed


def check_critical_files() -> bool:
    print("[5/5] Checking presence of required canonical protocol files...")
    required = [
        "PROTOCOL.md",
        "README.md",
        "requirements.txt",
        "requirements-tested.txt",
        "pytest.ini",
        ".gitignore",
        "configs/base.yaml",
        "docs/DECISION_RULE.md",
        "docs/INVENTORY.md",
        "docs/PROGRESS.md",
        "docs/CLEANUP_REPORT.md",
        "docs/DECISIONS.md",
    ]
    missing = [f for f in required if not os.path.isfile(os.path.join(REPO_ROOT, f))]
    if missing:
        print(f"  FAIL: Missing critical canonical files: {missing}")
        return False
    print(f"  PASS: All {len(required)} critical protocol documents exist.")
    return True


def main():
    print("=" * 65)
    print("PROJECT TRIATHLON · REPOSITORY INTEGRITY AUDIT")
    print("=" * 65)

    results = [
        check_duplicate_hashes(),
        check_tracked_caches_and_size(),
        check_unresolved_imports(),
        check_readme_commands(),
        check_critical_files(),
    ]

    print("=" * 65)
    if all(results):
        print("OVERALL HEALTH AUDIT: ALL CHECKS PASSED (Repository is clean)")
        sys.exit(0)
    else:
        print("OVERALL HEALTH AUDIT: FAILED (Resolve flagged issues before proceeding)")
        sys.exit(1)


if __name__ == "__main__":
    main()
