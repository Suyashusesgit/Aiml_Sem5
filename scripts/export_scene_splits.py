#!/usr/bin/env python3
"""
Scene Splits & Filename Parser Script
Module Lead: Ananya Kasaudhan (Data & Splits Lead)

Parses filenames from the degraded image directory (`low/`), assigns physical
`scene_id` groupings such that variants sharing root scene identifiers (e.g., all `1_*` files)
share the exact same group, and exports clean CSVs with columns:
    filename, scene_id

Usage:
    python3 scripts/export_scene_splits.py
    python3 scripts/export_scene_splits.py --split our485 --output data/scene_splits.csv
    python3 scripts/export_scene_splits.py --low-dir data/lol_dataset/our485/low --output data/scene_splits.csv
"""

import os
import sys
import argparse
import csv
import pandas as pd
from typing import List, Dict, Tuple, Optional

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dataset import parse_scene_id, resolve_dataset_dir


def parse_low_directory(low_dir: str) -> List[Dict[str, str]]:
    """
    Parses all image filenames from a low/ directory and groups by scene prefix.

    Args:
        low_dir: Path to directory containing low-light images.

    Returns:
        List of dicts with 'filename' and 'scene_id'.
    """
    if not os.path.exists(low_dir):
        raise FileNotFoundError(f"Directory not found: {low_dir}")

    valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    all_files = sorted([
        f for f in os.listdir(low_dir)
        if f.lower().endswith(valid_extensions) and not f.startswith(".")
    ])

    records = []
    for f in all_files:
        sid = parse_scene_id(f)
        records.append({
            "filename": f,
            "scene_id": sid
        })
    return records


def export_csv(records: List[Dict[str, str]], output_path: str) -> str:
    """Exports records to CSV with exact columns: filename, scene_id."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "scene_id"])
        for r in records:
            writer.writerow([r["filename"], r["scene_id"]])
    return os.path.abspath(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Parse filenames from low/ folder and assign scene_id group."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/lol_dataset",
        help="Base directory for LOL dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="our485",
        choices=["our485", "eval15", "all"],
        help="Dataset split to parse (our485, eval15, or all)"
    )
    parser.add_argument(
        "--low-dir",
        type=str,
        default=None,
        help="Direct path to low/ folder (overrides data-dir/split)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/scene_splits.csv",
        help="Target CSV output path"
    )
    parser.add_argument(
        "--also-results",
        action="store_true",
        default=True,
        help="Also copy/export to results/scene_splits.csv"
    )
    args = parser.parse_args()

    if args.low_dir:
        print(f"Parsing directly from low directory: {args.low_dir}")
        records = parse_low_directory(args.low_dir)
    else:
        resolved_data_dir = resolve_dataset_dir(args.data_dir)
        print(f"Resolved base dataset directory: {resolved_data_dir}")

        if args.split in ["our485", "eval15"]:
            target_low = os.path.join(resolved_data_dir, args.split, "low")
            if not os.path.exists(target_low):
                # Check nested
                nested = os.path.join(resolved_data_dir, "lol_dataset", args.split, "low")
                if os.path.exists(nested):
                    target_low = nested
                else:
                    raise FileNotFoundError(f"Could not locate {target_low}")
            print(f"Parsing split '{args.split}' from: {target_low}")
            records = parse_low_directory(target_low)
        else: # "all"
            records = []
            for sp in ["our485", "eval15"]:
                sp_low = os.path.join(resolved_data_dir, sp, "low")
                if not os.path.exists(sp_low):
                    sp_low = os.path.join(resolved_data_dir, "lol_dataset", sp, "low")
                if os.path.exists(sp_low):
                    records.extend(parse_low_directory(sp_low))

    # Export primary CSV
    out_path = export_csv(records, args.output)
    print(f" Exported primary CSV ({len(records)} rows) to: {out_path}")

    # Optionally mirror to results/
    if args.also_results:
        results_path = os.path.join(REPO_ROOT, "results", "scene_splits.csv")
        export_csv(records, results_path)
        print(f" Mirrored CSV to results directory: {results_path}")

    # Summary Statistics
    df = pd.DataFrame(records)
    unique_scenes = df["scene_id"].nunique()
    print("\n" + "=" * 50)
    print("SCENE SPLIT MANIFEST SUMMARY")
    print("=" * 50)
    print(f"Total Images Parsed : {len(df)}")
    print(f"Unique Scene Groups : {unique_scenes}")
    print(f"Grouping Rule       : all '1_*' files mapped to scene_id '1'")
    print(f"CSV Columns         : {list(df.columns)}")
    print("=" * 50)

    print("\nSAMPLE TABLE PREVIEW (Top 10 rows):")
    print(df.head(10).to_markdown(index=False))

    print("\nSAMPLE TABLE PREVIEW (Bottom 5 rows):")
    print(df.tail(5).to_markdown(index=False))


if __name__ == "__main__":
    main()

