"""
Command Line Interface Module for Project Triathlon
Implements CLI subcommands: env, groups, run, paper, demo, smoke
Per Section 4, 5 & 6.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import sys
import argparse
import time
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import yaml
import cv2

from src.methods import CANONICAL_METHODS, METHODS
from src.features import FEATURE_NAMES, extract_features
from src.labels import compute_psnr, compute_ssim
from src.dataset import resolve_dataset_dir, get_dataset_manifest, load_and_preprocess_pair, compute_dhash, compute_scene_groups_dhash
from src.selectors import make_selector
from src.train import make_folds
from src.evaluate import GroupResampler, ci_of, paired_ci
from src.pipeline import prepare


def cmd_env(args: argparse.Namespace) -> None:
    """Checks dependencies, Python environment, and dataset paths."""
    print("=" * 60)
    print("PROJECT TRIATHLON · ENVIRONMENT VERIFICATION")
    print("=" * 60)
    print(f"Python Version: {sys.version.split()[0]}")

    packages = ["numpy", "cv2", "pandas", "sklearn", "xgboost", "joblib", "matplotlib", "yaml"]
    for p in packages:
        try:
            mod = __import__(p)
            v = getattr(mod, "__version__", "installed")
            print(f"  [OK] {p:<12}: {v}")
        except ImportError:
            print(f"  [FAIL] {p:<12}: NOT INSTALLED")

    print("\n--- Dataset Paths Verification ---")
    data_dir = resolve_dataset_dir()
    print(f"Base data directory: {data_dir}")

    datasets_to_check = [
        ("lol_v1_train", "our485", os.path.join(data_dir, "our485", "high")),
        ("lol_v1_test", "eval15", os.path.join(data_dir, "eval15", "high")),
        ("lol_v2_real_train", "Real_captured_train", "data/LOL-v2/Real_captured/Train/High"),
        ("lol_v2_real_test", "Real_captured_test", "data/LOL-v2/Real_captured/Test/Normal"),
    ]

    all_ok = True
    for name, split, path in datasets_to_check:
        if os.path.isdir(path):
            n = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg'))])
            print(f"  [OK] {name:<20}: {n} images found at {path}")
        else:
            # Check fallback
            if "Real_captured" in path:
                alt = path.replace("High", "Normal")
                if os.path.isdir(alt):
                    n = len([f for f in os.listdir(alt) if f.lower().endswith(('.png', '.jpg'))])
                    print(f"  [OK] {name:<20}: {n} images found at {alt}")
                    continue
            print(f"  [WARN] {name:<20}: Directory not found at {path}")
            all_ok = False

    if all_ok:
        print("\nENVIRONMENT STATUS: OK")
    else:
        print("\nENVIRONMENT STATUS: PARTIAL (Core LOL-v1 is ready)")


def cmd_groups(args: argparse.Namespace) -> None:
    """Computes reference image dHash and generates contact sheet of scene groups."""
    print("=" * 60)
    print("PROJECT TRIATHLON · SCENE GROUPING AUDIT")
    print("=" * 60)
    data = prepare(dataset=args.dataset, split=args.split, hamming_threshold=args.threshold)
    groups = data["groups"]
    manifest = data["manifest"]
    n_groups = len(np.unique(groups))
    print(f"Dataset: {args.dataset} [{args.split}]")
    print(f"Total pairs: {len(manifest)}")
    print(f"Hamming threshold: <= {args.threshold}")
    print(f"Unique scene groups found: {n_groups}")

    # Generate contact sheet for multi-image scene groups
    os.makedirs("results", exist_ok=True)
    out_png = f"results/groups_{args.dataset}_{args.split}.png"

    from collections import Counter
    counts = Counter(groups)
    multi_groups = [g for g, c in counts.items() if c > 1]
    print(f"Multi-image scenes: {len(multi_groups)} (largest scene has {max(counts.values())} images)")

    # Build contact sheet of top multi-image groups
    import matplotlib.pyplot as plt
    top_groups = [g for g, c in counts.most_common(6) if c > 1]
    if top_groups:
        fig, axes = plt.subplots(len(top_groups), 4, figsize=(12, 3 * len(top_groups)))
        if len(top_groups) == 1:
            axes = axes[np.newaxis, :]
        for row_idx, g in enumerate(top_groups):
            member_indices = np.where(groups == g)[0][:4]
            for col_idx in range(4):
                ax = axes[row_idx, col_idx]
                if col_idx < len(member_indices):
                    idx = member_indices[col_idx]
                    img = cv2.imread(manifest[idx]["high_path"])
                    if img is not None:
                        img_rgb = cv2.cvtColor(cv2.resize(img, (200, 150)), cv2.COLOR_BGR2RGB)
                        ax.imshow(img_rgb)
                    ax.set_title(f"Scene {g}: #{manifest[idx]['filename']}", fontsize=9)
                ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved contact sheet to {out_png}")
    else:
        print("No multi-image scenes found to plot.")


def cmd_smoke(args: argparse.Namespace) -> None:
    """Fast smoke test on synthetic data (never quote these numbers)."""
    print("=" * 60)
    print("PROJECT TRIATHLON · SMOKE TEST (Synthetic Data)")
    print("=" * 60)
    np.random.seed(42)
    n = 60
    d = 8
    X = np.random.randn(n, d)
    classes = CANONICAL_METHODS
    y = np.random.choice(classes, size=n)
    groups = np.repeat(np.arange(15), 4)

    # 5-fold CV test
    folds = make_folds(X, y, groups, n_splits=5, seed=42)
    print(f"Created {len(folds)} StratifiedGroupKFold folds successfully.")

    for mname in ["logreg", "rf", "best_fixed", "random"]:
        sel = make_selector(mname, feature_names=FEATURE_NAMES)
        tr, va = folds[0]
        sel.fit(X[tr], y[tr])
        preds = sel.predict(X[va])
        assert len(preds) == len(va)
        acc = float(np.mean(preds == y[va])) * 100.0
        print(f"  Smoke selector '{mname:<12}': Fold 0 accuracy = {acc:.1f}%")

    resampler = GroupResampler(groups, n_bootstraps=100, seed=42)
    vals = np.random.uniform(0.5, 3.5, size=n)
    pt, lo, hi = ci_of(vals, resampler)
    print(f"  Smoke cluster bootstrap: mean = {pt:.3f} [{lo:.3f}, {hi:.3f}]")
    print("SMOKE TEST OK (All components operational)")


def cmd_demo(args: argparse.Namespace) -> None:
    """Inference on a single image pair."""
    from src.demo import run_single_inference
    run_single_inference(args.image, args.reference, args.output, args.model_path)


def cmd_run(args: argparse.Namespace) -> None:
    """Executes the canonical evaluation run."""
    from experiments.run_experiment import run_pipeline_cv
    run_pipeline_cv(run_name=args.run_name, no_assets=args.no_assets)


def cmd_paper(args: argparse.Namespace) -> None:
    """Builds LaTeX paper assets from run outputs."""
    from src.paper_assets import generate_all_paper_assets
    generate_all_paper_assets(run_name=args.run)


def main():
    parser = argparse.ArgumentParser(description="Project Triathlon CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # env
    p_env = subparsers.add_parser("env", help="Verify dependencies and dataset paths")

    # groups
    p_groups = subparsers.add_parser("groups", help="Audit scene grouping via reference dHash")
    p_groups.add_argument("--dataset", default="lol_v1", help="Dataset name")
    p_groups.add_argument("--split", default="our485", help="Split name")
    p_groups.add_argument("--threshold", type=int, default=10, help="dHash Hamming distance threshold")

    # smoke
    p_smoke = subparsers.add_parser("smoke", help="Run fast synthetic smoke test")

    # demo
    p_demo = subparsers.add_parser("demo", help="Run inference on single image")
    p_demo.add_argument("--image", required=True, help="Path to input degraded image")
    p_demo.add_argument("--reference", default=None, help="Path to clean ground truth reference")
    p_demo.add_argument("--output", default="results/demo_output", help="Output directory")
    p_demo.add_argument("--model_path", default=None, help="Model checkpoint path")

    # run
    p_run = subparsers.add_parser("run", help="Execute evaluation run")
    p_run.add_argument("--run-name", default="explore_default", help="Run name identifier")
    p_run.add_argument("--no-assets", action="store_true", help="Skip figure/table generation")

    # paper
    p_paper = subparsers.add_parser("paper", help="Generate paper tables and figures")
    p_paper.add_argument("--run", default="phase0_candidate", help="Run name to build assets from")

    args = parser.parse_args()

    if args.command == "env":
        cmd_env(args)
    elif args.command == "groups":
        cmd_groups(args)
    elif args.command == "smoke":
        cmd_smoke(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "paper":
        cmd_paper(args)


if __name__ == "__main__":
    main()
