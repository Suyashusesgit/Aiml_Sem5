"""
Comprehensive Decision Experiments Runner (E0–E5)
Implements Section 6 specifications for resolving Logistic Regression vs. Random Forest.
"""

import os
import sys
import time
import json
import hashlib
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import yaml
import cv2
from scipy.sparse import csgraph

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.methods import CANONICAL_METHODS, METHODS
from src.features import FEATURE_NAMES, validate_feature_dataframe, extract_features
from src.labels import compute_psnr, compute_ssim
from src.dataset import compute_dhash, compute_scene_groups_dhash, load_and_preprocess_pair
from src.selectors import make_selector, ClassifierSelector
from src.train import make_folds
from src.evaluate import GroupResampler, ci_of, paired_ci
from src.pipeline import prepare

CANONICAL_SEEDS = [0, 1, 2]


def compute_meta(run_id: str, command: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import sklearn, xgboost, cv2
    meta = {
        "run_id": run_id,
        "command": command,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": CANONICAL_SEEDS,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": sys.modules.get("scipy", type("M", (), {"__version__": "unknown"})).__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "cv2": cv2.__version__,
        },
    }
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        meta["git_commit"] = commit
    except Exception:
        meta["git_commit"] = "unknown"
    if extra:
        meta.update(extra)
    return meta


# ==============================================================================
# E0: REPRODUCE & SEED SENSITIVITY
# ==============================================================================

def run_e0_reproduce(
    data: Dict[str, Any],
    out_dir: str = "results/decision/explore_default",
    seeds: List[int] = CANONICAL_SEEDS
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("EXPERIMENT E0: REPRODUCE DEFAULT ARM (StratifiedGroupKFold x Seeds)")
    print("=" * 70)
    os.makedirs(out_dir, exist_ok=True)

    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    df_scores = data["df_scores"]
    n_samples = len(X)

    oracle_psnr = df_scores[[f"psnr_{m}" for m in CANONICAL_METHODS]].max(axis=1).values

    models_to_run = [
        ("Logistic Regression", "logreg", {}),
        ("Random Forest", "rf", {"n_jobs": 1}),
        ("HistGradientBoosting", "histgb", {}),
        ("Ridge PTO", "ridge_pto", {}),
        ("Hand-Written Rule", "rule_fixed", {}),
        ("Best Fixed Method", "best_fixed", {}),
        ("Fixed Majority", "fixed_majority", {}),
        ("Random Choice", "random", {}),
        ("Theoretical Oracle", "oracle", {}),
    ]

    # Per-sample predictions across repetitions: (n_seeds, n_samples)
    oof_predictions = {name: np.empty((len(seeds), n_samples), dtype=object) for name, _, _ in models_to_run}
    oof_regrets = {name: np.empty((len(seeds), n_samples), dtype=np.float64) for name, _, _ in models_to_run}

    resampler = GroupResampler(groups, n_bootstraps=2000, seed=42)

    for s_idx, seed in enumerate(seeds):
        folds = make_folds(X, y, groups, n_splits=5, seed=seed)
        for fold, (tr_idx, va_idx) in enumerate(folds):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]
            scores_tr = df_scores.iloc[tr_idx]

            for name, sel_type, kwargs in models_to_run:
                if name == "Theoretical Oracle":
                    preds = y_va
                else:
                    sel = make_selector(sel_type, feature_names=FEATURE_NAMES, random_state=seed, **kwargs)
                    sel.fit(X_tr, y_tr, df_scores=scores_tr)
                    preds = sel.predict(X_va)

                oof_predictions[name][s_idx, va_idx] = preds

                # Compute fold regret
                for i, row_idx in enumerate(va_idx):
                    chosen = preds[i]
                    p_chosen = df_scores.iloc[row_idx][f"psnr_{chosen}"]
                    oof_regrets[name][s_idx, row_idx] = oracle_psnr[row_idx] - p_chosen

    # Average regrets per image across the 3 seeds
    summary_rows = []
    mean_regrets_per_image = {}

    # Best-fixed regret for gap-closed calculation
    best_fixed_avg_regret = np.mean(np.mean(oof_regrets["Best Fixed Method"], axis=0))

    for name, _, _ in models_to_run:
        avg_regret_i = np.mean(oof_regrets[name], axis=0) # (n_samples,)
        mean_regrets_per_image[name] = avg_regret_i

        # Accuracies across repetitions
        correct_per_rep = [np.mean(oof_predictions[name][s] == y) * 100.0 for s in range(len(seeds))]
        mean_acc = float(np.mean(correct_per_rep))

        # Bootstrap CIs on mean regret
        pt_reg, lo_reg, hi_reg = ci_of(avg_regret_i, resampler)
        max_r = float(np.max(avg_regret_i))
        p90_r = float(np.percentile(avg_regret_i, 90.0))

        # Gap closed
        gap_closed = (best_fixed_avg_regret - pt_reg) / best_fixed_avg_regret * 100.0 if best_fixed_avg_regret > 0 else 0.0

        summary_rows.append({
            "selector": name,
            "top1_acc": round(mean_acc, 2),
            "mean_regret_db": round(pt_reg, 3),
            "ci_lo": round(lo_reg, 3),
            "ci_hi": round(hi_reg, 3),
            "p90_regret_db": round(p90_r, 3),
            "max_regret_db": round(max_r, 3),
            "gap_closed_pct": round(gap_closed, 1),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, "summary_cv.csv"), index=False)

    # Paired comparison: d = RF - LR
    rf_regrets = mean_regrets_per_image["Random Forest"]
    lr_regrets = mean_regrets_per_image["Logistic Regression"]
    d_point, d_lo, d_hi = paired_ci(rf_regrets, lr_regrets, resampler)

    df_paired = pd.DataFrame([{
        "comparison": "Random Forest - Logistic Regression",
        "d_mean_db": round(d_point, 4),
        "d_ci_lo": round(d_lo, 4),
        "d_ci_hi": round(d_hi, 4),
        "rf_mean": round(float(np.mean(rf_regrets)), 3),
        "lr_mean": round(float(np.mean(lr_regrets)), 3),
        "seeds": str(seeds),
    }])
    df_paired.to_csv(os.path.join(out_dir, "paired_learners.csv"), index=False)

    print(df_summary.to_string(index=False))
    print(f"\nPaired Difference (RF - LR): {d_point:+.3f} dB [95% CI: {d_lo:+.3f}, {d_hi:+.3f}] dB")
    if d_point > 0:
        print("  -> Logistic Regression achieves LOWER regret than Random Forest.")
    else:
        print("  -> Random Forest achieves LOWER regret than Logistic Regression.")

    # Sweep seeds 0 to 9 to evaluate fold assignment variance
    print("\n--- Sweeping Seeds 0–9 for LR vs. RF Stability ---")
    seed_rows = []
    for s in range(10):
        f_folds = make_folds(X, y, groups, n_splits=5, seed=s)
        rf_reg_s = np.empty(n_samples)
        lr_reg_s = np.empty(n_samples)

        for tr, va in f_folds:
            sel_lr = make_selector("logreg", feature_names=FEATURE_NAMES, random_state=s)
            sel_rf = make_selector("rf", feature_names=FEATURE_NAMES, random_state=s, n_jobs=1)
            sel_lr.fit(X[tr], y[tr], df_scores=df_scores.iloc[tr])
            sel_rf.fit(X[tr], y[tr], df_scores=df_scores.iloc[tr])
            p_lr = sel_lr.predict(X[va])
            p_rf = sel_rf.predict(X[va])

            for idx, r_idx in enumerate(va):
                lr_reg_s[r_idx] = oracle_psnr[r_idx] - df_scores.iloc[r_idx][f"psnr_{p_lr[idx]}"]
                rf_reg_s[r_idx] = oracle_psnr[r_idx] - df_scores.iloc[r_idx][f"psnr_{p_rf[idx]}"]

        d_s = float(np.mean(rf_reg_s) - np.mean(lr_reg_s))
        seed_rows.append({
            "seed": s,
            "lr_mean_regret": round(float(np.mean(lr_reg_s)), 3),
            "rf_mean_regret": round(float(np.mean(rf_reg_s)), 3),
            "d_db": round(d_s, 4),
        })

    df_seeds = pd.DataFrame(seed_rows)
    df_seeds.to_csv(os.path.join(out_dir, "seed_sensitivity_e0.csv"), index=False)
    print(df_seeds.to_string(index=False))

    # Meta
    meta = compute_meta("explore_default", "python -m src run --run-name explore_default", {
        "d_mean_db": d_point, "d_ci_lo": d_lo, "d_ci_hi": d_hi
    })
    with open(os.path.join(out_dir, "run_meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)

    with open(os.path.join(out_dir, "README.txt"), "w") as fp:
        fp.write(f"Experiment E0: Default Arm Reproduction\nCommand: python -m src run --run-name explore_default\nSeeds: {seeds}\nGit commit: {meta['git_commit']}\n")

    return {"summary": df_summary, "paired": df_paired, "seeds": df_seeds, "d": d_point, "d_ci": (d_lo, d_hi)}


# ==============================================================================
# E1: FAIR NESTED TUNING (LR vs. RF with Identical Budget)
# ==============================================================================

def generate_tuning_grids(budget: int = 30, rng_seed: int = 123) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generates 30 random hyperparameter configurations for LR and RF."""
    rng = np.random.RandomState(rng_seed)

    # Config 0 is default
    lr_configs = [{"C": 1.0, "class_weight": None, "max_iter": 2000}]
    rf_configs = [{"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": None}]

    for _ in range(budget - 1):
        # LR grid: C = 10^U(-3, 3), class_weight in {None, 'balanced'}
        u = rng.uniform(-3.0, 3.0)
        c_val = float(10.0 ** u)
        cw = rng.choice([None, "balanced"])
        lr_configs.append({"C": c_val, "class_weight": cw, "max_iter": 2000})

        # RF grid: max_depth in {3, 5, 8, None}, min_samples_leaf in {1, 2, 3, 5, 10},
        # max_features in {'sqrt', 0.5, 1.0}, class_weight in {None, 'balanced_subsample'}
        depth = rng.choice([3, 5, 8, None])
        leaf = int(rng.choice([1, 2, 3, 5, 10]))
        feat = rng.choice(["sqrt", 0.5, 1.0])
        rf_cw = rng.choice([None, "balanced_subsample"])
        rf_configs.append({
            "n_estimators": 300,
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "max_features": feat,
            "class_weight": rf_cw,
        })
    return lr_configs, rf_configs


def run_e1_nested_tuning(
    data: Dict[str, Any],
    out_dir: str = "results/decision/e1_tuning",
    seeds: List[int] = CANONICAL_SEEDS,
    budget: int = 30
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"EXPERIMENT E1: FAIR NESTED TUNING (Budget = {budget} configs each)")
    print("=" * 70)
    os.makedirs(out_dir, exist_ok=True)

    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    df_scores = data["df_scores"]
    n_samples = len(X)
    oracle_psnr = df_scores[[f"psnr_{m}" for m in CANONICAL_METHODS]].max(axis=1).values

    lr_configs, rf_configs = generate_tuning_grids(budget=budget, rng_seed=123)

    # Storage for outer fold predictions and regrets
    tuned_preds = {"lr": np.empty((len(seeds), n_samples), dtype=object), "rf": np.empty((len(seeds), n_samples), dtype=object)}
    tuned_regrets = {"lr": np.empty((len(seeds), n_samples), dtype=np.float64), "rf": np.empty((len(seeds), n_samples), dtype=np.float64)}
    default_regrets = {"lr": np.empty((len(seeds), n_samples), dtype=np.float64), "rf": np.empty((len(seeds), n_samples), dtype=np.float64)}

    chosen_configs_log = []

    resampler = GroupResampler(groups, n_bootstraps=2000, seed=42)
    t0 = time.time()

    for s_idx, seed in enumerate(seeds):
        outer_folds = make_folds(X, y, groups, n_splits=5, seed=seed)

        for fold_idx, (out_tr, out_va) in enumerate(outer_folds):
            # Anti-leakage assertion: outer test strictly isolated
            assert len(set(groups[out_tr]).intersection(set(groups[out_va]))) == 0
            
            X_tr, y_tr = X[out_tr], y[out_tr]
            groups_tr = groups[out_tr]
            scores_tr = df_scores.iloc[out_tr]
            oracle_psnr_tr = oracle_psnr[out_tr]

            X_va, y_va = X[out_va], y[out_va]
            scores_va = df_scores.iloc[out_va]
            oracle_psnr_va = oracle_psnr[out_va]

            # Inner 3-fold CV over outer-training groups
            inner_folds = make_folds(X_tr, y_tr, groups_tr, n_splits=3, seed=123)

            # 1. Tune LR
            best_lr_idx = 0
            best_lr_score = float("inf")
            for c_idx, cfg in enumerate(lr_configs):
                inner_regrets = []
                for in_tr, in_va in inner_folds:
                    sel = make_selector("logreg", feature_names=FEATURE_NAMES, random_state=seed, **cfg)
                    sel.fit(X_tr[in_tr], y_tr[in_tr], df_scores=scores_tr.iloc[in_tr])
                    preds = sel.predict(X_tr[in_va])
                    for i, r_i in enumerate(in_va):
                        p_m = scores_tr.iloc[r_i][f"psnr_{preds[i]}"]
                        inner_regrets.append(oracle_psnr_tr[r_i] - p_m)
                mean_r = float(np.mean(inner_regrets))
                if mean_r < best_lr_score:
                    best_lr_score = mean_r
                    best_lr_idx = c_idx

            # 2. Tune RF
            best_rf_idx = 0
            best_rf_score = float("inf")
            for c_idx, cfg in enumerate(rf_configs):
                inner_regrets = []
                for in_tr, in_va in inner_folds:
                    sel = make_selector("rf", feature_names=FEATURE_NAMES, random_state=seed, n_jobs=1, **cfg)
                    sel.fit(X_tr[in_tr], y_tr[in_tr], df_scores=scores_tr.iloc[in_tr])
                    preds = sel.predict(X_tr[in_va])
                    for i, r_i in enumerate(in_va):
                        p_m = scores_tr.iloc[r_i][f"psnr_{preds[i]}"]
                        inner_regrets.append(oracle_psnr_tr[r_i] - p_m)
                mean_r = float(np.mean(inner_regrets))
                if mean_r < best_rf_score:
                    best_rf_score = mean_r
                    best_rf_idx = c_idx

            # Refit best tuned configs on entire outer-training part
            best_lr_cfg = lr_configs[best_lr_idx]
            best_rf_cfg = rf_configs[best_rf_idx]

            sel_lr_tuned = make_selector("logreg", feature_names=FEATURE_NAMES, random_state=seed, **best_lr_cfg)
            sel_rf_tuned = make_selector("rf", feature_names=FEATURE_NAMES, random_state=seed, n_jobs=1, **best_rf_cfg)
            sel_lr_tuned.fit(X_tr, y_tr, df_scores=scores_tr)
            sel_rf_tuned.fit(X_tr, y_tr, df_scores=scores_tr)

            p_lr = sel_lr_tuned.predict(X_va)
            p_rf = sel_rf_tuned.predict(X_va)

            tuned_preds["lr"][s_idx, out_va] = p_lr
            tuned_preds["rf"][s_idx, out_va] = p_rf

            for i, r_i in enumerate(out_va):
                tuned_regrets["lr"][s_idx, r_i] = oracle_psnr_va[i] - scores_va.iloc[i][f"psnr_{p_lr[i]}"]
                tuned_regrets["rf"][s_idx, r_i] = oracle_psnr_va[i] - scores_va.iloc[i][f"psnr_{p_rf[i]}"]

            # Also evaluate defaults on outer fold for fair comparison
            sel_lr_def = make_selector("logreg", feature_names=FEATURE_NAMES, random_state=seed, **lr_configs[0])
            sel_rf_def = make_selector("rf", feature_names=FEATURE_NAMES, random_state=seed, n_jobs=1, **rf_configs[0])
            sel_lr_def.fit(X_tr, y_tr, df_scores=scores_tr)
            sel_rf_def.fit(X_tr, y_tr, df_scores=scores_tr)
            p_lr_def = sel_lr_def.predict(X_va)
            p_rf_def = sel_rf_def.predict(X_va)

            for i, r_i in enumerate(out_va):
                default_regrets["lr"][s_idx, r_i] = oracle_psnr_va[i] - scores_va.iloc[i][f"psnr_{p_lr_def[i]}"]
                default_regrets["rf"][s_idx, r_i] = oracle_psnr_va[i] - scores_va.iloc[i][f"psnr_{p_rf_def[i]}"]

            chosen_configs_log.append({
                "seed": seed, "fold": fold_idx,
                "best_lr_idx": best_lr_idx, "best_lr_C": best_lr_cfg["C"], "best_lr_cw": str(best_lr_cfg["class_weight"]),
                "best_rf_idx": best_rf_idx, "best_rf_depth": str(best_rf_cfg["max_depth"]), "best_rf_leaf": best_rf_cfg["min_samples_leaf"],
                "best_rf_feat": str(best_rf_cfg["max_features"]),
            })
            print(f"  Seed {seed} Fold {fold_idx}: Selected LR config #{best_lr_idx} (C={best_lr_cfg['C']:.3f}), RF config #{best_rf_idx} (depth={best_rf_cfg['max_depth']}, leaf={best_rf_cfg['min_samples_leaf']})")

    elapsed = time.time() - t0
    print(f"\nNested tuning completed in {elapsed:.1f}s.")

    # Save chosen configs
    df_configs = pd.DataFrame(chosen_configs_log)
    df_configs.to_csv(os.path.join(out_dir, "chosen_configs_per_fold.csv"), index=False)

    # Average regrets across the 3 seeds
    lr_tuned_i = np.mean(tuned_regrets["lr"], axis=0)
    rf_tuned_i = np.mean(tuned_regrets["rf"], axis=0)
    lr_def_i = np.mean(default_regrets["lr"], axis=0)
    rf_def_i = np.mean(default_regrets["rf"], axis=0)

    # Paired differences: tuned minus default per model
    lr_gain_pt, lr_gain_lo, lr_gain_hi = paired_ci(lr_tuned_i, lr_def_i, resampler)
    rf_gain_pt, rf_gain_lo, rf_gain_hi = paired_ci(rf_tuned_i, rf_def_i, resampler)

    # Key Pre-Committed Decision Quantity: d = mean regret(RF_tuned) - mean regret(LR_tuned)
    d_point, d_lo, d_hi = paired_ci(rf_tuned_i, lr_tuned_i, resampler)

    lr_t_mean, lr_t_lo, lr_t_hi = ci_of(lr_tuned_i, resampler)
    rf_t_mean, rf_t_lo, rf_t_hi = ci_of(rf_tuned_i, resampler)

    results_table = pd.DataFrame([
        {"arm": "LR Default", "mean_regret_db": round(float(np.mean(lr_def_i)), 3)},
        {"arm": "LR Tuned", "mean_regret_db": round(lr_t_mean, 3), "ci_lo": round(lr_t_lo, 3), "ci_hi": round(lr_t_hi, 3), "tuning_gain_dB": round(lr_gain_pt, 4), "gain_ci": f"[{lr_gain_lo:.3f}, {lr_gain_hi:.3f}]"},
        {"arm": "RF Default", "mean_regret_db": round(float(np.mean(rf_def_i)), 3)},
        {"arm": "RF Tuned", "mean_regret_db": round(rf_t_mean, 3), "ci_lo": round(rf_t_lo, 3), "ci_hi": round(rf_t_hi, 3), "tuning_gain_dB": round(rf_gain_pt, 4), "gain_ci": f"[{rf_gain_lo:.3f}, {rf_gain_hi:.3f}]"},
        {"arm": "TUNED DIFFERENCE (RF - LR)", "mean_regret_db": round(d_point, 4), "ci_lo": round(d_lo, 4), "ci_hi": round(d_hi, 4), "tuning_gain_dB": "--", "gain_ci": "--"}
    ])
    results_table.to_csv(os.path.join(out_dir, "e1_tuning_summary.csv"), index=False)

    print("\n" + "=" * 70)
    print("EXPERIMENT E1 TUNING RESULTS:")
    print("=" * 70)
    print(results_table.to_string(index=False))
    print(f"\nDECISION QUANTITY d (Tuned RF - Tuned LR): {d_point:+.4f} dB [95% CI: {d_lo:+.4f}, {d_hi:+.4f}] dB")

    meta = compute_meta("e1_tuning", "run_e1_nested_tuning", {
        "budget": budget, "d_mean_db": d_point, "d_ci_lo": d_lo, "d_ci_hi": d_hi, "elapsed_sec": elapsed
    })
    with open(os.path.join(out_dir, "run_meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)

    return {
        "d": d_point, "d_ci": (d_lo, d_hi), "lr_tuned": (lr_t_mean, lr_t_lo, lr_t_hi),
        "rf_tuned": (rf_t_mean, rf_t_lo, rf_t_hi), "lr_gain": (lr_gain_pt, lr_gain_lo, lr_gain_hi),
        "rf_gain": (rf_gain_pt, rf_gain_lo, rf_gain_hi)
    }


# ==============================================================================
# E2: LEAKAGE & GROUPING SENSITIVITY
# ==============================================================================

def run_e2_leakage(
    data: Dict[str, Any],
    out_dir: str = "results/decision/e2_leakage",
    seeds: List[int] = CANONICAL_SEEDS
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("EXPERIMENT E2: LEAKAGE & GROUPING SENSITIVITY")
    print("=" * 70)
    os.makedirs(out_dir, exist_ok=True)

    from sklearn.model_selection import StratifiedKFold

    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    df_scores = data["df_scores"]
    hashes = data["hashes"]
    n_samples = len(X)
    oracle_psnr = df_scores[[f"psnr_{m}" for m in CANONICAL_METHODS]].max(axis=1).values

    # E2(a): Plain StratifiedKFold vs. Grouped CV
    plain_regrets = {"lr": [], "rf": []}
    grouped_regrets = {"lr": [], "rf": []}

    for seed in seeds:
        # Grouped
        g_folds = make_folds(X, y, groups, n_splits=5, seed=seed)
        for tr, va in g_folds:
            sel_lr = make_selector("logreg", feature_names=FEATURE_NAMES, random_state=seed)
            sel_rf = make_selector("rf", feature_names=FEATURE_NAMES, random_state=seed, n_jobs=1)
            sel_lr.fit(X[tr], y[tr], df_scores=df_scores.iloc[tr])
            sel_rf.fit(X[tr], y[tr], df_scores=df_scores.iloc[tr])
            p_lr = sel_lr.predict(X[va])
            p_rf = sel_rf.predict(X[va])
            for i, r_i in enumerate(va):
                grouped_regrets["lr"].append(oracle_psnr[r_i] - df_scores.iloc[r_i][f"psnr_{p_lr[i]}"])
                grouped_regrets["rf"].append(oracle_psnr[r_i] - df_scores.iloc[r_i][f"psnr_{p_rf[i]}"])

        # Plain (ungrouped) StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y):
            sel_lr = make_selector("logreg", feature_names=FEATURE_NAMES, random_state=seed)
            sel_rf = make_selector("rf", feature_names=FEATURE_NAMES, random_state=seed, n_jobs=1)
            sel_lr.fit(X[tr], y[tr], df_scores=df_scores.iloc[tr])
            sel_rf.fit(X[tr], y[tr], df_scores=df_scores.iloc[tr])
            p_lr = sel_lr.predict(X[va])
            p_rf = sel_rf.predict(X[va])
            for i, r_i in enumerate(va):
                plain_regrets["lr"].append(oracle_psnr[r_i] - df_scores.iloc[r_i][f"psnr_{p_lr[i]}"])
                plain_regrets["rf"].append(oracle_psnr[r_i] - df_scores.iloc[r_i][f"psnr_{p_rf[i]}"])

    lr_grp_mean = float(np.mean(grouped_regrets["lr"]))
    lr_pln_mean = float(np.mean(plain_regrets["lr"]))
    rf_grp_mean = float(np.mean(grouped_regrets["rf"]))
    rf_pln_mean = float(np.mean(plain_regrets["rf"]))

    lr_inflation = lr_grp_mean - lr_pln_mean
    rf_inflation = rf_grp_mean - rf_pln_mean

    df_a = pd.DataFrame([
        {"model": "Logistic Regression", "grouped_regret": round(lr_grp_mean, 3), "plain_regret": round(lr_pln_mean, 3), "inflation_dB": round(lr_inflation, 4)},
        {"model": "Random Forest", "grouped_regret": round(rf_grp_mean, 3), "plain_regret": round(rf_pln_mean, 3), "inflation_dB": round(rf_inflation, 4)}
    ])
    df_a.to_csv(os.path.join(out_dir, "e2a_leakage_inflation.csv"), index=False)
    print("\n--- E2(a) Leakage Inflation (Grouped minus Plain) ---")
    print(df_a.to_string(index=False))

    # E2(b): Grouping Threshold Sweep (6, 10, 14, 18)
    print("\n--- E2(b) Grouping Threshold Sweep (6, 10, 14, 18) ---")
    thresholds = [6, 10, 14, 18]
    thresh_rows = []

    resampler_base = GroupResampler(groups, n_bootstraps=2000, seed=42)

    thr_18_d_hi = None

    for thr in thresholds:
        grp_thr = compute_scene_groups_dhash(hashes, hamming_threshold=thr)
        n_grp = len(np.unique(grp_thr))

        rf_reg_all = np.empty((len(seeds), n_samples))
        lr_reg_all = np.empty((len(seeds), n_samples))

        for s_idx, seed in enumerate(seeds):
            folds = make_folds(X, y, grp_thr, n_splits=5, seed=seed)
            for tr, va in folds:
                sel_lr = make_selector("logreg", feature_names=FEATURE_NAMES, random_state=seed)
                sel_rf = make_selector("rf", feature_names=FEATURE_NAMES, random_state=seed, n_jobs=1)
                sel_lr.fit(X[tr], y[tr], df_scores=df_scores.iloc[tr])
                sel_rf.fit(X[tr], y[tr], df_scores=df_scores.iloc[tr])
                p_lr = sel_lr.predict(X[va])
                p_rf = sel_rf.predict(X[va])
                for i, r_i in enumerate(va):
                    lr_reg_all[s_idx, r_i] = oracle_psnr[r_i] - df_scores.iloc[r_i][f"psnr_{p_lr[i]}"]
                    rf_reg_all[s_idx, r_i] = oracle_psnr[r_i] - df_scores.iloc[r_i][f"psnr_{p_rf[i]}"]

        lr_avg = np.mean(lr_reg_all, axis=0)
        rf_avg = np.mean(rf_reg_all, axis=0)

        res_thr = GroupResampler(grp_thr, n_bootstraps=2000, seed=42)
        d_p, d_lo, d_hi = paired_ci(rf_avg, lr_avg, res_thr)

        if thr == 18:
            thr_18_d_hi = d_hi

        thresh_rows.append({
            "threshold": thr,
            "n_scene_groups": n_grp,
            "lr_mean_regret": round(float(np.mean(lr_avg)), 3),
            "rf_mean_regret": round(float(np.mean(rf_avg)), 3),
            "d_db": round(d_p, 4),
            "d_ci_lo": round(d_lo, 4),
            "d_ci_hi": round(d_hi, 4),
        })

    df_b = pd.DataFrame(thresh_rows)
    df_b.to_csv(os.path.join(out_dir, "e2b_threshold_sweep.csv"), index=False)
    print(df_b.to_string(index=False))

    meta = compute_meta("e2_leakage", "run_e2_leakage", {"thr_18_d_hi": thr_18_d_hi})
    with open(os.path.join(out_dir, "run_meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)

    return {"e2a": df_a, "e2b": df_b, "thr_18_d_hi": thr_18_d_hi}


# ==============================================================================
# E3: LEARNING CURVES
# ==============================================================================

def run_e3_learning_curves(
    data: Dict[str, Any],
    out_dir: str = "results/decision/e3_learning_curve",
    fractions: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0],
    seeds: List[int] = CANONICAL_SEEDS
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("EXPERIMENT E3: LEARNING CURVES (Scene Fraction Sweep)")
    print("=" * 70)
    os.makedirs(out_dir, exist_ok=True)

    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    df_scores = data["df_scores"]
    n_samples = len(X)
    oracle_psnr = df_scores[[f"psnr_{m}" for m in CANONICAL_METHODS]].max(axis=1).values

    curve_rows = []

    for frac in fractions:
        lr_reg_frac = []
        rf_reg_frac = []

        for seed in seeds:
            folds = make_folds(X, y, groups, n_splits=5, seed=seed)
            for tr, va in folds:
                tr_groups = np.unique(groups[tr])
                if frac < 1.0:
                    rng = np.random.RandomState(seed + int(frac * 100))
                    n_keep = max(1, int(len(tr_groups) * frac))
                    keep_groups = set(rng.choice(tr_groups, size=n_keep, replace=False))
                    sub_tr = [idx for idx in tr if groups[idx] in keep_groups]
                else:
                    sub_tr = tr

                sel_lr = make_selector("logreg", feature_names=FEATURE_NAMES, random_state=seed)
                sel_rf = make_selector("rf", feature_names=FEATURE_NAMES, random_state=seed, n_jobs=1)
                sel_lr.fit(X[sub_tr], y[sub_tr], df_scores=df_scores.iloc[sub_tr])
                sel_rf.fit(X[sub_tr], y[sub_tr], df_scores=df_scores.iloc[sub_tr])

                p_lr = sel_lr.predict(X[va])
                p_rf = sel_rf.predict(X[va])

                for i, r_i in enumerate(va):
                    lr_reg_frac.append(oracle_psnr[r_i] - df_scores.iloc[r_i][f"psnr_{p_lr[i]}"])
                    rf_reg_frac.append(oracle_psnr[r_i] - df_scores.iloc[r_i][f"psnr_{p_rf[i]}"])

        lr_mean = float(np.mean(lr_reg_frac))
        rf_mean = float(np.mean(rf_reg_frac))
        d_val = rf_mean - lr_mean

        curve_rows.append({
            "scene_fraction": frac,
            "approx_scenes": int(228 * 0.8 * frac),
            "lr_mean_regret": round(lr_mean, 3),
            "rf_mean_regret": round(rf_mean, 3),
            "d_db": round(d_val, 4),
        })

    df_curve = pd.DataFrame(curve_rows)
    df_curve.to_csv(os.path.join(out_dir, "learning_curves.csv"), index=False)
    print(df_curve.to_string(index=False))

    meta = compute_meta("e3_learning_curve", "run_e3_learning_curves")
    with open(os.path.join(out_dir, "run_meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)

    return {"curve": df_curve}


# ==============================================================================
# E4: EXTERNAL VALIDITY & NEAR-DUPLICATE AUDIT
# ==============================================================================

def run_e4_external_validity(
    data_v1: Dict[str, Any],
    out_dir: str = "results/decision/e4_external"
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("EXPERIMENT E4: EXTERNAL VALIDITY (LOL-v2 Real & Reverse Direction)")
    print("=" * 70)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load LOL-v2 Real Test pairs
    real_test_low = "data/LOL-v2/Real_captured/Test/Low"
    real_test_norm = "data/LOL-v2/Real_captured/Test/Normal"

    pairs_test = []
    for f in sorted(os.listdir(real_test_low)):
        if f.endswith(".png"):
            pairs_test.append({
                "low_path": os.path.join(real_test_low, f),
                "high_path": os.path.join(real_test_norm, f.replace("low", "normal")),
                "filename": f,
            })

    # Load LOL-v1 eval15 pairs
    eval15_low = "data/lol_dataset/eval15/low"
    eval15_high = "data/lol_dataset/eval15/high"
    pairs_eval15 = []
    if os.path.isdir(eval15_low):
        for f in sorted(os.listdir(eval15_low)):
            if f.endswith(".png"):
                pairs_eval15.append({
                    "low_path": os.path.join(eval15_low, f),
                    "high_path": os.path.join(eval15_high, f),
                    "filename": f,
                })

    # Near-duplicate audit (Hamming <= 10) between v1 train and test sets
    print(f"Auditing near-duplicates against {len(data_v1['hashes'])} LOL-v1 train hashes...")
    v1_hashes = data_v1["hashes"]

    # Process test pairs
    def process_test_split(pairs_list: List[Dict[str, str]], name: str):
        valid_pairs = []
        excluded_count = 0
        feature_rows = []
        score_rows = []

        for p in pairs_list:
            clean, deg, _ = load_and_preprocess_pair(p["low_path"], p["high_path"], target_long_side=512)
            h = compute_dhash(clean)
            # Check min distance to v1 train
            min_dist = int(np.min(np.sum(v1_hashes != h, axis=1)))
            if min_dist <= 10:
                excluded_count += 1
                continue

            valid_pairs.append(p)
            feats = extract_features(deg, include_engineered=False)
            feats["filename"] = p["filename"]
            feature_rows.append(feats)

            clean_rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
            deg_rgb = cv2.cvtColor(deg, cv2.COLOR_BGR2RGB)
            s_row = {"filename": p["filename"]}
            for m in CANONICAL_METHODS:
                enh_bgr = METHODS[m](deg)
                enh_rgb = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2RGB)
                s_row[f"psnr_{m}"] = compute_psnr(clean_rgb, enh_rgb, max_val=255.0)
            s_row["oracle_psnr"] = max(s_row[f"psnr_{m}"] for m in CANONICAL_METHODS)
            score_rows.append(s_row)

        print(f"  {name}: {len(pairs_list)} total pairs -> {excluded_count} excluded near-duplicates -> {len(valid_pairs)} evaluated")
        return pd.DataFrame(feature_rows), pd.DataFrame(score_rows), excluded_count

    df_f_real, df_s_real, excl_real = process_test_split(pairs_test, "LOL-v2 Real Test")
    df_f_eval, df_s_eval, excl_eval = process_test_split(pairs_eval15, "LOL-v1 eval15")

    # Direction 1: Fit on all LOL-v1 our485 with E1 procedure
    print("\nFitting champion candidates on entire LOL-v1 our485...")
    X_v1 = data_v1["X"]
    y_v1 = data_v1["y"]
    df_scores_v1 = data_v1["df_scores"]

    sel_lr_dir1 = make_selector("logreg", feature_names=FEATURE_NAMES, random_state=42)
    sel_rf_dir1 = make_selector("rf", feature_names=FEATURE_NAMES, random_state=42, n_jobs=1)
    sel_lr_dir1.fit(X_v1, y_v1, df_scores=df_scores_v1)
    sel_rf_dir1.fit(X_v1, y_v1, df_scores=df_scores_v1)

    # Evaluate on LOL-v2 Real Test
    feat_cols = [c for c in FEATURE_NAMES if c in df_f_real.columns]
    X_real = df_f_real[feat_cols].values
    pred_lr_real = sel_lr_dir1.predict(X_real)
    pred_rf_real = sel_rf_dir1.predict(X_real)

    r_lr_real = [df_s_real.iloc[i]["oracle_psnr"] - df_s_real.iloc[i][f"psnr_{pred_lr_real[i]}"] for i in range(len(X_real))]
    r_rf_real = [df_s_real.iloc[i]["oracle_psnr"] - df_s_real.iloc[i][f"psnr_{pred_rf_real[i]}"] for i in range(len(X_real))]

    d_real_test = float(np.mean(r_rf_real) - np.mean(r_lr_real))

    # Evaluate on LOL-v1 eval15
    X_eval = df_f_eval[feat_cols].values
    pred_lr_eval = sel_lr_dir1.predict(X_eval)
    pred_rf_eval = sel_rf_dir1.predict(X_eval)
    r_lr_eval = [df_s_eval.iloc[i]["oracle_psnr"] - df_s_eval.iloc[i][f"psnr_{pred_lr_eval[i]}"] for i in range(len(X_eval))]
    r_rf_eval = [df_s_eval.iloc[i]["oracle_psnr"] - df_s_eval.iloc[i][f"psnr_{pred_rf_eval[i]}"] for i in range(len(X_eval))]
    d_eval15 = float(np.mean(r_rf_eval) - np.mean(r_lr_eval))

    # Direction 2 (Reverse): Train on LOL-v2 Real Train -> Test on LOL-v1 our485
    print("\nExecuting Reverse Direction: Train on LOL-v2 Real Train -> Evaluate on LOL-v1...")
    real_train_low = "data/LOL-v2/Real_captured/Train/Low"
    real_train_norm = "data/LOL-v2/Real_captured/Train/Normal"
    pairs_real_train = []
    if os.path.isdir(real_train_low):
        for f in sorted(os.listdir(real_train_low)):
            if f.endswith(".png"):
                pairs_real_train.append({
                    "low_path": os.path.join(real_train_low, f),
                    "high_path": os.path.join(real_train_norm, f.replace("low", "normal")),
                    "filename": f,
                })

    df_f_rtrain, df_s_rtrain, _ = process_test_split(pairs_real_train, "LOL-v2 Real Train")

    # Fit on LOL-v2 Real Train
    X_rtrain = df_f_rtrain[feat_cols].values
    # Oracle label on LOL-v2 Real Train
    y_rtrain = []
    for i in range(len(df_s_rtrain)):
        best_m = max(CANONICAL_METHODS, key=lambda m: df_s_rtrain.iloc[i][f"psnr_{m}"])
        y_rtrain.append(best_m)
    y_rtrain = np.array(y_rtrain)

    sel_lr_rev = make_selector("logreg", feature_names=FEATURE_NAMES, random_state=42)
    sel_rf_rev = make_selector("rf", feature_names=FEATURE_NAMES, random_state=42, n_jobs=1)
    sel_lr_rev.fit(X_rtrain, y_rtrain, df_scores=df_s_rtrain)
    sel_rf_rev.fit(X_rtrain, y_rtrain, df_scores=df_s_rtrain)

    # Test reverse on LOL-v1 our485
    pred_lr_v1 = sel_lr_rev.predict(X_v1)
    pred_rf_v1 = sel_rf_rev.predict(X_v1)

    r_lr_rev = [df_scores_v1.iloc[i]["oracle_psnr"] - df_scores_v1.iloc[i][f"psnr_{pred_lr_v1[i]}"] for i in range(len(X_v1))]
    r_rf_rev = [df_scores_v1.iloc[i]["oracle_psnr"] - df_scores_v1.iloc[i][f"psnr_{pred_rf_v1[i]}"] for i in range(len(X_v1))]

    d_reverse = float(np.mean(r_rf_rev) - np.mean(r_lr_rev))

    df_ext = pd.DataFrame([
        {
            "evaluation": "LOL-v2 Real Test (100 pairs)",
            "excluded_near_dupes": excl_real,
            "lr_mean_regret": round(float(np.mean(r_lr_real)), 3),
            "rf_mean_regret": round(float(np.mean(r_rf_real)), 3),
            "d_db (RF - LR)": round(d_real_test, 4),
        },
        {
            "evaluation": "LOL-v1 eval15 (15 pairs)",
            "excluded_near_dupes": excl_eval,
            "lr_mean_regret": round(float(np.mean(r_lr_eval)), 3),
            "rf_mean_regret": round(float(np.mean(r_rf_eval)), 3),
            "d_db (RF - LR)": round(d_eval15, 4),
        },
        {
            "evaluation": "Reverse: Real Train -> LOL-v1 Test",
            "excluded_near_dupes": 0,
            "lr_mean_regret": round(float(np.mean(r_lr_rev)), 3),
            "rf_mean_regret": round(float(np.mean(r_rf_rev)), 3),
            "d_db (RF - LR)": round(d_reverse, 4),
        }
    ])
    df_ext.to_csv(os.path.join(out_dir, "external_validity_e4.csv"), index=False)
    print("\n" + df_ext.to_string(index=False))

    meta = compute_meta("e4_external", "run_e4_external_validity", {
        "d_real_test": d_real_test, "d_reverse": d_reverse
    })
    with open(os.path.join(out_dir, "run_meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)

    return {"df_ext": df_ext, "d_real_test": d_real_test, "d_reverse": d_reverse}


# ==============================================================================
# E5: INTEGRITY CONTROLS
# ==============================================================================

def run_e5_controls(
    data: Dict[str, Any],
    out_dir: str = "results/decision/e5_controls",
    seeds: List[int] = CANONICAL_SEEDS
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("EXPERIMENT E5: SANITY & INTEGRITY CONTROLS")
    print("=" * 70)
    os.makedirs(out_dir, exist_ok=True)

    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    df_scores = data["df_scores"]
    n_samples = len(X)
    oracle_psnr = df_scores[[f"psnr_{m}" for m in CANONICAL_METHODS]].max(axis=1).values
    resampler = GroupResampler(groups, n_bootstraps=2000, seed=42)

    # Control (a): Permute per-image score rows relative to X
    print("\n--- Control (a): Permute Score Rows (Null Signal) ---")
    rng = np.random.RandomState(42)
    perm_idx = rng.permutation(n_samples)
    y_perm = y[perm_idx]
    df_scores_perm = df_scores.iloc[perm_idx].reset_index(drop=True)

    # Control (b): Replace X by Gaussian noise
    print("--- Control (b): Gaussian Noise Input (Null Signal) ---")
    X_noise = rng.normal(0.0, 1.0, size=X.shape)

    # Control (c): Append one-hot oracle label to X
    print("--- Control (c): Label Leakage Oracle Channel (Positive Control) ---")
    one_hot = pd.get_dummies(y)[CANONICAL_METHODS].values
    X_leakage = np.hstack([X, one_hot])

    controls_summary = []

    # Run Control (a)
    ca_lr_r, ca_rf_r, ca_fix_r = [], [], []
    for s in seeds:
        folds = make_folds(X, y_perm, groups, n_splits=5, seed=s)
        for tr, va in folds:
            sel_lr = make_selector("logreg", feature_names=FEATURE_NAMES, random_state=s)
            sel_rf = make_selector("rf", feature_names=FEATURE_NAMES, random_state=s, n_jobs=1)
            sel_fix = make_selector("best_fixed", random_state=s)

            sel_lr.fit(X[tr], y_perm[tr], df_scores=df_scores_perm.iloc[tr])
            sel_rf.fit(X[tr], y_perm[tr], df_scores=df_scores_perm.iloc[tr])
            sel_fix.fit(X[tr], y_perm[tr], df_scores=df_scores_perm.iloc[tr])

            p_lr = sel_lr.predict(X[va])
            p_rf = sel_rf.predict(X[va])
            p_fix = sel_fix.predict(X[va])

            for i, r_i in enumerate(va):
                ca_lr_r.append(oracle_psnr[r_i] - df_scores.iloc[r_i][f"psnr_{p_lr[i]}"])
                ca_rf_r.append(oracle_psnr[r_i] - df_scores.iloc[r_i][f"psnr_{p_rf[i]}"])
                ca_fix_r.append(oracle_psnr[r_i] - df_scores.iloc[r_i][f"psnr_{p_fix[i]}"])

    # Advantage over fixed: fixed_regret - model_regret. Under null, CI must include 0 or be negative!
    lr_adv_a, lr_a_lo, lr_a_hi = paired_ci(np.array(ca_fix_r), np.array(ca_lr_r), resampler)
    rf_adv_a, rf_a_lo, rf_a_hi = paired_ci(np.array(ca_fix_r), np.array(ca_rf_r), resampler)

    ca_pass = (lr_a_lo <= 0.0) and (rf_a_lo <= 0.0)
    print(f"  Control (a) Result: LR advantage over fixed = {lr_adv_a:+.3f} [{lr_a_lo:+.3f}, {lr_a_hi:+.3f}] dB, RF advantage = {rf_adv_a:+.3f} [{rf_a_lo:+.3f}, {rf_a_hi:+.3f}] dB -> {'PASS' if ca_pass else 'FAIL'}")

    # Run Control (b)
    cb_lr_r, cb_rf_r, cb_fix_r = [], [], []
    for s in seeds:
        folds = make_folds(X_noise, y, groups, n_splits=5, seed=s)
        for tr, va in folds:
            sel_lr = make_selector("logreg", feature_names=FEATURE_NAMES, random_state=s)
            sel_rf = make_selector("rf", feature_names=FEATURE_NAMES, random_state=s, n_jobs=1)
            sel_fix = make_selector("best_fixed", random_state=s)

            sel_lr.fit(X_noise[tr], y[tr], df_scores=df_scores.iloc[tr])
            sel_rf.fit(X_noise[tr], y[tr], df_scores=df_scores.iloc[tr])
            sel_fix.fit(X_noise[tr], y[tr], df_scores=df_scores.iloc[tr])

            p_lr = sel_lr.predict(X_noise[va])
            p_rf = sel_rf.predict(X_noise[va])
            p_fix = sel_fix.predict(X_noise[va])

            for i, r_i in enumerate(va):
                cb_lr_r.append(oracle_psnr[r_i] - df_scores.iloc[r_i][f"psnr_{p_lr[i]}"])
                cb_rf_r.append(oracle_psnr[r_i] - df_scores.iloc[r_i][f"psnr_{p_rf[i]}"])
                cb_fix_r.append(oracle_psnr[r_i] - df_scores.iloc[r_i][f"psnr_{p_fix[i]}"])

    lr_adv_b, lr_b_lo, lr_b_hi = paired_ci(np.array(cb_fix_r), np.array(cb_lr_r), resampler)
    rf_adv_b, rf_b_lo, rf_b_hi = paired_ci(np.array(cb_fix_r), np.array(cb_rf_r), resampler)
    cb_pass = (lr_b_lo <= 0.0) and (rf_b_lo <= 0.0)
    print(f"  Control (b) Result: LR advantage over fixed = {lr_adv_b:+.3f} [{lr_b_lo:+.3f}, {lr_b_hi:+.3f}] dB, RF advantage = {rf_adv_b:+.3f} [{rf_b_lo:+.3f}, {rf_b_hi:+.3f}] dB -> {'PASS' if cb_pass else 'FAIL'}")

    # Run Control (c)
    cc_lr_acc, cc_rf_acc = [], []
    for s in seeds:
        folds = make_folds(X_leakage, y, groups, n_splits=5, seed=s)
        for tr, va in folds:
            sel_lr = make_selector("logreg", random_state=s)
            sel_rf = make_selector("rf", random_state=s, n_jobs=1)
            sel_lr.fit(X_leakage[tr], y[tr], df_scores=df_scores.iloc[tr])
            sel_rf.fit(X_leakage[tr], y[tr], df_scores=df_scores.iloc[tr])
            p_lr = sel_lr.predict(X_leakage[va])
            p_rf = sel_rf.predict(X_leakage[va])
            cc_lr_acc.append(np.mean(p_lr == y[va]))
            cc_rf_acc.append(np.mean(p_rf == y[va]))

    lr_leak_acc = float(np.mean(cc_lr_acc)) * 100.0
    rf_leak_acc = float(np.mean(cc_rf_acc)) * 100.0
    cc_pass = (lr_leak_acc >= 95.0) and (rf_leak_acc >= 95.0)
    print(f"  Control (c) Result: LR accuracy with label = {lr_leak_acc:.1f}%, RF accuracy = {rf_leak_acc:.1f}% -> {'PASS' if cc_pass else 'FAIL'}")

    all_controls_pass = ca_pass and cb_pass and cc_pass

    df_controls = pd.DataFrame([
        {"control": "Control (a): Score Permutation", "criterion": "Neither beats Best-Fixed (CI <= 0)", "lr_metric": f"{lr_adv_a:+.3f} [{lr_a_lo:+.3f}, {lr_a_hi:+.3f}]", "rf_metric": f"{rf_adv_a:+.3f} [{rf_a_lo:+.3f}, {rf_a_hi:+.3f}]", "status": "PASS" if ca_pass else "FAIL"},
        {"control": "Control (b): Gaussian Noise Input", "criterion": "Neither beats Best-Fixed (CI <= 0)", "lr_metric": f"{lr_adv_b:+.3f} [{lr_b_lo:+.3f}, {lr_b_hi:+.3f}]", "rf_metric": f"{rf_adv_b:+.3f} [{rf_b_lo:+.3f}, {rf_b_hi:+.3f}]", "status": "PASS" if cb_pass else "FAIL"},
        {"control": "Control (c): Label Channel", "criterion": "Both reach >= 95.0% accuracy", "lr_metric": f"{lr_leak_acc:.1f}%", "rf_metric": f"{rf_leak_acc:.1f}%", "status": "PASS" if cc_pass else "FAIL"},
    ])
    df_controls.to_csv(os.path.join(out_dir, "controls_summary.csv"), index=False)
    print("\n--- Controls Summary Table ---")
    print(df_controls.to_string(index=False))

    meta = compute_meta("e5_controls", "run_e5_controls", {"all_controls_pass": all_controls_pass})
    with open(os.path.join(out_dir, "run_meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)

    return {"df_controls": df_controls, "all_pass": all_controls_pass}


# ==============================================================================
# MAIN EXPERIMENT PIPELINE (E0 -> E5) & DECISION RULE EVALUATION
# ==============================================================================

def run_pipeline_cv(run_name: str = "explore_default", no_assets: bool = False) -> None:
    print("\n" + "#" * 75)
    print("PROJECT TRIATHLON · MASTER DECISION EXPERIMENT PROTOCOL (E0–E5)")
    print("#" * 75)

    data = prepare(dataset="lol_v1", split="our485", hamming_threshold=10)

    # 1. E0: Default Arm Reproduction
    res_e0 = run_e0_reproduce(data, out_dir=f"results/decision/{run_name}")

    # 2. E1: Nested Fair Tuning
    res_e1 = run_e1_nested_tuning(data, out_dir="results/decision/e1_tuning", budget=30)

    # 3. E2: Leakage & Threshold Sensitivity
    res_e2 = run_e2_leakage(data, out_dir="results/decision/e2_leakage")

    # 4. E3: Learning Curves
    res_e3 = run_e3_learning_curves(data, out_dir="results/decision/e3_learning_curve")

    # 5. E4: External Validity
    res_e4 = run_e4_external_validity(data, out_dir="results/decision/e4_external")

    # 6. E5: Controls
    res_e5 = run_e5_controls(data, out_dir="results/decision/e5_controls")

    # --------------------------------------------------------------------------
    # EVALUATE PRE-COMMITTED DECISION RULE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("EVALUATING PRE-COMMITTED DECISION RULE (docs/DECISION_RULE.md)")
    print("=" * 75)

    # d = mean regret(RF) - mean regret(LR) in dB, TUNED arms, LOL-v1 grouped CV
    d_point = res_e1["d"]
    d_lo, d_hi = res_e1["d_ci"]
    delta = 0.10

    d_real_test = res_e4["d_real_test"]
    d_reverse = res_e4["d_reverse"]
    thr_18_d_hi = res_e2["thr_18_d_hi"]
    all_controls_pass = res_e5["all_pass"]

    cond_1 = d_hi < -delta
    cond_2 = (d_real_test < 0.0) and (d_reverse < 0.0)
    cond_3 = thr_18_d_hi < 0.0 if thr_18_d_hi is not None else False
    cond_4 = all_controls_pass

    print(f"Condition 1 (d_hi < -delta): {d_hi:.4f} < {-delta:.2f} -> {cond_1}")
    print(f"Condition 2 (d < 0 on BOTH external evals): Real Test = {d_real_test:+.4f}, Reverse = {d_reverse:+.4f} -> {cond_2}")
    print(f"Condition 3 (d_hi < 0 at Hamming threshold 18): thr_18_d_hi = {thr_18_d_hi:+.4f} -> {cond_3}")
    print(f"Condition 4 (All E5 controls pass): {cond_4}")

    if cond_1 and cond_2 and cond_3 and cond_4:
        verdict = "RF_WINS"
        winner = "Random Forest"
    elif d_hi < 0.0 and not (cond_2 and cond_3 and cond_4):
        verdict = "RF_CV_ONLY"
        winner = "Logistic Regression (RF flagged for probable overfitting / failure on external validity)"
    elif d_lo > 0.0:
        verdict = "LR_WINS"
        winner = "Logistic Regression"
    else:
        verdict = "TIE"
        winner = "Logistic Regression (Burden of proof on complex model; LR selected)"

    print("-" * 75)
    print(f"VERDICT LABEL : {verdict}")
    print(f"DEPLOYED MODEL: {winner}")
    print("-" * 75)

    # Record decision in results/decision/verdict.json
    verdict_record = {
        "verdict": verdict,
        "deployed_model": winner,
        "d_mean_db": round(d_point, 4),
        "d_ci_lo": round(d_lo, 4),
        "d_ci_hi": round(d_hi, 4),
        "d_real_test": round(d_real_test, 4),
        "d_reverse": round(d_reverse, 4),
        "thr_18_d_hi": round(thr_18_d_hi, 4) if thr_18_d_hi else None,
        "all_controls_pass": all_controls_pass,
        "conditions": {
            "cond_1_d_hi_less_than_minus_delta": bool(cond_1),
            "cond_2_external_both_negative": bool(cond_2),
            "cond_3_threshold_18_d_hi_negative": bool(cond_3),
            "cond_4_controls_pass": bool(cond_4),
        }
    }
    with open("results/decision/verdict.json", "w") as fp:
        json.dump(verdict_record, fp, indent=2)

    if not no_assets:
        from src.paper_assets import generate_all_paper_assets
        generate_all_paper_assets(run_name=run_name)


if __name__ == "__main__":
    run_pipeline_cv()
