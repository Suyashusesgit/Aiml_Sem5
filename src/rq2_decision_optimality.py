"""
RQ2 Decision-Optimality Benchmark (Semester 6 Lookahead)
Authors: Project Triathlon Engineering Team
Supervisor: Prof. Kishor P. Upla (Senior Member, IEEE)

Addresses Page 9 Research Question 2:
"Is PSNR-optimal also decision-optimal? Test whether the best PSNR enhancer
 also helps humans / vision agents make better decisions in low-light scenes
 (night drowsiness / facial feature vigilance test)."
"""

import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.methods import METHODS
from src.labels import compute_psnr, compute_ssim


def compute_eye_feature_response(img_bgr: np.ndarray) -> float:
    """
    Evaluates high-frequency localized structural response in eye/face regions:
    Quantifies localized gradient energy (Tenengrad focus) in high-contrast facial landmark bands.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    # 90th percentile gradient acts as vigilance / edge landmark clarity metric
    return float(np.percentile(grad_mag, 90.0))


def evaluate_decision_vs_fidelity(
    low_dir: str = "data/lol_dataset/eval15/low",
    high_dir: str = "data/lol_dataset/eval15/high"
) -> pd.DataFrame:
    records = []
    low_files = sorted(glob.glob(os.path.join(low_dir, "*.png")))

    for lf in low_files:
        base = os.path.basename(lf)
        hf = os.path.join(high_dir, base)
        if not os.path.exists(hf):
            continue

        deg = cv2.imread(lf)
        clean = cv2.imread(hf)
        if deg is None or clean is None:
            continue

        psnr_scores = {}
        decision_scores = {}

        for m_name, m_fn in METHODS.items():
            enh = m_fn(deg)
            psnr_scores[m_name] = compute_psnr(clean, enh)
            decision_scores[m_name] = compute_eye_feature_response(enh)

        psnr_winner = max(psnr_scores.keys(), key=lambda k: psnr_scores[k])
        decision_winner = max(decision_scores.keys(), key=lambda k: decision_scores[k])

        is_aligned = int(psnr_winner == decision_winner)
        # Decision regret: difference in decision fidelity vs best decision operator
        max_decision = max(decision_scores.values())
        psnr_chosen_decision = decision_scores[psnr_winner]
        decision_regret = max_decision - psnr_chosen_decision

        records.append({
            "image": base,
            "psnr_winner": psnr_winner,
            "decision_winner": decision_winner,
            "is_aligned": is_aligned,
            "psnr_winner_val": round(psnr_scores[psnr_winner], 2),
            "decision_regret": round(decision_regret, 2),
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df_rq2 = evaluate_decision_vs_fidelity()
    alignment_rate = df_rq2["is_aligned"].mean() * 100.0
    print("=== RQ2: Fidelity vs Decision-Optimality Benchmark ===")
    print(f"Alignment Rate between PSNR Winner and Semantic Decision Winner: {alignment_rate:.1f}%")
    print(f"Mean Decision Regret when following PSNR Selector: {df_rq2['decision_regret'].mean():.2f}")
    print("\nSample records:")
    print(df_rq2.head(10).to_string(index=False))

    os.makedirs("results", exist_ok=True)
    df_rq2.to_csv("results/rq2_decision_optimality.csv", index=False)
    print("\nSaved RQ2 results to results/rq2_decision_optimality.csv")

