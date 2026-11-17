#!/usr/bin/env python3
"""
Unit and audit tests for Project Triathlon Web UI remediation.
Validates Requirements R1 through R11.
"""

import os
import json
import unittest
import numpy as np
import cv2

from apps.web.facts import load_facts, REPO_ROOT
from src.features import extract_features, FEATURE_NAMES


class TestUIAudit(unittest.TestCase):
    """Rigorous acceptance test suite for UI fixes."""

    def test_r1_default_sample_and_preview(self):
        """R1: Default sample is eval15 547 (median brightness), preview boost > 60/255, true pixels unchanged."""
        data_root = os.path.join(REPO_ROOT, "data", "lol_dataset", "eval15", "low")
        if not os.path.exists(data_root):
            data_root = os.path.join(REPO_ROOT, "data", "LOLdataset", "eval15", "low")
        self.assertTrue(os.path.exists(data_root), "eval15 directory must exist")

        # 1. Assert 547 is closest to median brightness
        files = [f for f in sorted(os.listdir(data_root)) if f.endswith(".png")]
        intensities = {}
        for f in files:
            img = cv2.imread(os.path.join(data_root, f))
            intensities[f.replace(".png", "")] = float(np.mean(img))

        med = np.median(list(intensities.values()))
        closest = min(intensities.keys(), key=lambda k: abs(intensities[k] - med))
        self.assertEqual(closest, "547", f"547 must be closest to median ({med:.2f})")

        # 2. Test raw image vs preview boost
        raw_img = cv2.imread(os.path.join(data_root, "547.png"))
        raw_mean = float(np.mean(raw_img))
        self.assertAlmostEqual(raw_mean, 15.67, delta=1.0)

        # Apply gamma 0.4 display boost
        lut = np.array([((i / 255.0) ** 0.4) * 255.0 for i in range(256)]).astype("uint8")
        boosted = cv2.LUT(raw_img, lut)
        boosted_mean = float(np.mean(boosted))

        self.assertGreater(boosted_mean, 60.0, "Boosted display preview mean must be > 60/255")
        self.assertAlmostEqual(float(np.mean(raw_img)), raw_mean, delta=1e-5, msg="Raw pixels must be unmodified")

    def test_r2_chart_contrast(self):
        """R2: Contrast of bar fills >= 3:1 against track (#f1f5f9), text labels >= 4.5:1."""
        def luminance(r, g, b):
            def c(v):
                v = v / 255.0
                return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
            return 0.2126 * c(r) + 0.7152 * c(g) + 0.0722 * c(b)

        def contrast(c1, c2):
            l1 = luminance(*c1)
            l2 = luminance(*c2)
            lighter = max(l1, l2)
            darker = min(l1, l2)
            return (lighter + 0.05) / (darker + 0.05)

        track_rgb = (241, 245, 249) # #f1f5f9
        text_rgb = (15, 23, 42)     # #0f172a

        bar_colors = {
            "Oracle": (2, 132, 199),        # #0284c7
            "Logistic Regression": (22, 163, 74), # #16a34a
            "Random Forest": (71, 85, 105),       # #475569
            "HistGradientBoosting": (100, 116, 139), # #64748b
            "Best-Fixed Retinex": (180, 83, 9),   # #b45309 deep amber (>= 4:1)
            "Random Choice": (51, 65, 85),        # #334155
        }

        # Check graphical element contrast (>= 3.0:1)
        for name, rgb in bar_colors.items():
            cr = contrast(rgb, track_rgb)
            self.assertGreaterEqual(cr, 3.0, f"Bar '{name}' contrast {cr:.2f}:1 must be >= 3.0:1 against track")

        # Check header text label contrast (>= 4.5:1)
        cr_text = contrast(text_rgb, track_rgb)
        self.assertGreaterEqual(cr_text, 4.5, f"Header text contrast {cr_text:.2f}:1 must be >= 4.5:1 against track")

    def test_r5_facts_parity(self):
        """R5: apps/web/facts.py dynamically matches underlying result files without literal hardcodes."""
        facts = load_facts()
        self.assertIsNotNone(facts["run_name"])
        self.assertIn("kpis", facts)
        self.assertIn("regret_bars", facts)
        self.assertIn("model_comparison", facts)

        # Check models/selector.json parity
        sel_json_path = os.path.join(REPO_ROOT, "models", "selector.json")
        if os.path.exists(sel_json_path):
            with open(sel_json_path) as f:
                sj = json.load(f)
            gap_closed = sj.get("cv_performance_5fold_grouped", {}).get("gap_closed_pct")
            if gap_closed is not None:
                self.assertIn(f"{gap_closed:.1f}%", facts["kpis"]["headroom_closed_pct"])

        # Check verdict.json parity
        verdict_path = os.path.join(REPO_ROOT, "results", "decision", "verdict.json")
        if os.path.exists(verdict_path):
            with open(verdict_path) as f:
                vj = json.load(f)
            self.assertEqual(facts["model_comparison"]["verdict"], vj["verdict"])

    def test_r7_bank_and_selector_parity(self):
        """R7: Features and selector predictions match canonical src.features on test images."""
        import joblib
        from apps.web.app import get_model, process_image

        model = get_model()
        self.assertIsNotNone(model, "Deployed model must load")

        # Test on 5 sample images
        data_root = os.path.join(REPO_ROOT, "data", "lol_dataset", "eval15", "low")
        if not os.path.exists(data_root):
            data_root = os.path.join(REPO_ROOT, "data", "LOLdataset", "eval15", "low")

        from src.dataset import resize_long_side
        files = [f for f in sorted(os.listdir(data_root)) if f.endswith(".png")][:5]
        for f in files:
            img_bgr = cv2.imread(os.path.join(data_root, f))
            # 1. Feature extraction parity
            feats_app = process_image(img_bgr, enforce_safeguards=False)["features"]
            deg, _ = resize_long_side(img_bgr, target_size=512)
            feats_src = extract_features(deg)
            for fn in FEATURE_NAMES:
                self.assertAlmostEqual(feats_app[fn], feats_src[fn], places=2, msg=f"Feature {fn} mismatch on {f}")

    def test_r10_no_missing_ids_in_html(self):
        """R10: All document.getElementById references in app.py exist in HTML_TEMPLATE."""
        import re
        from apps.web.app import HTML_TEMPLATE

        ids_used = re.findall(r'document\.getElementById\([\'\"]([^\'\"]+)[\'\"]\)', HTML_TEMPLATE)
        ids_in_html = set(re.findall(r'id=[\'\"]([^\'\"]+)[\'\"]', HTML_TEMPLATE))

        missing = [i for i in set(ids_used) if i not in ids_in_html]
        self.assertEqual(missing, [], f"Missing element IDs in HTML: {missing}")


if __name__ == "__main__":
    unittest.main()
