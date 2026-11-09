"""
Unit tests for LOL-v2 dataset ingestion, synthesis, and evaluation contract.
"""

import os
import unittest
import numpy as np
import cv2

from src.dataset_v2 import (
    load_and_preprocess_pair_v2,
    get_lol_v2_manifest,
    synthesize_lol_v2_pair,
    resolve_lol_v2_dir
)
from src.labels import score_pair
from src.features import extract_features


class TestLOLv2Pipeline(unittest.TestCase):

    def setUp(self):
        # Create a synthetic test pair in memory
        self.clean_dummy = np.full((600, 800, 3), 180, dtype=np.uint8)
        # Add gradient to make it non-trivial
        for c in range(3):
            self.clean_dummy[:, :, c] = np.linspace(50, 220, 800, dtype=np.uint8)
        self.deg_dummy = synthesize_lol_v2_pair(self.clean_dummy)

    def test_synthesize_pair(self):
        """Verifies synthetic degradation darkens and adds noise."""
        self.assertEqual(self.deg_dummy.shape, self.clean_dummy.shape)
        self.assertEqual(self.deg_dummy.dtype, np.uint8)
        # Degraded should have lower mean intensity
        self.assertLess(float(self.deg_dummy.mean()), float(self.clean_dummy.mean()))

    def test_load_and_preprocess_contract(self):
        """Verifies 512px long-side contract on LOL-v2."""
        manifest = get_lol_v2_manifest("synthetic")
        if manifest:
            sample = manifest[0]
            clean, deg, scale = load_and_preprocess_pair_v2(sample["low_path"], sample["high_path"], target_long_side=512)
            self.assertLessEqual(max(clean.shape[:2]), 512)
            self.assertLessEqual(max(deg.shape[:2]), 512)
            self.assertEqual(clean.shape, deg.shape)
            self.assertEqual(clean.dtype, np.uint8)
            self.assertEqual(deg.dtype, np.uint8)

    def test_score_pair_on_v2(self):
        """Verifies score_pair works correctly on v2 pairs."""
        res = score_pair(self.clean_dummy, self.deg_dummy)
        self.assertIn("label", res)
        self.assertIn("oracle_psnr", res)
        self.assertIn("oracle_ssim", res)
        self.assertGreater(res["oracle_psnr"], 0.0)

    def test_feature_extraction_on_v2(self):
        """Verifies physical features extract cleanly on v2 inputs."""
        feats = extract_features(self.deg_dummy, include_engineered=True)
        self.assertIn("brightness", feats)
        self.assertIn("contrast", feats)
        self.assertIn("noise", feats)
        self.assertIn("dark_channel", feats)
        self.assertGreaterEqual(feats["brightness"], 0.0)


if __name__ == "__main__":
    unittest.main()

