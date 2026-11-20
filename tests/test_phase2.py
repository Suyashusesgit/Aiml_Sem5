"""
Unit Tests for Phase 2 Modules:
  - Milestone 2.1: Multi-Degradation Optical Taxonomy & Hierarchical Cascades
  - Milestone 2.2: Predict-Then-Optimize (PTO) Continuous Selection
  - Milestone 2.3: Adaptive 2-Stage Micro-Pipelines
  - Milestone 2.4: Latent Spectral & Hybrid Feature Representation
"""

import unittest
import numpy as np
import cv2

from src.multi_degradation import (
    extract_extended_features,
    hierarchical_selector,
    synthesize_degradations,
    high_boost_sharpen,
    dcp_dehaze,
)
from src.predict_then_optimize import PredictThenOptimizeSelector
from src.micro_pipeline import AdaptiveMicroPipeline, bilateral_denoise
from src.latent_features import (
    extract_latent_spectral_features,
    extract_hybrid_features,
    LATENT_FEATURE_NAMES,
    HYBRID_FEATURE_NAMES,
)


class TestPhase2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create deterministic synthetic test image (400 x 600 x 3)
        np.random.seed(42)
        cls.img_clean = np.full((120, 160, 3), 128, dtype=np.uint8)
        # Add high-contrast text/edge pattern
        cv2.putText(cls.img_clean, "TRIATHLON", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.circle(cls.img_clean, (120, 60), 20, (30, 200, 30), -1)

        # Create dark image
        cls.img_dark = (cls.img_clean * 0.15).astype(np.uint8)

    def test_extract_extended_features(self):
        """Milestone 2.1: Verify 10D physical feature extraction."""
        feats = extract_extended_features(self.img_clean)
        self.assertIn("blur_metric", feats)
        self.assertIn("haze_metric", feats)
        self.assertIn("laplacian_var", feats)
        self.assertIn("brightness", feats)
        self.assertIn("contrast", feats)

        self.assertGreaterEqual(feats["blur_metric"], 0.0)
        self.assertLessEqual(feats["blur_metric"], 1.0)
        self.assertGreaterEqual(feats["haze_metric"], 0.0)
        self.assertLessEqual(feats["haze_metric"], 1.0)

    def test_hierarchical_selector_routing(self):
        """Milestone 2.1: Verify hierarchical regime diagnosis and routing."""
        # Low-light sample
        feats_dark = extract_extended_features(self.img_dark)
        regime_dark, method_dark, reason_dark = hierarchical_selector(feats_dark)
        self.assertEqual(regime_dark, "Low-Light")
        self.assertIn(method_dark, ["gamma", "clahe", "hist_eq", "retinex", "bilat_unsharp"])

        # Hazy sample
        hazy_feats = {
            "brightness": 95.0,
            "contrast": 18.0,
            "entropy": 6.8,
            "dark_channel": 45.0,
            "noise": 1.2,
            "laplacian_var": 80.0,
            "color_cast": 0.02,
            "colourfulness": 15.0,
            "blur_metric": 0.25,
            "haze_metric": 0.45,
        }
        regime_haze, method_haze, _ = hierarchical_selector(hazy_feats)
        self.assertEqual(regime_haze, "Atmospheric Haze")
        self.assertEqual(method_haze, "dcp_dehaze")

    def test_predict_then_optimize(self):
        """Milestone 2.2: Verify continuous PTO regressors and inference."""
        X_dummy = np.random.randn(20, 10)
        y_dummy = {
            m: np.random.uniform(10.0, 22.0, size=20)
            for m in ["identity", "gamma", "clahe", "hist_eq", "retinex", "bilat_unsharp"]
        }

        pto = PredictThenOptimizeSelector(model_type="ridge")
        pto.fit(X_dummy, y_dummy)

        selected_methods, pred_matrix = pto.select(X_dummy[:3])
        self.assertEqual(len(selected_methods), 3)
        self.assertEqual(pred_matrix.shape, (3, 6))

        # Test single feature dict prediction
        feats = {
            "brightness": 12.0,
            "contrast": 8.0,
            "entropy": 4.5,
            "dark_channel": 4.0,
            "noise": 2.5,
            "laplacian_var": 50.0,
            "color_cast": 0.01,
            "colourfulness": 8.0,
        }
        best_m, exp_psnr = pto.predict_best_method(feats)
        self.assertIn(best_m, pto.methods)
        self.assertIsInstance(exp_psnr, float)

    def test_adaptive_micro_pipeline(self):
        """Milestone 2.3: Verify 2-stage dynamic micro-pipeline execution."""
        pipeline = AdaptiveMicroPipeline()
        res = pipeline.process(self.img_dark)

        self.assertIn("stage1_operator", res)
        self.assertIn("stage2_operator", res)
        self.assertIn("pipeline_name", res)
        self.assertIn("latency_ms", res)
        self.assertEqual(res["enhanced_image"].shape, self.img_dark.shape)
        self.assertLess(res["latency_ms"]["total"], 500.0)  # Safe budget for test runner

    def test_latent_spectral_and_hybrid_features(self):
        """Milestone 2.4: Verify latent spectral 16D and hybrid 26D feature extraction."""
        latent = extract_latent_spectral_features(self.img_clean)
        self.assertEqual(len(latent), 16)
        for k in LATENT_FEATURE_NAMES:
            self.assertIn(k, latent)

        hybrid = extract_hybrid_features(self.img_clean)
        self.assertGreaterEqual(len(hybrid), 26)
        for k in HYBRID_FEATURE_NAMES:
            self.assertIn(k, hybrid)


if __name__ == "__main__":
    unittest.main()
