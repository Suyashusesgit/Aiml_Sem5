"""
Unit tests for Phase 0 Deferred Scope Milestones
Tests:
- Predict-Then-Optimize (PTO)
- Downstream ORB Inlier Matching
- Latency-Aware Selection
"""

import unittest
import numpy as np

from src.predict_then_optimize import PredictThenOptimizeSelector, CANONICAL_METHODS
from src.downstream_matching import compute_orb_inlier_matches, evaluate_enhancement_matching_gain
from src.latency_aware_selection import LatencyAwareSelector, METHOD_LATENCIES_MS


class TestDeferredMilestones(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.img1 = np.random.randint(15, 70, (120, 160, 3), dtype=np.uint8)
        self.img2 = np.random.randint(20, 120, (120, 160, 3), dtype=np.uint8)

    def test_pto_selector_fit_predict(self):
        pto = PredictThenOptimizeSelector(model_type="ridge")
        X = np.random.randn(30, 10)
        y_dict = {m: np.random.uniform(10.0, 25.0, 30) for m in CANONICAL_METHODS}

        pto.fit(X, y_dict)
        preds = pto.predict_psnr(X[:5])
        self.assertEqual(len(preds), len(CANONICAL_METHODS))
        for m in CANONICAL_METHODS:
            self.assertEqual(len(preds[m]), 5)

        chosen, matrix = pto.select(X[:5])
        self.assertEqual(len(chosen), 5)
        self.assertEqual(matrix.shape, (5, len(CANONICAL_METHODS)))

    def test_orb_inlier_matches(self):
        res = compute_orb_inlier_matches(self.img1, self.img2)
        self.assertIn("inlier_count", res)
        self.assertIn("inlier_ratio_pct", res)
        self.assertIn("canvas_bgr", res)
        self.assertIsInstance(res["canvas_bgr"], np.ndarray)

        gain_res = evaluate_enhancement_matching_gain(self.img1, self.img2)
        self.assertIn("inlier_gain", gain_res)
        self.assertIn("inlier_gain_pct", gain_res)

    def test_latency_aware_selector(self):
        sel_fast = LatencyAwareSelector(latency_budget_ms=1.0)
        admissible_fast = sel_fast.get_admissible_methods()
        self.assertIn("gamma", admissible_fast)
        self.assertNotIn("bilat_unsharp", admissible_fast)

        probs = {"gamma": 0.4, "bilat_unsharp": 0.6}
        chosen = sel_fast.select(probs)
        self.assertEqual(chosen, "gamma")  # Should pick gamma because bilat is pruned!


if __name__ == "__main__":
    unittest.main()
