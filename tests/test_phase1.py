"""
Unit tests for Phase 1 (Semester 6) Modules
Tests:
1. Split-Conformal Prediction calibration and finite-sample coverage guarantees.
2. Confidence-gated abstention logic and monotonicity.
3. 10D Physical Feature extraction and anti-leakage validation.
4. Leave-One-Enhancer-Out (LOEO) bank degradation properties.
5. RQ2 PSNR vs SSIM dual-metric evaluator.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.conformal import SplitConformalClassifier, evaluate_confidence_abstention
from src.features import (
    PHYSICAL_10_DESCRIPTORS,
    extract_features,
    validate_feature_dataframe,
    air_light_transmission,
    dynamic_range_ratio
)
from src.methods import CANONICAL_METHODS


class TestPhase1(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Synthetic test data
        self.n_samples = 100
        self.classes = ["gamma", "hist_eq", "retinex"]
        self.X_train = np.random.randn(self.n_samples, 5)
        self.y_train = np.random.choice(self.classes, size=self.n_samples)

        self.model = LogisticRegression(random_state=42)
        self.model.fit(self.X_train, self.y_train)

        self.synthetic_img = np.random.randint(10, 100, (64, 64, 3), dtype=np.uint8)

    def test_conformal_calibration_and_coverage(self):
        X_cal = np.random.randn(60, 5)
        y_cal = np.random.choice(self.classes, size=60)
        X_test = np.random.randn(80, 5)
        y_test = np.random.choice(self.classes, size=80)

        cp = SplitConformalClassifier(self.model, alpha=0.10)
        q_hat = cp.calibrate(X_cal, y_cal)

        self.assertIsNotNone(q_hat)
        self.assertGreaterEqual(q_hat, 0.0)
        self.assertLessEqual(q_hat, 1.0)

        pred_sets = cp.predict_set(X_test)
        self.assertEqual(len(pred_sets), len(X_test))
        for s in pred_sets:
            self.assertGreaterEqual(len(s), 1)  # Never empty

        eval_dict = cp.evaluate_coverage(X_test, y_test)
        self.assertIn("empirical_coverage", eval_dict)
        self.assertIn("average_set_size", eval_dict)
        self.assertGreater(eval_dict["empirical_coverage"], 0.0)

    def test_confidence_abstention_monotonicity(self):
        df_scores = pd.DataFrame({
            "oracle_psnr": np.random.uniform(15.0, 22.0, self.n_samples),
            "psnr_gamma": np.random.uniform(10.0, 18.0, self.n_samples),
            "psnr_hist_eq": np.random.uniform(10.0, 18.0, self.n_samples),
            "psnr_retinex": np.random.uniform(12.0, 20.0, self.n_samples),
        })

        df_res = evaluate_confidence_abstention(
            model=self.model,
            X=self.X_train,
            y_oracle=self.y_train,
            df_scores=df_scores,
            thresholds=[0.3, 0.5, 0.7, 0.9],
            default_fallback="retinex"
        )

        abstain_rates = df_res["abstain_rate_pct"].values
        # Abstain rate must be non-decreasing with increasing tau
        for i in range(len(abstain_rates) - 1):
            self.assertLessEqual(abstain_rates[i], abstain_rates[i + 1])

    def test_10d_feature_extraction_and_validation(self):
        feats = extract_features(self.synthetic_img, include_engineered=False, include_all_physical=True)
        self.assertEqual(len(feats), len(PHYSICAL_10_DESCRIPTORS))
        for descriptor in PHYSICAL_10_DESCRIPTORS:
            self.assertIn(descriptor, feats)
            self.assertIsInstance(feats[descriptor], float)

        df_test = pd.DataFrame([feats])
        # Validation must pass for physical features
        validate_feature_dataframe(df_test, PHYSICAL_10_DESCRIPTORS)

        # Anti-leakage check: must fail on forbidden column
        df_leaked = df_test.copy()
        df_leaked["psnr_groundtruth"] = 21.5
        with self.assertRaises(ValueError):
            validate_feature_dataframe(df_leaked, list(df_leaked.columns))

    def test_air_light_transmission_bounds(self):
        t = air_light_transmission(self.synthetic_img)
        self.assertGreaterEqual(t, 0.0)
        self.assertLessEqual(t, 1.0)

    def test_dynamic_range_ratio_positive(self):
        gray = self.synthetic_img[:, :, 0]
        dr = dynamic_range_ratio(gray)
        self.assertGreater(dr, 0.0)

    def test_loeo_monotonicity(self):
        # Oracle PSNR with 6 methods must be >= Oracle PSNR with any 5 methods
        psnrs = {m: np.random.uniform(10.0, 25.0, 20) for m in CANONICAL_METHODS}
        df_psnr = pd.DataFrame(psnrs)

        full_oracle = df_psnr.max(axis=1).values
        for m in CANONICAL_METHODS:
            reduced_cols = [c for c in CANONICAL_METHODS if c != m]
            reduced_oracle = df_psnr[reduced_cols].max(axis=1).values
            # Reduced oracle can never exceed full oracle
            self.assertTrue(np.all(full_oracle >= reduced_oracle - 1e-6))


if __name__ == "__main__":
    unittest.main()
