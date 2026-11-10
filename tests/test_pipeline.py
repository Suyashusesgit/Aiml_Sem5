"""
Unit and Integration Tests for Project Triathlon Pipeline
Tests methods bank, feature extraction, scoring metrics, and model inference.
"""

import os
import unittest
import numpy as np
import cv2
import joblib

from src.methods import METHODS
from src.features import extract_features, FEATURE_NAMES
from src.labels import compute_mse, compute_psnr, compute_ssim, score_pair


class TestEnhancementMethods(unittest.TestCase):
    def setUp(self):
        # Create synthetic dark 100x100 BGR image
        np.random.seed(42)
        self.sample_img = np.random.randint(10, 60, (100, 100, 3), dtype=np.uint8)

    def test_all_methods_exist(self):
        expected_methods = {"identity", "gamma", "clahe", "hist_eq", "retinex", "bilat_unsharp"}
        self.assertEqual(set(METHODS.keys()), expected_methods)

    def test_method_output_shapes_and_types(self):
        for name, fn in METHODS.items():
            enhanced = fn(self.sample_img)
            self.assertIsInstance(enhanced, np.ndarray, f"Failed for {name}")
            self.assertEqual(enhanced.shape, self.sample_img.shape, f"Shape mismatch for {name}")
            self.assertEqual(enhanced.dtype, np.uint8, f"Data type mismatch for {name}")
            self.assertTrue(0 <= enhanced.min() and enhanced.max() <= 255, f"Values out of bounds for {name}")


class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.sample_img = np.random.randint(20, 80, (120, 120, 3), dtype=np.uint8)

    def test_feature_names_and_finite_values(self):
        feats = extract_features(self.sample_img, include_engineered=True)
        for name in FEATURE_NAMES:
            self.assertIn(name, feats)
            val = feats[name]
            self.assertFalse(np.isnan(val), f"NaN feature value: {name}")
            self.assertFalse(np.isinf(val), f"Inf feature value: {name}")
            self.assertIsInstance(val, float)


class TestMetricsAndScoring(unittest.TestCase):
    def setUp(self):
        self.img1 = np.ones((50, 50, 3), dtype=np.uint8) * 100
        self.img2 = np.ones((50, 50, 3), dtype=np.uint8) * 100
        self.img_diff = np.ones((50, 50, 3), dtype=np.uint8) * 120

    def test_identical_psnr_and_ssim(self):
        self.assertEqual(compute_mse(self.img1, self.img2), 0.0)
        self.assertEqual(compute_psnr(self.img1, self.img2), 100.0)
        self.assertAlmostEqual(compute_ssim(self.img1, self.img2), 1.0, places=3)

    def test_different_images(self):
        psnr_val = compute_psnr(self.img1, self.img_diff)
        ssim_val = compute_ssim(self.img1, self.img_diff)
        self.assertTrue(psnr_val < 100.0)
        self.assertTrue(0.0 <= ssim_val <= 1.0)


class TestModelInference(unittest.TestCase):
    def test_saved_model_prediction(self):
        model_path = "models/rf_selector.joblib"
        if not os.path.exists(model_path):
            self.skipTest(f"Model file {model_path} does not exist.")

        model = joblib.load(model_path)
        sample_img = np.random.randint(15, 70, (100, 100, 3), dtype=np.uint8)
        feats = extract_features(sample_img)
        X = np.array([[feats[k] for k in FEATURE_NAMES]])

        pred = model.predict(X)[0]
        self.assertIn(pred, METHODS.keys())

    def test_onnx_model_inference(self):
        onnx_path = "models/rf_selector.onnx"
        if not os.path.exists(onnx_path):
            self.skipTest(f"ONNX model file {onnx_path} does not exist.")

        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime not installed.")

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        sample_img = np.random.randint(15, 70, (100, 100, 3), dtype=np.uint8)
        feats = extract_features(sample_img)
        X = np.array([[feats[k] for k in FEATURE_NAMES]], dtype=np.float32)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: X})
        pred_label = outputs[0][0]
        self.assertIn(pred_label, METHODS.keys())


class TestMultiDegradationPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.sample_img = np.random.randint(30, 150, (100, 100, 3), dtype=np.uint8)

    def test_extended_features(self):
        from src.multi_degradation import extract_extended_features, EXTENDED_FEATURE_NAMES
        feats = extract_extended_features(self.sample_img)
        for name in EXTENDED_FEATURE_NAMES:
            self.assertIn(name, feats)
            val = feats[name]
            self.assertFalse(np.isnan(val), f"NaN in extended feature: {name}")
            self.assertFalse(np.isinf(val), f"Inf in extended feature: {name}")

    def test_multi_methods(self):
        from src.multi_degradation import MULTI_METHODS, high_boost_sharpen, dcp_dehaze
        self.assertEqual(len(MULTI_METHODS), 8)

        sharpened = high_boost_sharpen(self.sample_img)
        self.assertEqual(sharpened.shape, self.sample_img.shape)
        self.assertEqual(sharpened.dtype, np.uint8)

        dehazed = dcp_dehaze(self.sample_img)
        self.assertEqual(dehazed.shape, self.sample_img.shape)
        self.assertEqual(dehazed.dtype, np.uint8)


class TestDatasetAndSceneSplits(unittest.TestCase):
    def test_parse_scene_id_grouping(self):
        from src.dataset import parse_scene_id
        # Multi-shot variants must share identical scene group
        self.assertEqual(parse_scene_id("1_0.png"), "1")
        self.assertEqual(parse_scene_id("1_1.png"), "1")
        self.assertEqual(parse_scene_id("1_2.png"), "1")
        self.assertEqual(parse_scene_id("1_ambient.png"), "1")
        self.assertEqual(parse_scene_id("2_0.png"), "2")
        self.assertEqual(parse_scene_id("10_1.png"), "10")
        # Standard numeric stems
        self.assertEqual(parse_scene_id("1.png"), "1")
        self.assertEqual(parse_scene_id("10.png"), "10")
        self.assertEqual(parse_scene_id("100.png"), "100")
        self.assertEqual(parse_scene_id("780.png"), "780")

    def test_scene_splits_csv_integrity(self):
        import pandas as pd
        csv_path = "data/scene_splits.csv"
        if not os.path.exists(csv_path):
            from src.dataset import export_scene_splits_csv
            export_scene_splits_csv()

        self.assertTrue(os.path.exists(csv_path))
        df = pd.read_csv(csv_path)
        self.assertEqual(list(df.columns), ["filename", "scene_id"])
        self.assertEqual(len(df), 485)
        self.assertEqual(df["filename"].isnull().sum(), 0)
        self.assertEqual(df["scene_id"].isnull().sum(), 0)


if __name__ == "__main__":
    unittest.main()

