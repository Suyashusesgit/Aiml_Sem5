"""
Unit tests for Video Generalization & Temporal Feature Smoothing Engine (Phase 2)
"""

import unittest
import numpy as np
import joblib
import os

from src.video_pipeline import (
    TemporalFeatureFilter,
    MarkovianHysteresisGate,
    FastTemporalInvertDehazer,
    VideoStreamProcessor,
)
from src.features import FEATURE_NAMES


class DummyModel:
    def __init__(self, fixed_class="gamma"):
        self.classes_ = ["gamma", "invert_dehaze", "clahe", "retinex"]
        self.fixed_class = fixed_class

    def predict_proba(self, X):
        probs = [0.1, 0.1, 0.1, 0.1]
        idx = self.classes_.index(self.fixed_class)
        probs[idx] = 0.7
        return np.array([probs])

    def predict(self, X):
        return [self.fixed_class]


class TestVideoPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.frame1 = np.random.randint(15, 60, (240, 320, 3), dtype=np.uint8)
        self.frame2 = np.random.randint(15, 60, (240, 320, 3), dtype=np.uint8)

    def test_temporal_feature_filter(self):
        filt = TemporalFeatureFilter(alpha=0.3, keyframe_interval=3)
        feats1, is_key1 = filt.update(self.frame1)
        self.assertTrue(is_key1)
        self.assertEqual(len(feats1), len(FEATURE_NAMES))

        # Second frame: non-keyframe with small motion
        feats2, is_key2 = filt.update(self.frame1)
        self.assertFalse(is_key2)
        self.assertEqual(feats1, feats2)

    def test_hysteresis_gate_prevents_flicker(self):
        gate = MarkovianHysteresisGate(switch_penalty=0.20)
        
        # Frame 1: Gamma slightly ahead
        p1 = {"gamma": 0.52, "invert_dehaze": 0.48}
        m1 = gate.select(p1)
        self.assertEqual(m1, "gamma")

        # Frame 2: Invert_dehaze barely ahead by 0.02 (below 0.20 penalty margin)
        p2 = {"gamma": 0.49, "invert_dehaze": 0.51}
        m2 = gate.select(p2)
        self.assertEqual(m2, "gamma")  # Should NOT flicker!
        self.assertEqual(gate.switch_count, 0)

        # Frame 3: Invert_dehaze strongly ahead (0.75 vs 0.25)
        p3 = {"gamma": 0.25, "invert_dehaze": 0.75}
        m3 = gate.select(p3)
        self.assertEqual(m3, "invert_dehaze")
        self.assertEqual(gate.switch_count, 1)

    def test_fast_temporal_dehazer(self):
        dehazer = FastTemporalInvertDehazer()
        out1 = dehazer.enhance(self.frame1)
        self.assertEqual(out1.shape, self.frame1.shape)
        self.assertEqual(out1.dtype, np.uint8)
        self.assertTrue(0 <= out1.min() and out1.max() <= 255)

        out2 = dehazer.enhance(self.frame2)
        self.assertEqual(out2.shape, self.frame2.shape)

    def test_video_stream_processor(self):
        model = DummyModel("invert_dehaze")
        proc = VideoStreamProcessor(model=model, switch_penalty=0.15)
        
        for _ in range(5):
            enhanced, method, meta = proc.process_frame(self.frame1)
            self.assertEqual(enhanced.shape, self.frame1.shape)
            self.assertEqual(enhanced.dtype, np.uint8)
            self.assertEqual(method, "invert_dehaze")
            self.assertIn("fps", meta)
            self.assertGreater(meta["fps"], 0)


if __name__ == "__main__":
    unittest.main()
