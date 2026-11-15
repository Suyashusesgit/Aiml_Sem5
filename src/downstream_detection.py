"""
Backwards-compatibility re-export for Downstream Keypoint Detection.
Implementation lives in extensions/downstream/downstream_detection.py.
"""

from extensions.downstream.downstream_detection import (
    evaluate_downstream_keypoints,
)

__all__ = [
    "evaluate_downstream_keypoints",
]
