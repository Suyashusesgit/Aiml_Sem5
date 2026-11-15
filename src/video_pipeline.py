"""
Backwards-compatibility re-export for Video Generalization Engine.
Implementation lives in extensions/video/video_pipeline.py.
"""

from extensions.video.video_pipeline import (
    TemporalFeatureFilter,
    MarkovianHysteresisGate,
    FastTemporalInvertDehazer,
    VideoStreamProcessor,
)

__all__ = [
    "TemporalFeatureFilter",
    "MarkovianHysteresisGate",
    "FastTemporalInvertDehazer",
    "VideoStreamProcessor",
]
