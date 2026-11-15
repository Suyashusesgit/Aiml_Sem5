"""
Backwards-compatibility re-export for Downstream Feature Matching.
Implementation lives in extensions/downstream/downstream_matching.py.
"""

from extensions.downstream.downstream_matching import (
    compute_orb_inlier_matches,
    evaluate_enhancement_matching_gain,
)

__all__ = [
    "compute_orb_inlier_matches",
    "evaluate_enhancement_matching_gain",
]
