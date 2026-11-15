"""
Backwards-compatibility re-export for Downstream Semantic Evaluation.
Implementation lives in extensions/downstream/downstream_semantic_eval.py.
"""

from extensions.downstream.downstream_semantic_eval import (
    evaluate_dataset_semantic_enhancement,
)

__all__ = [
    "evaluate_dataset_semantic_enhancement",
]
