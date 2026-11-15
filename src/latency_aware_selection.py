"""
Backwards-compatibility re-export for Latency-Aware Selection.
Implementation lives in extensions/latency/latency_aware_selection.py.
"""

from extensions.latency.latency_aware_selection import (
    LatencyAwareSelector,
    METHOD_LATENCIES_MS,
    compute_pareto_latency_regret_curve,
)

__all__ = [
    "LatencyAwareSelector",
    "METHOD_LATENCIES_MS",
    "compute_pareto_latency_regret_curve",
]
