"""
Backwards-compatibility re-export for Edge Hardware Profiling.
Implementation lives in extensions/latency/edge_hardware_profile.py.
"""

from extensions.latency.edge_hardware_profile import (
    profile_pipeline,
)

__all__ = [
    "profile_pipeline",
]
