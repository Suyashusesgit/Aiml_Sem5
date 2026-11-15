"""
Plots Module
Generates publication-quality scientific visualizations for Project Triathlon.
Per Section 5 target layout.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluate import (
    plot_oracle_distribution,
    plot_confusion_matrix,
    plot_regret_cdf,
    plot_feature_importances,
    plot_qualitative_strip,
)

__all__ = [
    "plot_oracle_distribution",
    "plot_confusion_matrix",
    "plot_regret_cdf",
    "plot_feature_importances",
    "plot_qualitative_strip",
]
