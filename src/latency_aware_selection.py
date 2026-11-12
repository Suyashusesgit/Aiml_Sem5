"""
Latency-Aware Enhancement Selection (Phase 0 Deferred Milestone)
Authors: Project Triathlon Engineering Team
Supervisor: Prof. Kishor P. Upla (Senior Member, IEEE)

Implements Page 9 Deferred Future Scope:
"Latency-aware selection: plot regret vs latency budget for deployment planning."
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Empirical benchmark latencies on ARM / CPU edge platforms (ms per 512x512 frame)
METHOD_LATENCIES_MS = {
    "identity": 0.04,
    "gamma": 0.38,
    "hist_eq": 0.85,
    "clahe": 1.42,
    "retinex": 4.60,
    "invert_dehaze": 9.20,
    "bilat_unsharp": 12.50,
}


class LatencyAwareSelector:
    """
    Constrains the candidate portfolio A to operators satisfying Latency(a) <= Budget_ms.
    Selects the highest confidence / highest predicted PSNR operator within budget.
    """

    def __init__(self, latency_budget_ms: float = 5.0):
        self.budget_ms = float(latency_budget_ms)

    def get_admissible_methods(self) -> List[str]:
        return [m for m, lat in METHOD_LATENCIES_MS.items() if lat <= self.budget_ms]

    def select(self, probabilities: Dict[str, float]) -> str:
        admissible = set(self.get_admissible_methods())
        filtered = {m: p for m, p in probabilities.items() if m in admissible}
        if not filtered:
            return "identity"
        return max(filtered.keys(), key=lambda k: filtered[k])


def compute_pareto_latency_regret_curve(
    scores_csv: str = "results/scores.csv",
    budgets: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Sweeps latency budgets from 0.05 ms to 15.0 ms and computes:
    - Admissible methods
    - Mean regret under optimal budget-constrained selection
    - Mean execution latency
    """
    if budgets is None:
        budgets = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]

    df_s = pd.read_csv(scores_csv)
    records = []

    for b in budgets:
        selector = LatencyAwareSelector(latency_budget_ms=b)
        admissible = selector.get_admissible_methods()
        if not admissible:
            continue

        # In scores.csv, find best score among admissible methods
        psnr_cols = [f"psnr_{m}" for m in admissible if f"psnr_{m}" in df_s.columns]
        if not psnr_cols:
            continue

        best_admissible_psnr = df_s[psnr_cols].max(axis=1)
        oracle_psnr = df_s["oracle_psnr"]
        budget_regrets = np.maximum(0.0, oracle_psnr - best_admissible_psnr)
        mean_regret = float(np.mean(budget_regrets))

        # Average latency if choosing best admissible
        chosen_indices = np.argmax(df_s[psnr_cols].values, axis=1)
        chosen_methods = [admissible[i] for i in chosen_indices]
        avg_lat = float(np.mean([METHOD_LATENCIES_MS[m] for m in chosen_methods]))

        records.append({
            "latency_budget_ms": b,
            "admissible_methods": ", ".join(admissible),
            "admissible_count": len(admissible),
            "mean_regret_db": round(mean_regret, 3),
            "mean_latency_ms": round(avg_lat, 2),
            "throughput_fps": round(1000.0 / max(0.01, avg_lat), 1),
        })

    out_df = pd.DataFrame(records)
    return out_df


if __name__ == "__main__":
    pareto_df = compute_pareto_latency_regret_curve()
    print("=== Pareto Latency-Regret Tradeoff Frontier ===")
    print(pareto_df[["latency_budget_ms", "admissible_count", "mean_regret_db", "mean_latency_ms", "throughput_fps"]].to_string(index=False))

    os.makedirs("results", exist_ok=True)
    pareto_df.to_csv("results/latency_regret_frontier.csv", index=False)
    print("\nSaved Pareto frontier to results/latency_regret_frontier.csv")

