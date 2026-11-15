"""
Paper Assets Module
Generates reproducible LaTeX tables and figures from experiment result CSVs
for publication in the IEEE Transactions paper.
Per Section 5 & 7.
"""

import os
import sys
from typing import Optional
import numpy as np
import pandas as pd


def generate_primary_table(
    csv_path: str = "results/decision/explore_default/summary_cv.csv",
    out_tex: str = "paper/generated/tab_primary_results.tex"
) -> None:
    """
    Generates Table I LaTeX code from CV summary CSV.
    """
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    if not os.path.isfile(csv_path):
        # Placeholder if run not executed yet
        with open(out_tex, "w") as f:
            f.write("% Awaiting experiment run output\n")
        return

    df = pd.read_csv(csv_path)
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Primary Benchmark Results (5-Fold Grouped CV across 3 Random Seeds)}",
        r"\label{tab:primary_results}",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Selector / Strategy} & \textbf{Top-1 (\%)} & \textbf{Mean Regret (dB)} & \textbf{95\% CI (dB)} & \textbf{Max Regret (dB)} \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        name = row["selector"]
        top1 = f"{row['top1_acc']:.2f}"
        reg = f"{row['mean_regret_db']:.3f}"
        ci = f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]" if "ci_lo" in row else "--"
        max_r = f"{row['max_regret_db']:.2f}" if "max_regret_db" in row else "--"

        is_champion = "Logistic" in name or "Champion" in name
        if is_champion:
            lines.append(f"\\textbf{{{name}}} & \\textbf{{{top1}}} & \\textbf{{{reg}}} & \\textbf{{{ci}}} & \\textbf{{{max_r}}} \\\\")
        else:
            lines.append(f"{name} & {top1} & {reg} & {ci} & {max_r} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ])

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Generated {out_tex}")


def generate_all_paper_assets(run_name: str = "explore_default") -> None:
    """Orchestrates asset generation from a named experiment run."""
    run_dir = f"results/decision/{run_name}"
    summary_csv = os.path.join(run_dir, "summary_cv.csv")
    generate_primary_table(summary_csv)
    print(f"Paper assets generated from {run_dir}")


if __name__ == "__main__":
    generate_all_paper_assets()
