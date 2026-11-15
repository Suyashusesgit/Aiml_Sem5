"""
Conformal Prediction & Risk-Controlled Abstention Module (Phase 1)
Authors: Team Triathlon (Logi Vasan, Suyash Srivastav, Ananya Kasaudhan)
Supervisor: Prof. Kishor P. Upla (Senior Member, IEEE)

Implements:
1. Split-Conformal Prediction sets with distribution-free finite-sample coverage guarantees.
2. Non-conformity scoring: s_i = 1 - p(y_i | x_i).
3. Conformal quantile calibration at level 1 - alpha (e.g. 90%, 95%).
4. Confidence-gated abstention: when max_a p(a | x) < tau, abstain to safe Best-Fixed default (Retinex).
5. Empirical coverage and regret-vs-abstention curves.
"""

import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


class SplitConformalClassifier:
    """
    Split-Conformal Predictor for multi-class algorithm selection.
    Guarantees that for any test sample (x, y):
        P(y in C(x)) >= 1 - alpha
    without making any parametric assumptions about the underlying distribution.
    """

    def __init__(self, model: Any, alpha: float = 0.10):
        """
        Args:
            model: Fitted scikit-learn classifier providing predict_proba(X).
            alpha: Desired error rate (default 0.10 -> 90% target coverage).
        """
        self.model = model
        self.alpha = float(alpha)
        self.classes_ = np.array(model.classes_)
        self.q_hat_: Optional[float] = None
        self.calibration_scores_: Optional[np.ndarray] = None

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> float:
        """
        Calibrates the non-conformity threshold q_hat on a held-out calibration set.
        Non-conformity score: s_i = 1 - p_model(y_i | x_i).
        """
        probs = self.model.predict_proba(X_cal)
        n = len(y_cal)
        if n == 0:
            raise ValueError("Calibration set cannot be empty")

        # Map y_cal to class indices and compute non-conformity score
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        scores = np.ones(n, dtype=np.float64)
        for i, y_true in enumerate(y_cal):
            if y_true in class_to_idx:
                scores[i] = 1.0 - probs[i, class_to_idx[y_true]]
            else:
                scores[i] = 1.0  # Zero probability for unseen class -> max non-conformity
        self.calibration_scores_ = scores

        # Finite-sample adjusted conformal quantile: ceil((n+1)(1-alpha)) / n
        p_val = min(1.0, np.ceil((n + 1.0) * (1.0 - self.alpha)) / float(n))
        self.q_hat_ = float(np.quantile(scores, p_val, method="higher"))
        return self.q_hat_

    def predict_set(self, X: np.ndarray) -> List[List[str]]:
        """
        Returns the prediction set C(x) for each query sample in X.
        C(x) = {y : p_model(y | x) >= 1 - q_hat}
        """
        if self.q_hat_ is None:
            raise RuntimeError("Conformal predictor must be calibrated before predicting sets.")

        probs = self.model.predict_proba(X)
        threshold = 1.0 - self.q_hat_

        prediction_sets = []
        for p in probs:
            admissible_classes = [self.classes_[i] for i in range(len(self.classes_)) if p[i] >= threshold]
            # Ensure non-empty prediction set by fallback to argmax
            if not admissible_classes:
                admissible_classes = [self.classes_[np.argmax(p)]]
            prediction_sets.append(admissible_classes)

        return prediction_sets

    def evaluate_coverage(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluates empirical coverage rate and average set size on a test split.
        """
        pred_sets = self.predict_set(X_test)
        n = len(y_test)
        covered = sum(1 for i in range(n) if y_test[i] in pred_sets[i])
        avg_size = np.mean([len(s) for s in pred_sets])

        return {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": float(covered / n),
            "average_set_size": float(avg_size),
            "n_samples": n,
        }


def evaluate_confidence_abstention(
    model: Any,
    X: np.ndarray,
    y_oracle: np.ndarray,
    df_scores: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
    default_fallback: str = "retinex"
) -> pd.DataFrame:
    """
    Evaluates confidence-gated abstention (Phase 1 Semester 6).
    When max_a p(a | x) < tau, the selector abstains from risky prediction and
    defaults to static Retinex.
    """
    if thresholds is None:
        thresholds = [0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    probs = model.predict_proba(X)
    classes = list(model.classes_)
    max_confs = np.max(probs, axis=1)
    argmax_classes = np.array([classes[i] for i in np.argmax(probs, axis=1)])

    oracle_psnrs = df_scores["oracle_psnr"].values

    records = []
    for tau in thresholds:
        accepted_mask = max_confs >= tau
        abstain_mask = ~accepted_mask

        n_total = len(X)
        n_accepted = int(np.sum(accepted_mask))
        n_abstained = int(np.sum(abstain_mask))
        abstain_rate = float(n_abstained / n_total) * 100.0

        # Effective decision: model pick if accepted, fallback if abstained
        decisions = []
        for i in range(n_total):
            if accepted_mask[i]:
                decisions.append(argmax_classes[i])
            else:
                decisions.append(default_fallback)
        decisions = np.array(decisions)

        # Regret calculations
        decision_psnrs = np.array([df_scores.iloc[i][f"psnr_{decisions[i]}"] for i in range(n_total)])
        regrets = np.maximum(0.0, oracle_psnrs - decision_psnrs)
        mean_regret = float(np.mean(regrets))

        # Metrics strictly on accepted subset
        if n_accepted > 0:
            acc_top1 = float(np.mean(argmax_classes[accepted_mask] == y_oracle[accepted_mask])) * 100.0
            acc_psnrs = np.array([df_scores.iloc[i][f"psnr_{argmax_classes[i]}"] for i in np.where(accepted_mask)[0]])
            acc_regret = float(np.mean(np.maximum(0.0, oracle_psnrs[accepted_mask] - acc_psnrs)))
        else:
            acc_top1 = 0.0
            acc_regret = 0.0

        records.append({
            "confidence_threshold_tau": tau,
            "abstain_rate_pct": round(abstain_rate, 1),
            "accepted_count": n_accepted,
            "abstained_count": n_abstained,
            "accepted_top1_acc_pct": round(acc_top1, 2),
            "accepted_mean_regret_db": round(acc_regret, 3),
            "overall_effective_regret_db": round(mean_regret, 3),
        })

    return pd.DataFrame(records)
