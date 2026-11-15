"""
Selectors Module
Implements classical machine learning selectors and baseline strategies
for Project Triathlon per Appendix A & Section 6.
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, RandomForestRegressor

from src.methods import CANONICAL_METHODS


class BaseSelector:
    """Abstract base class for enhancement selectors."""
    def __init__(self, name: str, methods: Optional[List[str]] = None):
        self.name = name
        self.methods = methods if methods is not None else list(CANONICAL_METHODS)
        self.classes_ = np.array(self.methods)

    def fit(self, X: np.ndarray, y: np.ndarray, df_scores: Optional[pd.DataFrame] = None) -> "BaseSelector":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ClassifierSelector(BaseSelector):
    """
    Standard classifier-based selector wrapping a scikit-learn pipeline
    with StandardScaler strictly inside the pipeline.
    """
    def __init__(
        self,
        name: str,
        estimator: Any,
        methods: Optional[List[str]] = None,
        use_scaler: bool = True
    ):
        super().__init__(name, methods)
        if use_scaler:
            self.pipeline = make_pipeline(StandardScaler(), estimator)
        else:
            self.pipeline = make_pipeline(estimator)
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray, df_scores: Optional[pd.DataFrame] = None) -> "ClassifierSelector":
        self.pipeline.fit(X, y)
        clf = self.pipeline.steps[-1][1]
        self.classes_ = getattr(clf, "classes_", np.array(self.methods))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X)
        preds = self.predict(X)
        proba = np.zeros((len(X), len(self.classes_)))
        for i, p in enumerate(preds):
            idx = np.where(self.classes_ == p)[0]
            if len(idx) > 0:
                proba[i, idx[0]] = 1.0
        return proba


class RidgePTOSelector(BaseSelector):
    """
    Predict-Then-Optimize (PTO) Selector using Ridge regression on
    per-image-centered PSNR values of candidate enhancement methods.
    """
    def __init__(self, alpha: float = 1.0, methods: Optional[List[str]] = None):
        super().__init__("ridge_pto", methods)
        self.alpha = alpha
        self.models: Dict[str, Pipeline] = {}

    def fit(self, X: np.ndarray, y: np.ndarray, df_scores: Optional[pd.DataFrame] = None) -> "RidgePTOSelector":
        if df_scores is None:
            raise ValueError("RidgePTOSelector requires df_scores with psnr_* columns")
        
        # Center PSNR across methods per image
        psnr_cols = [f"psnr_{m}" for m in self.methods]
        psnr_vals = df_scores[psnr_cols].values
        centered_psnr = psnr_vals - np.mean(psnr_vals, axis=1, keepdims=True)

        for j, m in enumerate(self.methods):
            pipe = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha, random_state=42))
            pipe.fit(X, centered_psnr[:, j])
            self.models[m] = pipe
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred_matrix = np.zeros((len(X), len(self.methods)))
        for j, m in enumerate(self.methods):
            pred_matrix[:, j] = self.models[m].predict(X)
        best_indices = np.argmax(pred_matrix, axis=1)
        return np.array([self.methods[idx] for idx in best_indices])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pred_matrix = np.zeros((len(X), len(self.methods)))
        for j, m in enumerate(self.methods):
            pred_matrix[:, j] = self.models[m].predict(X)
        # Softmax over predicted centered PSNR
        exp_vals = np.exp(pred_matrix - np.max(pred_matrix, axis=1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


class BestFixedSelector(BaseSelector):
    """Selects the single method that minimizes mean regret on the training fold."""
    def __init__(self, methods: Optional[List[str]] = None):
        super().__init__("best_fixed", methods)
        self.best_method = self.methods[0]

    def fit(self, X: np.ndarray, y: np.ndarray, df_scores: Optional[pd.DataFrame] = None) -> "BestFixedSelector":
        if df_scores is not None:
            oracle_psnr = df_scores["oracle_psnr"].values if "oracle_psnr" in df_scores else None
            if oracle_psnr is None:
                psnr_cols = [f"psnr_{m}" for m in self.methods if f"psnr_{m}" in df_scores]
                oracle_psnr = df_scores[psnr_cols].max(axis=1).values
            
            mean_regrets = {}
            for m in self.methods:
                if f"psnr_{m}" in df_scores:
                    r = oracle_psnr - df_scores[f"psnr_{m}"].values
                    mean_regrets[m] = float(np.mean(r))
            if mean_regrets:
                self.best_method = min(mean_regrets.keys(), key=lambda k: mean_regrets[k])
        else:
            # Fallback to majority vote
            from collections import Counter
            counts = Counter(y)
            self.best_method = counts.most_common(1)[0][0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.best_method] * len(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = np.zeros((len(X), len(self.methods)))
        idx = self.methods.index(self.best_method)
        proba[:, idx] = 1.0
        return proba


class MajoritySelector(BaseSelector):
    """Selects the majority class in the training oracle labels."""
    def __init__(self, methods: Optional[List[str]] = None):
        super().__init__("fixed_majority", methods)
        self.majority_method = self.methods[0]

    def fit(self, X: np.ndarray, y: np.ndarray, df_scores: Optional[pd.DataFrame] = None) -> "MajoritySelector":
        from collections import Counter
        counts = Counter(y)
        self.majority_method = counts.most_common(1)[0][0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.majority_method] * len(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = np.zeros((len(X), len(self.methods)))
        idx = self.methods.index(self.majority_method)
        proba[:, idx] = 1.0
        return proba


class RandomSelector(BaseSelector):
    """Random choice baseline with reproducible seed."""
    def __init__(self, seed: int = 42, methods: Optional[List[str]] = None):
        super().__init__("random", methods)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def fit(self, X: np.ndarray, y: np.ndarray, df_scores: Optional[pd.DataFrame] = None) -> "RandomSelector":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        indices = self.rng.randint(0, len(self.methods), size=len(X))
        return np.array([self.methods[i] for i in indices])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = 1.0 / len(self.methods)
        return np.full((len(X), len(self.methods)), p)


class RuleSelector(BaseSelector):
    """Handcrafted domain heuristic rule based on brightness, contrast, and noise."""
    def __init__(self, feature_names: List[str], methods: Optional[List[str]] = None):
        super().__init__("rule_fixed", methods)
        self.feature_names = feature_names

    def fit(self, X: np.ndarray, y: np.ndarray, df_scores: Optional[pd.DataFrame] = None) -> "RuleSelector":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        b_idx = self.feature_names.index("brightness") if "brightness" in self.feature_names else 0
        c_idx = self.feature_names.index("contrast") if "contrast" in self.feature_names else 1
        n_idx = self.feature_names.index("noise") if "noise" in self.feature_names else 3

        preds = []
        for row in X:
            mu, sigma, noise = row[b_idx], row[c_idx], row[n_idx]
            if mu < 20.0:
                preds.append("retinex")
            elif noise > 7.0:
                preds.append("bilat_unsharp")
            elif sigma < 25.0:
                preds.append("clahe")
            elif mu < 50.0:
                preds.append("gamma")
            elif mu > 120.0:
                preds.append("identity")
            else:
                preds.append("hist_eq")
        return np.array(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        proba = np.zeros((len(X), len(self.methods)))
        for i, p in enumerate(preds):
            if p in self.methods:
                proba[i, self.methods.index(p)] = 1.0
        return proba


def make_selector(
    name: str,
    feature_names: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    **kwargs
) -> BaseSelector:
    """
    Factory function for instantiation of selectors and baselines.
    """
    m = methods if methods is not None else list(CANONICAL_METHODS)
    name_lower = name.lower().replace("-", "_").replace(" ", "_")

    if name_lower in ("logreg", "logistic_regression"):
        max_iter = kwargs.get("max_iter", 2000)
        C = kwargs.get("C", 1.0)
        class_weight = kwargs.get("class_weight", None)
        random_state = kwargs.get("random_state", 42)
        solver = kwargs.get("solver", "lbfgs")
        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver
        )
        return ClassifierSelector("Logistic Regression", clf, methods=m)

    elif name_lower in ("rf", "random_forest"):
        n_estimators = kwargs.get("n_estimators", 300)
        max_depth = kwargs.get("max_depth", None)
        min_samples_leaf = kwargs.get("min_samples_leaf", 1)
        max_features = kwargs.get("max_features", "sqrt")
        class_weight = kwargs.get("class_weight", None)
        random_state = kwargs.get("random_state", 42)
        n_jobs = kwargs.get("n_jobs", 1)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )
        return ClassifierSelector("Random Forest", clf, methods=m)

    elif name_lower in ("histgb", "hist_gradient_boosting", "gboost"):
        random_state = kwargs.get("random_state", 42)
        clf = HistGradientBoostingClassifier(random_state=random_state)
        return ClassifierSelector("HistGradientBoosting", clf, methods=m)

    elif name_lower in ("ridge_pto", "pto"):
        alpha = kwargs.get("alpha", 1.0)
        return RidgePTOSelector(alpha=alpha, methods=m)

    elif name_lower in ("best_fixed", "fixed_mean"):
        return BestFixedSelector(methods=m)

    elif name_lower in ("fixed_majority", "majority"):
        return MajoritySelector(methods=m)

    elif name_lower in ("random", "random_choice"):
        seed = kwargs.get("random_state", kwargs.get("seed", 42))
        return RandomSelector(seed=seed, methods=m)

    elif name_lower in ("rule_fixed", "hand_rule", "rule"):
        fn = feature_names if feature_names is not None else []
        return RuleSelector(feature_names=fn, methods=m)

    else:
        raise ValueError(f"Unknown selector name: {name}")
