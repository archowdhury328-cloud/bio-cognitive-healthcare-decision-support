"""Conventional baselines on past-only tabular summaries. All expose ``predict_raw`` (real score)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import Features
from .schema import GLUCOSE_IDX


class LastValueBaseline:
    """Score = last observed glucose. Missing last value is filled with the training minimum."""

    name = "last_value"

    def fit(self, f: Features, y: np.ndarray, **_: Any) -> LastValueBaseline:
        known = f.last_val[:, GLUCOSE_IDX]
        self.fill_ = float(np.nanmin(known)) if np.isfinite(known).any() else 0.0
        return self

    def predict_raw(self, f: Features) -> np.ndarray:
        x = f.last_val[:, GLUCOSE_IDX]
        return np.where(np.isnan(x), self.fill_, x) / 100.0

    def save(self, path: Path) -> None:
        joblib.dump(self, path)


class LogisticBaseline:
    """Regularised logistic regression; C is chosen on validation patients (average precision)."""

    name = "logistic"
    C_GRID = (0.003, 0.03, 0.3)

    def __init__(self, cfg: dict[str, Any], seed: int = 0):
        self.c, self.seed = float(cfg["models"]["lr_c"]), seed

    def _pipe(self, c: float):
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=self.seed))

    def fit(self, f: Features, y: np.ndarray, val: tuple[Features, np.ndarray] | None = None,
            **_: Any) -> LogisticBaseline:
        if val is not None and val[1].sum() > 0:
            scores = {c: average_precision_score(val[1], self._pipe(c).fit(f.tab, y).decision_function(val[0].tab))
                      for c in self.C_GRID}
            self.c = max(scores, key=scores.get)
            self.c_grid_scores_ = {str(k): float(v) for k, v in scores.items()}
        self.pipe_ = self._pipe(self.c).fit(f.tab, y)
        return self

    def predict_raw(self, f: Features) -> np.ndarray:
        return self.pipe_.decision_function(f.tab)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)


class GradientBoostedBaseline:
    name = "gbt"

    def __init__(self, cfg: dict[str, Any], seed: int = 0):
        self.params, self.seed = dict(cfg["models"]["gbt"]), seed

    def fit(self, f: Features, y: np.ndarray, **_: Any) -> GradientBoostedBaseline:
        self.clf_ = HistGradientBoostingClassifier(random_state=self.seed, class_weight="balanced",
                                                   early_stopping=False, **self.params)
        self.clf_.fit(f.tab, y)
        return self

    def predict_raw(self, f: Features) -> np.ndarray:
        return self.clf_.decision_function(f.tab)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)


def load_baseline(path: Path):
    """Load a saved baseline. Files are joblib pickles: only load files you produced yourself."""
    return joblib.load(path)
