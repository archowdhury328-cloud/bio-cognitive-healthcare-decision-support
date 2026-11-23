"""Platt calibration. Fitted ONLY on validation patients' labelled windows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """p = sigmoid(a * raw + b). Falls back to identity (a=1, b=0) if a class is missing."""

    def __init__(self) -> None:
        self.a, self.b, self.fitted_on_n, self.identity = 1.0, 0.0, 0, True

    def fit(self, raw: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        raw, y = np.asarray(raw, float), np.asarray(y, int)
        self.fitted_on_n = len(y)
        if len(np.unique(y)) < 2:
            self.a, self.b, self.identity = 1.0, 0.0, True
            return self
        lr = LogisticRegression(C=100.0, max_iter=1000).fit(raw.reshape(-1, 1), y)
        self.a, self.b, self.identity = float(lr.coef_[0, 0]), float(lr.intercept_[0]), False
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        z = self.a * np.asarray(raw, float) + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "fitted_on_n": self.fitted_on_n, "identity": self.identity}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> PlattCalibrator:
        d = json.loads(path.read_text(encoding="utf-8"))
        c = cls()
        c.a, c.b, c.fitted_on_n, c.identity = d["a"], d["b"], d["fitted_on_n"], d["identity"]
        return c
