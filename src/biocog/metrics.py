"""Window-level metrics with patient-level bootstrap. Unestimable metrics are NaN, never invented."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

METRIC_NAMES = ("accuracy", "sensitivity", "specificity", "precision", "auroc", "average_precision", "brier")


def window_metrics(y: np.ndarray, p: np.ndarray, patients: np.ndarray, thr: float,
                   min_patients: int = 3) -> dict[str, Any]:
    """Metrics on labelled windows. Needs >= ``min_patients`` positive AND negative patients."""
    y, p, patients = np.asarray(y, int), np.asarray(p, float), np.asarray(patients)
    pos_pat, neg_pat = len(set(patients[y == 1])), len(set(patients[y == 0]))
    out: dict[str, Any] = {"n_windows": len(y), "n_positive_windows": int(y.sum()),
                           "n_patients": len(set(patients)), "n_positive_patients": pos_pat,
                           "n_negative_patients": neg_pat, "prevalence": float(y.mean()) if len(y) else np.nan,
                           "threshold": float(thr)}
    out["estimable"] = bool(pos_pat >= min_patients and neg_pat >= min_patients)
    if not out["estimable"]:
        out.update({k: np.nan for k in METRIC_NAMES})
        out["note"] = (f"unestimable: {pos_pat} positive and {neg_pat} negative patients "
                       f"(need >= {min_patients} of each)")
        return out
    yhat = p >= thr
    tp, fp = int((yhat & (y == 1)).sum()), int((yhat & (y == 0)).sum())
    fn, tn = int((~yhat & (y == 1)).sum()), int((~yhat & (y == 0)).sum())
    out.update({
        "accuracy": (tp + tn) / len(y),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    })
    return out


def patient_bootstrap(y: np.ndarray, p: np.ndarray, patients: np.ndarray, thr: float, n_boot: int,
                      seed: int, min_event_patients: int, min_patients: int = 3) -> dict[str, Any]:
    """Cluster (patient-level) percentile bootstrap. Returns {} when positives are too few."""
    y, p, patients = np.asarray(y, int), np.asarray(p, float), np.asarray(patients)
    if len(set(patients[y == 1])) < min_event_patients:
        return {}
    ids = np.array(sorted(set(patients)))
    by = {i: np.where(patients == i)[0] for i in ids}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in METRIC_NAMES}
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        sel = np.concatenate([by[i] for i in pick])
        # relabel duplicate patients so patient counts reflect distinct draws
        m = window_metrics(y[sel], p[sel], patients[sel], thr, min_patients)
        if m["estimable"]:
            for k in METRIC_NAMES:
                draws[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
            for k, v in draws.items() if len(v) >= max(20, n_boot // 5)}


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    """Fixed-width probability bins. Bins with fewer than 5 windows are kept but flagged."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        rows.append({"bin": k, "lo": edges[k], "hi": edges[k + 1], "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()) if m.any() else np.nan,
                     "observed_rate": float(y[m].mean()) if m.any() else np.nan,
                     "few_windows": bool(m.sum() < 5)})
    return pd.DataFrame(rows)
