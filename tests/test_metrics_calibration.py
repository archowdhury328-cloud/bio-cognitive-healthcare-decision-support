"""Metrics honesty (unestimable stays NaN), bootstrap, calibration isolation."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from biocog import metrics, pipeline
from biocog.calibrate import PlattCalibrator


def _data(n_pat=40, per=10, seed=0):
    rng = np.random.default_rng(seed)
    pats = np.repeat([f"P{i}" for i in range(n_pat)], per)
    y = (rng.random(len(pats)) < 0.25).astype(int)
    p = np.clip(0.3 + 0.35 * y + rng.normal(0, 0.2, len(y)), 0.01, 0.99)
    return y, p, pats


def test_window_metrics_match_sklearn_and_confusion_counts():
    y, p, pats = _data()
    m = metrics.window_metrics(y, p, pats, 0.5)
    assert m["auroc"] == pytest.approx(roc_auc_score(y, p))
    assert m["average_precision"] == pytest.approx(average_precision_score(y, p))
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == len(y)
    assert m["sensitivity"] == pytest.approx(m["tp"] / (m["tp"] + m["fn"]))
    assert m["n_positive_windows"] == int(y.sum())


def test_too_few_positive_patients_is_unestimable_not_zero():
    y = np.zeros(100, int)
    y[:5] = 1  # five positive windows, ONE patient
    pats = np.array(["A"] * 5 + [f"N{i}" for i in range(95)])
    m = metrics.window_metrics(y, np.linspace(0, 1, 100), pats, 0.5, min_patients=3)
    assert m["estimable"] is False and np.isnan(m["auroc"]) and "unestimable" in m["note"]
    assert m["n_positive_patients"] == 1 and m["n_windows"] == 100  # denominators still reported


def test_bootstrap_skipped_when_event_patients_below_minimum():
    y, p, pats = _data(n_pat=40)
    assert metrics.patient_bootstrap(y, p, pats, 0.5, 50, 0, min_event_patients=10_000) == {}


def test_bootstrap_is_patient_clustered_deterministic_and_brackets_estimate():
    y, p, pats = _data(n_pat=60, per=12)
    a = metrics.patient_bootstrap(y, p, pats, 0.5, 120, 0, 5)
    b = metrics.patient_bootstrap(y, p, pats, 0.5, 120, 0, 5)
    assert a == b
    point = metrics.window_metrics(y, p, pats, 0.5)["auroc"]
    lo, hi = a["auroc"]
    assert lo < point < hi


def test_calibration_table_counts_and_few_windows_flag():
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.05, 0.1, 0.9, 0.95, 0.85, 0.5])
    t = metrics.calibration_table(y, p, 5)
    assert t.n.sum() == 6 and t.few_windows.all()
    assert t.loc[t.bin == 4, "observed_rate"].iloc[0] == 1.0


def test_platt_monotone_roundtrip_and_identity_fallback(tmp_path):
    y, p, _ = _data()
    raw = np.log(p / (1 - p))
    c = PlattCalibrator().fit(raw, y)
    q = c.predict(np.sort(raw))
    assert np.all(np.diff(q) >= 0) and not c.identity
    c.save(tmp_path / "c.json")
    assert np.allclose(PlattCalibrator.load(tmp_path / "c.json").predict(raw), c.predict(raw))
    one = PlattCalibrator().fit(raw, np.ones_like(y))
    assert one.identity and np.allclose(one.predict(np.array([0.0])), 0.5)


def test_calibration_uses_validation_patients_only(bundle):
    model = pipeline.fit_model("last_value", bundle)
    ref = pipeline.fit_calibrator(model, bundle)
    idx = bundle.feats.index.copy()
    flip = idx.split.isin(["train", "test"])
    idx.loc[flip, "label"] = np.where(idx.loc[flip, "label"] == 1, 0, np.where(idx.loc[flip, "label"] == 0, 1, -1))
    scrambled = dataclasses.replace(bundle, feats=dataclasses.replace(bundle.feats, index=idx))
    other = pipeline.fit_calibrator(model, scrambled)
    assert (other.a, other.b) == (ref.a, ref.b)  # train/test labels cannot reach the calibrator
    idx2 = bundle.feats.index.copy()
    v = idx2.split == "val"
    idx2.loc[v, "label"] = np.where(idx2.loc[v, "label"] == 1, 0, np.where(idx2.loc[v, "label"] == 0, 1, -1))
    moved = pipeline.fit_calibrator(model, dataclasses.replace(bundle, feats=dataclasses.replace(bundle.feats, index=idx2)))
    assert (moved.a, moved.b) != (ref.a, ref.b)  # sanity: val labels DO matter
    assert ref.fitted_on_n == int(((bundle.windows.split == "val") & (bundle.windows.label >= 0)).sum())
