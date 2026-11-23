"""Future observations must not change features or predictions; unreleased results are invisible."""

from __future__ import annotations

import numpy as np
import pandas as pd

from biocog import pipeline
from biocog.features import build_features
from biocog.rules import apply_rules
from helpers import ev, stay, windows_for

ARRAYS = ("static", "seq", "tab", "age_h", "n_obs", "n_trend", "slope", "last_val", "seq_r", "rule_vec")


def _feats(st, e, windows, cfg):
    return apply_rules(build_features(st, e, windows, cfg), cfg)


def _equal(a, b):
    for name in ARRAYS:
        assert np.array_equal(getattr(a, name), getattr(b, name), equal_nan=True), name
    pd.testing.assert_frame_equal(a.flags, b.flags)


def _rich_stay():
    rows = []
    for h in range(0, 40):
        rows += [ev("S1", "hr", h, 80 + h % 7), ev("S1", "sbp", h, 120 - h % 5)]
    for h in (2, 6, 9, 13, 17, 22):
        rows.append(ev("S1", "glucose", h, 120 + h, avail_h=h + 1.5))
    for h in (5, 17):
        rows += [ev("S1", "creatinine", h, 1.0 + h / 50, avail_h=h + 2), ev("S1", "sodium", h, 139, avail_h=h + 2)]
    return [stay(followup_h=40.0)], rows


def test_changing_every_post_landmark_observation_leaves_features_identical(cfg):
    stays, rows = _rich_stay()
    st, e, _, w = windows_for(stays, rows, cfg)
    rng = np.random.default_rng(0)
    for i in (0, 3, 7, 12):
        t = float(w.t_h.iloc[i])
        base = _feats(st, e, w.iloc[[i]], cfg)
        pert = e.copy()
        future = pert.t_avail > t  # not yet known at the landmark (includes drawn-before-t, released-after-t)
        pert.loc[future, "value"] = rng.uniform(20, 400, future.sum())
        extra = pd.DataFrame({"stay_id": "S1", "variable": "glucose", "t_chart": t + 1.0, "t_avail": t + 2.0,
                              "value": 999.0}, index=[0])
        pert = pd.concat([pert, extra], ignore_index=True).sort_values(["stay_id", "variable", "t_chart"])
        assert future.sum() > 0
        _equal(base, _feats(st, pert, w.iloc[[i]], cfg))


def test_lab_drawn_before_landmark_but_released_after_is_unavailable(cfg):
    rows = [ev("S1", "glucose", 7.0, 150.0, avail_h=9.0), ev("S1", "glucose", 30.0, 110.0)]
    st, e, _, w = windows_for([stay()], rows, cfg)
    f = _feats(st, e, w, cfg)
    t = f.index.t_h.to_numpy()
    g = 5  # glucose column
    at8, at9 = f.take(np.where(t == 8.0)[0]), f.take(np.where(t == 9.0)[0])
    assert np.isnan(at8.last_val[0, g]) and at8.n_obs[0, g] == 0  # drawn at 7, released at 9: invisible at 8
    assert at9.last_val[0, g] == 150.0 and at9.age_h[0, g] == 2.0  # visible once released (<= inclusive)


def test_order_placed_before_prediction_cannot_expose_later_result(cfg):
    # same mechanism as above, expressed through the model-facing grid: no mask bit before release
    rows = [ev("S1", "glucose", 7.0, 150.0, avail_h=10.0), ev("S1", "glucose", 30.0, 100.0)]
    st, e, _, w = windows_for([stay()], rows, cfg)
    f = _feats(st, e, w, cfg)
    row = np.where(f.index.t_h.to_numpy() == 9.0)[0][0]
    assert f.seq[row, :, 5, 1].sum() == 0 and f.seq[row, :, 5, 0].max() == 0


def test_predictions_unchanged_when_future_is_perturbed(bundle, fixture_tables, tiny_cfg, tiny_proposed):
    """Model-level version of the same property, for baselines and the proposed network."""
    st, e = __import__("biocog.cohort", fromlist=["to_hours"]).to_hours(*fixture_tables)
    w = bundle.windows
    sid = w[w.split == "test"].groupby("stay_id").size().idxmax()
    rows = w[w.stay_id == sid].reset_index(drop=True)
    models = {n: pipeline.fit_model(n, bundle) for n in ("last_value", "logistic")}
    models["proposed"] = tiny_proposed
    rng = np.random.default_rng(1)
    for i in (2, len(rows) // 2, len(rows) - 2):
        t = float(rows.t_h.iloc[i])
        pert = e.copy()
        future = (pert.stay_id == sid) & (pert.t_avail > t)
        pert.loc[future, "value"] = rng.uniform(30, 300, future.sum())
        base = apply_rules(build_features(st, e, rows.iloc[[i]], tiny_cfg), tiny_cfg)
        alt = apply_rules(build_features(st, pert, rows.iloc[[i]], tiny_cfg), tiny_cfg)
        for name, m in models.items():
            assert np.allclose(m.predict_raw(base), m.predict_raw(alt), atol=1e-6), name
