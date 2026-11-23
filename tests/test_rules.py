"""Each rule: trigger, and exactly what it changes (feature, flag, or review decision)."""

from __future__ import annotations

import numpy as np

from biocog import explain, rules
from biocog.features import build_features
from biocog.schema import BIO_IDX, GLUCOSE_IDX, VARS
from helpers import ev, stay, windows_for


def _build(rows, cfg, followup=40.0):
    st, e, _, w = windows_for([stay(followup_h=followup)], rows, cfg)
    return build_features(st, e, w, cfg), w


def _row(f, t):
    return int(np.where(f.index.t_h.to_numpy() == t)[0][0])


def test_r1_trend_unavailable_below_two_values_and_available_at_two(cfg):
    rows = [ev("S1", "creatinine", 8.0, 1.0), ev("S1", "creatinine", 10.0, 1.4), ev("S1", "glucose", 30.0, 100.0)]
    f, _ = _build(rows, cfg)
    out = rules.apply_rules(f, cfg)
    cr = BIO_IDX.index(VARS.index("creatinine"))
    cols = {n: i for i, n in enumerate(rules.RULE_COLS)}
    r9, r11 = _row(f, 9.0), _row(f, 11.0)
    assert out.rule_vec[r9, cols["creatinine_trend_avail"]] == 0 and out.rule_vec[r9, cols["creatinine_trend"]] == 0
    assert out.rule_vec[r11, cols["creatinine_trend_avail"]] == 1
    assert abs(out.rule_vec[r11, cols["creatinine_trend"]] - 0.2) < 1e-9  # (1.4-1.0)/2h
    assert cr >= 0


def test_without_rule_naive_slope_would_read_as_stable(cfg):
    f, _ = _build([ev("S1", "creatinine", 8.0, 1.0), ev("S1", "glucose", 30.0, 100.0)], cfg)
    assert f.slope[_row(f, 9.0), VARS.index("creatinine")] == 0.0  # raw feature: looks "stable"
    out = rules.apply_rules(f, cfg)
    assert out.rule_vec[_row(f, 9.0), rules.RULE_COLS.index("creatinine_trend_avail")] == 0  # rule says unknown


def test_r2_stale_result_exposes_age_and_removes_old_values_from_sequence(cfg):
    f, _ = _build([ev("S1", "glucose", 6.5, 130.0), ev("S1", "glucose", 38.0, 100.0)], cfg)
    out = rules.apply_rules(f, cfg)
    young, old = _row(f, 12.0), _row(f, 20.0)  # glucose age 5.5 h vs 13.5 h; limit 12 h
    assert out.flags.R2_stale_vars.iloc[young] == "" and "glucose" in out.flags.R2_stale_vars.iloc[old]
    assert abs(out.flags.glucose_result_age_h.iloc[old] - 13.5) < 1e-9
    assert out.seq[old, :, GLUCOSE_IDX, 1].sum() > 0  # raw grid still holds the old value
    assert out.seq_r[old, :, GLUCOSE_IDX, 1].sum() == 0  # rule layer removed it
    assert out.seq_r[old, :, GLUCOSE_IDX, 0].sum() == 0
    assert out.seq_r[young, :, GLUCOSE_IDX, 1].sum() == out.seq[young, :, GLUCOSE_IDX, 1].sum()


def test_r3_and_r4_change_review_decision_only_not_model_inputs(cfg):
    f, _ = _build([ev("S1", "glucose", 30.0, 100.0)], cfg)  # no vitals, no glucose before 30
    out = rules.apply_rules(f, cfg)
    assert out.flags.R3_no_glucose_history.iloc[0] and out.flags.R4_vitals_absent.iloc[0]
    assert out.flags.needs_review.iloc[0]
    assert out.rule_vec.shape[1] == len(rules.RULE_COLS)  # no R3/R4 column feeds the model
    assert not any("absent" in c or "no_glucose" in c for c in rules.RULE_COLS)
    assert set(rules.active_rules(out.flags.iloc[0])) >= {"R3_NO_GLUCOSE_HISTORY", "R4_VITALS_ABSENT"}


def test_r4_clears_when_recent_vitals_present(cfg):
    rows = [ev("S1", "hr", h, 80.0) for h in range(0, 12)] + [ev("S1", "glucose", 5.0, 100.0)]
    f, _ = _build(rows, cfg)
    out = rules.apply_rules(f, cfg)
    assert not out.flags.R4_vitals_absent.iloc[_row(f, 8.0)]


def test_decide_routes_review_and_cooldown():
    assert explain.decide(0.1, 0.3, True, False, False) == "NONE"
    assert explain.decide(0.5, 0.3, True, False, False) == "ALERT"
    assert explain.decide(0.5, 0.3, True, True, False) == "QUEUE_REVIEW"
    assert explain.decide(0.5, 0.3, True, False, True) == "QUEUE_REVIEW"
    assert explain.decide(0.5, 0.3, False, False, False) == "NONE"  # suppressed by cooldown


def test_reasoning_disabled_zeroes_rule_features_and_uses_raw_sequence(bundle):
    f = bundle.feats.split("train").labelled().take(np.arange(5))
    d_on = bundle.pre.tensors(f, rules_on=True)
    d_off = bundle.pre.tensors(f, rules_on=False)
    assert np.abs(d_off["rule"]).sum() == 0 and np.abs(d_on["rule"]).sum() > 0
    assert d_off["seq"].shape == d_on["seq"].shape
