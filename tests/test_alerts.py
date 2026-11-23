"""Lead-time arithmetic, missed-event denominators, alert suppression, burden, paired comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biocog import alerts


def _stay(sid, ts, labels, scores, event=np.nan):
    return pd.DataFrame({"stay_id": sid, "patient_id": sid, "t_h": ts, "label": labels,
                         "event_time_h": event, "score": scores})


def _run(df, thr=0.5, cd=0.0):
    return alerts.evaluate_alerts(df, df.score.to_numpy(), thr, cd)


def test_lead_time_is_event_time_minus_first_true_alert():
    df = _stay("A", [6, 7, 8, 9], [1, 1, 1, 1], [0.1, 0.9, 0.9, 0.9], event=10.0)
    r = _run(df)
    assert r["detected_events"] == 1 and r["lead_times_h"] == {"A": 3.0} and r["lead_time_median_h"] == 3.0


def test_missed_event_stays_in_denominator():
    ev_stay = _stay("A", [6, 7, 8, 9], [1, 1, 1, 1], [0.1, 0.1, 0.1, 0.1], event=10.0)
    free = _stay("B", [6, 7, 8, 9], [0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1])
    r = _run(pd.concat([ev_stay, free]))
    assert (r["eligible_events"], r["detected_events"], r["missed_events"]) == (1, 0, 1)
    assert r["detection_rate"] == 0.0 and np.isnan(r["lead_time_median_h"])


def test_event_free_stays_do_not_count_as_events():
    r = _run(_stay("B", [6, 7], [0, 0], [0.9, 0.9]))
    assert r["eligible_events"] == 0 and np.isnan(r["detection_rate"])


def test_cooldown_suppresses_repeat_alerts_then_releases():
    df = _stay("A", [6, 7, 8, 9, 10, 11], [0] * 6, [0.9] * 6)
    assert alerts.apply_policy(df, df.score.to_numpy(), 0.5, 3).tolist() == [True, False, False, True, False, False]
    assert alerts.apply_policy(df, df.score.to_numpy(), 0.5, 0).all()


def test_early_alert_spends_cooldown_and_can_cause_a_miss():
    ts = list(range(6, 14))  # event at 14; labels 1 once t >= 8
    labels = [0, 0, 1, 1, 1, 1, 1, 1]
    df = _stay("A", ts, labels, [0.9] * 8, event=14.0)
    assert _run(df, cd=0.0)["lead_times_h"] == {"A": 6.0}  # first true alert at t=8
    r6 = _run(df, cd=6.0)  # alert at 6 (false), next allowed at 12 -> lead 2
    assert r6["lead_times_h"] == {"A": 2.0} and r6["false_alerts"] == 1
    assert _run(df, cd=8.0)["missed_events"] == 1  # next allowed at 14: no landmark left


def test_unknown_label_alerts_reported_separately_and_burden_per_patient_day():
    df = _stay("A", list(range(24)), [0] * 12 + [-1] * 12, [0.9] * 24)
    r = _run(df, cd=0.0)
    assert r["false_alerts"] == 12 and r["unconfirmed_alerts"] == 12 and r["true_alerts"] == 0
    assert r["monitored_patient_days"] == pytest.approx(1.0)
    assert r["false_alerts_per_patient_day"] == pytest.approx(12.0)
    assert r["unconfirmed_alerts_per_patient_day"] == pytest.approx(12.0)


def test_policy_is_per_stay_and_order_independent():
    a = _stay("A", [6, 7], [0, 0], [0.9, 0.9])
    b = _stay("B", [6, 7], [0, 0], [0.9, 0.9])
    df = pd.concat([a, b]).sample(frac=1, random_state=0).reset_index(drop=True)
    out = alerts.apply_policy(df, df.score.to_numpy(), 0.5, 3)
    assert out.sum() == 2 and set(df.stay_id[out]) == {"A", "B"} and (df.t_h[out] == 6).all()


def test_threshold_respects_false_alert_budget_and_budget_monotone():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        has = i % 3 == 0
        ts = list(range(6, 20))
        lab = [1 if (has and t >= 14) else 0 for t in ts]
        base = np.where(np.array(lab) == 1, 0.55, 0.3) + rng.normal(0, 0.15, len(ts))
        rows.append(_stay(f"S{i}", ts, lab, np.clip(base, 0, 1), event=20.0 if has else np.nan))
    df = pd.concat(rows, ignore_index=True)
    sc = df.score.to_numpy()
    detected = []
    for budget in (0.2, 1.0, 4.0):
        thr, res = alerts.choose_threshold(df, sc, budget, 3.0)
        assert res["false_alerts_per_patient_day"] <= budget + 1e-9
        detected.append(res["detected_events"])
    assert detected == sorted(detected)  # a bigger budget never detects fewer events


def test_impossible_budget_yields_no_alerts_reported_honestly():
    df = _stay("A", [6, 7, 8], [0, 0, 0], [0.9, 0.9, 0.9])
    thr, res = alerts.choose_threshold(df, df.score.to_numpy(), 0.0, 0.0)
    assert res["n_alerts"] == 0 and thr > 0.9


def test_select_policy_shared_cooldown_tie_goes_to_smaller():
    df = _stay("A", [6, 7, 8, 9], [1, 1, 1, 1], [0.9, 0.9, 0.9, 0.9], event=10.0)
    cd, thr = alerts.select_policy({"m": (df, df.score.to_numpy())}, 100.0, [0, 3, 6])
    assert cd == 0 and "m" in thr


def test_paired_detection_partitions_events():
    out = alerts.paired_detection({"a", "b"}, {"b", "c"}, {"a", "b", "c", "d"})
    assert out == {"both": 1, "only_a": 1, "only_b": 1, "neither": 1}
