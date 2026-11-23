"""Patient splits, timestamp boundaries, unknown labels, censoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from biocog import cohort
from helpers import ev, labels_by_t, stay, windows_for

G = "glucose"


def test_value_exactly_at_threshold_is_not_qualifying(cfg):
    _, _, tab, w = windows_for([stay()], [ev("S1", G, 8.0, 180.0), ev("S1", G, 20.0, 120.0)], cfg)
    assert tab.event_time_h.isna().all()
    assert labels_by_t(w)[6.0] == (0, "")  # measured in window, not > 180


def test_value_just_above_threshold_qualifies_and_sets_event_time(cfg):
    _, _, tab, w = windows_for([stay()], [ev("S1", G, 8.0, 180.01)], cfg)
    assert tab.event_time_h.iloc[0] == 8.0
    lab = labels_by_t(w)
    assert lab[6.0][0] == 1 and lab[7.0][0] == 1
    assert 8.0 not in lab  # scoring stops at the first qualifying event


def test_window_upper_bound_inclusive_lower_bound_exclusive(cfg):
    # landmark 6: window is (6, 12]
    _, _, _, w = windows_for([stay()], [ev("S1", G, 12.0, 190.0)], cfg)
    assert labels_by_t(w)[6.0][0] == 1  # t+H inclusive
    _, _, _, w2 = windows_for([stay()], [ev("S1", G, 6.0, 150.0), ev("S1", G, 20.0, 150.0)], cfg)
    assert labels_by_t(w2)[6.0] == (-1, cohort.UNKNOWN_NO_GLUCOSE)  # draw exactly at t is outside (t, t+H]


def test_historical_high_at_or_before_first_landmark_excludes_stay(cfg):
    _, _, tab, w = windows_for([stay()], [ev("S1", G, 6.0, 200.0)], cfg)
    assert tab.exclusion.iloc[0] == "historical_high" and w.empty
    _, _, tab2, w2 = windows_for([stay()], [ev("S1", G, 6.01, 200.0)], cfg)
    assert tab2.eligible.iloc[0] and labels_by_t(w2) == {6.0: (1, "")}


def test_pre_icu_high_value_excludes_stay(cfg):
    _, _, tab, _ = windows_for([stay()], [ev("S1", G, -5.0, 250.0)], cfg)
    assert tab.exclusion.iloc[0] == "historical_high"


def test_unknown_when_followup_ends_inside_horizon(cfg):
    _, _, _, w = windows_for([stay(followup_h=10.0)], [ev("S1", G, 8.0, 150.0)], cfg)
    lab = labels_by_t(w)
    assert set(lab) == {6.0, 7.0, 8.0, 9.0}  # landmarks stop at follow-up end
    assert all(v == (-1, cohort.UNKNOWN_INCOMPLETE) for v in lab.values())


def test_positive_even_when_horizon_extends_past_followup(cfg):
    _, _, _, w = windows_for([stay(followup_h=10.0)], [ev("S1", G, 9.0, 190.0)], cfg)
    assert [labels_by_t(w)[t][0] for t in (6.0, 7.0, 8.0)] == [1, 1, 1]


def test_negative_needs_full_horizon_and_a_measurement(cfg):
    _, _, _, w = windows_for([stay(followup_h=30.0)], [ev("S1", G, 10.0, 150.0)], cfg)
    lab = labels_by_t(w)
    assert lab[6.0] == (0, "")  # glucose at 10 inside (6,12]
    assert lab[10.0] == (-1, cohort.UNKNOWN_NO_GLUCOSE)  # nothing inside (10,16]
    _, _, _, w2 = windows_for([stay(followup_h=12.0)], [ev("S1", G, 8.0, 150.0)], cfg)
    lab2 = labels_by_t(w2)
    assert lab2[6.0] == (0, "")  # t+H == follow-up end: fully observed
    assert lab2[7.0] == (-1, cohort.UNKNOWN_INCOMPLETE)


def test_followup_shorter_than_first_landmark_is_excluded(cfg):
    _, _, tab, w = windows_for([stay(followup_h=5.0)], [], cfg)
    assert tab.exclusion.iloc[0] == "followup_before_first_landmark" and w.empty


def test_events_after_followup_are_dropped(cfg):
    st, e, tab, _ = windows_for([stay(followup_h=10.0)], [ev("S1", G, 11.0, 300.0)], cfg)
    assert e.empty and tab.event_time_h.isna().all()


def test_availability_never_before_draw_time(cfg):
    st, e, _, _ = windows_for([stay()], [ev("S1", G, 9.0, 150.0, avail_h=8.0)], cfg)
    assert (e.t_avail >= e.t_chart).all()


def test_invalid_values_dropped(cfg):
    _, e, _, _ = windows_for([stay()], [ev("S1", "hr", 3.0, 999.0), ev("S1", "hr", 4.0, 80.0),
                                        ev("S1", "hr", 4.0, float("nan"))], cfg)
    assert list(e.value) == [80.0]


def _two_patient_cohort(cfg, n=30, seed=0):
    rng = np.random.default_rng(seed)
    stays, events = [], []
    for i in range(n):
        pid = f"P{i}"
        for k in range(2 if i % 5 == 0 else 1):  # repeated admissions for some patients
            sid = f"S{i}_{k}"
            stays.append(stay(sid, pid, 30.0))
            if i % 3 == 0:
                events.append(ev(sid, G, 9.0, 200.0))
            else:
                events.append(ev(sid, G, 9.0, 120.0 + rng.normal(0, 5)))
    return stays, events


def test_patient_split_is_disjoint_and_keeps_repeat_admissions_together(cfg):
    stays, events = _two_patient_cohort(cfg)
    st, e, tab, w = windows_for(stays, events, cfg)
    smap = cohort.assign_splits(tab, 42, [0.6, 0.2, 0.2])
    w = cohort.make_windows(tab, e, smap, cfg)
    by_patient = w.groupby("patient_id").split.nunique()
    assert (by_patient == 1).all()
    assert set(w.split) == {"train", "val", "test"}
    multi = w[w.patient_id == "P0"]
    assert multi.stay_id.nunique() == 2 and multi.split.nunique() == 1


def test_split_is_assigned_before_windows_and_is_deterministic(cfg):
    stays, events = _two_patient_cohort(cfg)
    st, e, tab, _ = windows_for(stays, events, cfg)
    a = cohort.assign_splits(tab, 42, [0.6, 0.2, 0.2])
    b = cohort.assign_splits(tab, 42, [0.6, 0.2, 0.2])
    c = cohort.assign_splits(tab, 43, [0.6, 0.2, 0.2])
    assert a == b and a != c
    w = cohort.make_windows(tab, e, a, cfg)
    assert (w.split == w.patient_id.map(a)).all()


def test_split_stratified_so_every_split_has_event_patients(cfg):
    stays, events = _two_patient_cohort(cfg)
    _, _, tab, _ = windows_for(stays, events, cfg)
    smap = cohort.assign_splits(tab, 1, [0.6, 0.2, 0.2])
    ev_p = set(tab[tab.event_time_h.notna()].patient_id)
    assert {smap[p] for p in ev_p} == {"train", "val", "test"}


def test_ineligible_only_patients_get_no_split(cfg):
    _, _, tab, _ = windows_for([stay("S1", "PX", 30.0), stay("S2", "PY", 30.0)],
                               [ev("S1", G, 5.0, 250.0), ev("S2", G, 9.0, 100.0)], cfg)
    smap = cohort.assign_splits(tab, 0, [0.6, 0.2, 0.2])
    assert "PX" not in smap and "PY" in smap


def test_cohort_flow_is_internally_consistent(bundle):
    flow = bundle.flow
    for split in ("all", "train", "val", "test"):
        f = flow[flow.split == split].set_index("stage").n
        unk = f["windows_unknown:incomplete_followup"] + f["windows_unknown:no_glucose"]
        assert f["windows_total"] == f["windows_positive"] + f["windows_negative"] + unk
        assert f["eligible_stays"] == f["eligible_event_stays"] + f["event_free_stays"]
    allf = flow[flow.split == "all"].set_index("stage").n
    parts = flow[flow.split.isin(["train", "val", "test"]) & (flow.stage == "eligible_patients")].n.sum()
    assert parts == allf["eligible_patients"]


def test_cohort_matches_generator_ground_truth(fixture_tables, cfg):
    from biocog.datasets import FIXTURE_DIR

    truth = pd.read_csv(FIXTURE_DIR / "truth.csv").set_index("stay_id")
    st, e = cohort.to_hours(*fixture_tables)
    tab = cohort.stay_table(st, e, cfg).set_index("stay_id")
    # builder never sees ground truth; compare event times and historical-high exclusions
    ev_truth = truth.event_time_h
    both = tab.eligible & ev_truth.reindex(tab.index).notna()
    assert np.allclose(tab.event_time_h[both], ev_truth.reindex(tab.index)[both], atol=1e-6)
    assert ((tab.exclusion == "historical_high") == truth.historical_high.reindex(tab.index)).all()
    assert (tab.event_time_h.notna() == (ev_truth.reindex(tab.index).notna() & tab.eligible)).all()
