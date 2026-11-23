"""Hand-built canonical tables so tests control every timestamp exactly."""

from __future__ import annotations

import pandas as pd

from biocog import cohort

BASE = pd.Timestamp("2100-01-01")


def T(h: float) -> pd.Timestamp:
    return BASE + pd.Timedelta(hours=h)


def stay(sid="S1", pid="P1", followup_h=30.0, age=60, sex=1, emerg=1):
    return {"stay_id": sid, "patient_id": pid, "intime": T(0), "followup_end": T(followup_h),
            "age": age, "sex_male": sex, "emergency": emerg}


def ev(sid, var, chart_h, value, avail_h=None):
    return {"stay_id": sid, "variable": var, "charttime": T(chart_h),
            "availtime": T(chart_h if avail_h is None else avail_h), "value": value}


def windows_for(stays, events, cfg, split="train"):
    """Run the cohort pipeline on hand-built rows; every eligible patient goes to ``split``."""
    st, e = cohort.to_hours(pd.DataFrame(stays), pd.DataFrame(events))
    tab = cohort.stay_table(st, e, cfg)
    smap = {p: split for p in tab[tab.eligible].patient_id}
    return st, e, tab, cohort.make_windows(tab, e, smap, cfg)


def labels_by_t(windows, sid="S1"):
    w = windows[windows.stay_id == sid]
    return {float(t): (int(lab), why) for t, lab, why in zip(w.t_h, w.label, w.unknown_reason, strict=True)}
