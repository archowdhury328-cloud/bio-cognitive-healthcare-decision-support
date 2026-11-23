"""Cohort definition: eligibility, patient-level split, landmark windows, labels, censoring.

Order of operations matters and is enforced here:
  1. ``to_hours``        canonical tables -> hours since ICU admission, validity filters
  2. ``stay_table``      per-stay eligibility and first-excursion time (NO windows yet)
  3. ``assign_splits``   patients -> train/val/test (BEFORE any window exists)
  4. ``make_windows``    hourly landmarks per eligible stay, labelled, tagged with split

Label semantics at landmark t with horizon H (threshold strictly greater than):
  1  an observed qualifying glucose has charttime in (t, t+H]
  0  full horizon is under follow-up, >=1 valid glucose measured in it, none qualifying
 -1  unknown: either follow-up ends inside the horizon ("incomplete_followup") or no glucose
     was measured in it ("no_glucose"). Unknown windows are counted, never trained or scored.
Selective testing: a negative requires a measurement, so labels exist mostly when clinicians
chose to test. Results describe tested windows, not all ICU time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .schema import EVENT_COLUMNS, VALID_RANGES

UNKNOWN_INCOMPLETE = "incomplete_followup"
UNKNOWN_NO_GLUCOSE = "no_glucose"
SPLITS = ("train", "val", "test")


def to_hours(stays: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert timestamps to hours since ICU admission and apply validity filters.

    ``availtime`` is clipped to be no earlier than ``charttime`` (some sources store a result
    before its recorded draw time; a result cannot be known before it is drawn).
    Events recorded after the end of follow-up are dropped.
    """
    st = stays.copy()
    for col in ("intime", "followup_end"):
        st[col] = pd.to_datetime(st[col])
    st["followup_h"] = (st.followup_end - st.intime) / pd.Timedelta(hours=1)
    st = st.drop_duplicates("stay_id").reset_index(drop=True)

    ev = events.reindex(columns=EVENT_COLUMNS).copy()
    ev["charttime"] = pd.to_datetime(ev["charttime"])
    ev["availtime"] = pd.to_datetime(ev["availtime"])
    ev = ev.merge(st[["stay_id", "intime", "followup_h"]], on="stay_id", how="inner")
    ev["t_chart"] = (ev.charttime - ev.intime) / pd.Timedelta(hours=1)
    ev["t_avail"] = (ev.availtime - ev.intime) / pd.Timedelta(hours=1)
    ev["t_avail"] = np.maximum(ev.t_avail, ev.t_chart)
    ev = ev[ev.variable.isin(VALID_RANGES)].copy()
    lo = ev.variable.map(lambda v: VALID_RANGES[v][0])
    hi = ev.variable.map(lambda v: VALID_RANGES[v][1])
    ev = ev[ev.value.notna() & (ev.value >= lo) & (ev.value <= hi)]
    ev = ev[ev.t_chart <= ev.followup_h]
    ev = ev.drop_duplicates(["stay_id", "variable", "t_chart", "value"])
    ev = ev.sort_values(["stay_id", "variable", "t_chart"]).reset_index(drop=True)
    return st, ev[["stay_id", "variable", "t_chart", "t_avail", "value"]]


def stay_table(st: pd.DataFrame, ev: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Per-stay eligibility and first qualifying excursion time. No windows are created here."""
    task = cfg["task"]
    thr, t0 = float(task["threshold_mg_dl"]), float(task["first_landmark_h"])
    g = ev[ev.variable == task["event_variable"]]
    by_stay = {sid: grp for sid, grp in g.groupby("stay_id")}
    rows = []
    for r in st.itertuples():
        grp = by_stay.get(r.stay_id)
        chart = grp.t_chart.to_numpy() if grp is not None else np.array([])
        val = grp.value.to_numpy() if grp is not None else np.array([])
        reason, event_t = "eligible", np.nan
        if r.followup_h <= t0:
            reason = "followup_before_first_landmark"
        elif task.get("history_exclusion", True) and np.any((chart <= t0) & (val > thr)):
            reason = "historical_high"
        else:
            later = (chart > t0) & (val > thr)
            if later.any():
                event_t = float(chart[later][0])
        rows.append({"stay_id": r.stay_id, "patient_id": r.patient_id, "followup_h": r.followup_h,
                     "eligible": reason == "eligible", "exclusion": reason, "event_time_h": event_t,
                     "n_glucose": len(chart)})
    return pd.DataFrame(rows)


def assign_splits(stay_tab: pd.DataFrame, seed: int, fractions: list[float]) -> dict[str, str]:
    """Patient-level split, stratified by whether the patient has an eligible event stay.

    Patients with no eligible stay get no split (they contribute no windows). Deterministic for a
    given seed and patient set.
    """
    elig = stay_tab[stay_tab.eligible]
    has_event = elig.groupby("patient_id").event_time_h.apply(lambda s: bool(s.notna().any()))
    rng = np.random.default_rng(seed)
    out: dict[str, str] = {}
    for stratum in (True, False):
        ids = sorted(has_event[has_event == stratum].index)
        order = rng.permutation(len(ids))
        ids = [ids[i] for i in order]
        n = len(ids)
        n_val, n_test = round(n * fractions[1]), round(n * fractions[2])
        if n >= 3:
            n_val, n_test = max(n_val, 1), max(n_test, 1)
        n_train = n - n_val - n_test
        for k, pid in enumerate(ids):
            out[pid] = "train" if k < n_train else ("val" if k < n_train + n_val else "test")
    return out


def make_windows(st_tab: pd.DataFrame, ev: pd.DataFrame, split_map: dict[str, str],
                 cfg: dict[str, Any]) -> pd.DataFrame:
    """Hourly landmarks for eligible stays whose patient has a split, with labels."""
    task = cfg["task"]
    thr, H = float(task["threshold_mg_dl"]), float(task["horizon_h"])
    t0, step = float(task["first_landmark_h"]), float(task["cadence_h"])
    g_by = {sid: grp for sid, grp in ev[ev.variable == task["event_variable"]].groupby("stay_id")}
    out: list[dict] = []
    for r in st_tab[st_tab.eligible & st_tab.patient_id.isin(split_map)].itertuples():
        grp = g_by.get(r.stay_id)
        chart = grp.t_chart.to_numpy() if grp is not None else np.array([])
        val = grp.value.to_numpy() if grp is not None else np.array([])
        k = 0
        while True:
            t = t0 + k * step
            if t >= r.followup_h or (not np.isnan(r.event_time_h) and t >= r.event_time_h):
                break  # stop scoring at follow-up end and after the first qualifying event
            in_win = (chart > t) & (chart <= t + H)
            label, why = _label(in_win, val, thr, t + H <= r.followup_h)
            out.append({"stay_id": r.stay_id, "patient_id": r.patient_id, "split": split_map[r.patient_id],
                        "t_h": t, "label": label, "unknown_reason": why,
                        "event_time_h": r.event_time_h, "followup_h": r.followup_h})
            k += 1
    cols = ["stay_id", "patient_id", "split", "t_h", "label", "unknown_reason", "event_time_h", "followup_h"]
    return pd.DataFrame(out, columns=cols)


def _label(in_win: np.ndarray, val: np.ndarray, thr: float, horizon_observed: bool) -> tuple[int, str]:
    if in_win.any() and (val[in_win] > thr).any():
        return 1, ""
    if not horizon_observed:
        return -1, UNKNOWN_INCOMPLETE
    if in_win.any():
        return 0, ""
    return -1, UNKNOWN_NO_GLUCOSE


def cohort_flow(st_tab: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Cohort flow table: stays, patients, events and windows per split, plus exclusions."""
    rows: list[dict[str, Any]] = [
        {"stage": "all_stays", "split": "all", "n": len(st_tab)},
        {"stage": "all_patients", "split": "all", "n": int(st_tab.patient_id.nunique())},
    ]
    for reason, n in st_tab[~st_tab.eligible].exclusion.value_counts().items():
        rows.append({"stage": f"excluded:{reason}", "split": "all", "n": int(n)})
    elig = st_tab[st_tab.eligible]
    split_of = windows.drop_duplicates("patient_id").set_index("patient_id").split
    elig = elig.assign(split=elig.patient_id.map(split_of))
    for name, sub, wsub in [("all", elig, windows)] + [(s, elig[elig.split == s], windows[windows.split == s])
                                                      for s in SPLITS]:
        ev_stays = sub[sub.event_time_h.notna()]
        rows += [
            {"stage": "eligible_stays", "split": name, "n": len(sub)},
            {"stage": "eligible_patients", "split": name, "n": int(sub.patient_id.nunique())},
            {"stage": "eligible_event_stays", "split": name, "n": len(ev_stays)},
            {"stage": "eligible_event_patients", "split": name, "n": int(ev_stays.patient_id.nunique())},
            {"stage": "event_free_stays", "split": name, "n": int(len(sub) - len(ev_stays))},
            {"stage": "windows_total", "split": name, "n": len(wsub)},
            {"stage": "windows_positive", "split": name, "n": int((wsub.label == 1).sum())},
            {"stage": "windows_negative", "split": name, "n": int((wsub.label == 0).sum())},
            {"stage": f"windows_unknown:{UNKNOWN_INCOMPLETE}", "split": name,
             "n": int((wsub.unknown_reason == UNKNOWN_INCOMPLETE).sum())},
            {"stage": f"windows_unknown:{UNKNOWN_NO_GLUCOSE}", "split": name,
             "n": int((wsub.unknown_reason == UNKNOWN_NO_GLUCOSE).sum())},
        ]
    return pd.DataFrame(rows)
