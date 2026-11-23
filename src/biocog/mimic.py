"""Adapter: MIMIC-IV Clinical Database Demo v2.2 -> canonical ``stays`` / ``events`` tables.

Joins (all by the dataset's own keys; no cross-dataset patient matching):
  icu/icustays      stay_id, subject_id, hadm_id, intime, outtime      (one row per ICU stay)
  hosp/patients     gender, anchor_age, anchor_year                    (by subject_id)
  hosp/admissions   admission_type, admittime                          (by hadm_id)
  icu/chartevents   vital signs, joined by stay_id                     (itemids below)
  hosp/labevents    laboratory results, joined by hadm_id              (itemids below)

Availability timestamps
  labs:   availtime = storetime (when the result was recorded); charttime = specimen time
  charts: availtime = max(charttime, storetime). MIMIC sometimes stores a chart value *before* its
          charttime; a value cannot be known before it is measured, so charttime is the floor.

Lab rows with no hadm_id (outpatient / unlinked) are excluded: they cannot be tied to an ICU stay
by a key, and matching them by patient and time would be an inference, not a join.

Units (verified, not assumed): non-conforming unit strings are dropped and counted in the report.
Glucose uses a SINGLE source, itemid 50931 (Chemistry glucose, mg/dL); blood-gas glucose (50809,
52027) is not mixed in. Systolic BP combines non-invasive (220179) and arterial (220050) readings.
Temperature combines Celsius (223762) and Fahrenheit (223761, converted).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import UNITS, VALID_RANGES

CHART_ITEMS = {220045: "hr", 220179: "sbp", 220050: "sbp", 220210: "rr", 220277: "spo2",
               223762: "temp_c", 223761: "temp_c"}
LAB_ITEMS = {50931: "glucose", 50912: "creatinine", 50983: "sodium", 50971: "potassium",
             51301: "wbc", 51006: "bun"}
EMERGENCY_TYPES = {"EMERGENCY", "URGENT", "EW EMER.", "DIRECT EMER."}


def to_canonical(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = Path(root)
    icu = pd.read_csv(root / "icu" / "icustays.csv.gz", parse_dates=["intime", "outtime"])
    pat = pd.read_csv(root / "hosp" / "patients.csv.gz")
    adm = pd.read_csv(root / "hosp" / "admissions.csv.gz", parse_dates=["admittime"])
    rep: dict[str, Any] = {"icu_stays_in_source": len(icu), "patients_in_source": int(icu.subject_id.nunique())}

    st = icu.merge(pat[["subject_id", "gender", "anchor_age", "anchor_year"]], on="subject_id", how="left") \
            .merge(adm[["hadm_id", "admission_type", "admittime"]], on="hadm_id", how="left")
    st["age"] = st.anchor_age + (st.intime.dt.year - st.anchor_year)
    st["age"] = st.age.clip(lower=18, upper=120)
    stays = pd.DataFrame({
        "stay_id": st.stay_id.astype(str), "patient_id": st.subject_id.astype(str),
        "intime": st.intime, "followup_end": st.outtime, "age": st.age,
        "sex_male": (st.gender == "M").astype(int),
        "emergency": st.admission_type.isin(EMERGENCY_TYPES).astype(int)})
    rep["stays_missing_demographics"] = int(stays[["age"]].isna().any(axis=1).sum())
    stays = stays.dropna(subset=["age"])

    vit = _chart_events(root, icu, rep)
    lab = _lab_events(root, icu, rep)
    events = pd.concat([vit, lab], ignore_index=True)
    events["stay_id"] = events.stay_id.astype(str)
    events = events[events.stay_id.isin(stays.stay_id)]
    rep["events_per_variable"] = {k: int(v) for k, v in events.variable.value_counts().items()}
    has = events.groupby("stay_id").variable.agg(set)
    rep["stays_with_any_vitals"] = int(sum(bool(s & {"hr", "sbp", "rr", "spo2", "temp_c"}) for s in has))
    rep["stays_with_any_labs"] = int(sum(bool(s & set(LAB_ITEMS.values())) for s in has))
    rep["stays_with_glucose"] = int(sum("glucose" in s for s in has))
    rep["units"] = UNITS
    return stays.reset_index(drop=True), events.reset_index(drop=True), rep


def _chart_events(root: Path, icu: pd.DataFrame, rep: dict[str, Any]) -> pd.DataFrame:
    ce = pd.read_csv(root / "icu" / "chartevents.csv.gz",
                     usecols=["stay_id", "itemid", "charttime", "storetime", "valuenum"],
                     parse_dates=["charttime", "storetime"])
    ce = ce[ce.itemid.isin(CHART_ITEMS) & ce.stay_id.isin(icu.stay_id) & ce.valuenum.notna()].copy()
    ce["variable"] = ce.itemid.map(CHART_ITEMS)
    f = ce.itemid == 223761
    ce.loc[f, "valuenum"] = (ce.loc[f, "valuenum"] - 32.0) * 5.0 / 9.0
    ce["availtime"] = ce[["charttime", "storetime"]].max(axis=1)
    rep["chart_rows_before_range_filter"] = len(ce)
    out = _range_filter(ce.rename(columns={"valuenum": "value"}), rep, "chart")
    return out[["stay_id", "variable", "charttime", "availtime", "value"]]


def _lab_events(root: Path, icu: pd.DataFrame, rep: dict[str, Any]) -> pd.DataFrame:
    lab = pd.read_csv(root / "hosp" / "labevents.csv.gz",
                      usecols=["hadm_id", "itemid", "charttime", "storetime", "valuenum", "valueuom"],
                      parse_dates=["charttime", "storetime"])
    lab = lab[lab.itemid.isin(LAB_ITEMS)].copy()
    rep["lab_rows_total_selected_items"] = len(lab)
    rep["lab_rows_without_hadm_id_excluded"] = int(lab.hadm_id.isna().sum())
    lab = lab[lab.hadm_id.notna() & lab.valuenum.notna()].copy()
    lab["hadm_id"] = lab.hadm_id.astype(int)
    lab["variable"] = lab.itemid.map(LAB_ITEMS)
    bad_unit = lab.valueuom != lab.variable.map(UNITS)
    rep["lab_rows_dropped_nonconforming_unit"] = int(bad_unit.sum())
    lab = lab[~bad_unit]
    lab["availtime"] = lab.storetime.fillna(lab.charttime)
    lab["availtime"] = lab[["charttime", "availtime"]].max(axis=1)
    # one hadm may contain several ICU stays: attach each lab to the stay whose hadm it belongs to
    j = lab.merge(icu[["stay_id", "hadm_id"]], on="hadm_id", how="inner")
    rep["lab_rows_attached_to_icu_stays"] = len(j)
    out = _range_filter(j.rename(columns={"valuenum": "value"}), rep, "lab")
    return out[["stay_id", "variable", "charttime", "availtime", "value"]]


def _range_filter(df: pd.DataFrame, rep: dict[str, Any], tag: str) -> pd.DataFrame:
    lo = df.variable.map(lambda v: VALID_RANGES[v][0])
    hi = df.variable.map(lambda v: VALID_RANGES[v][1])
    keep = (df.value >= lo) & (df.value <= hi) & np.isfinite(df.value)
    rep[f"{tag}_rows_dropped_out_of_range"] = int((~keep).sum())
    return df[keep]
