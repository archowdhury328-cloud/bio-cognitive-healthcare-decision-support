"""Deterministic synthetic ICU generator. Everything it produces is SYNTHETIC.

It writes the same canonical tables as the MIMIC adapter: ``stays`` and ``events``.
The generator also returns ground truth (true first-excursion time, censoring flags)
so tests can check the cohort builder against knowledge the builder never sees.

Design features requested by the task: repeated observations, absent modalities,
delayed lab results, known event times, event-free stays, incomplete follow-up,
selective testing (glucose is measured more often when the latent level is high).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import VALID_RANGES

BASE_TIME = pd.Timestamp("2100-01-01 00:00:00")
THRESHOLD = 180.0


def _clip(name: str, x: float) -> float:
    lo, hi = VALID_RANGES[name]
    return float(min(max(x, lo), hi))


def generate(n_patients: int = 80, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(stays, events, truth)``; all synthetic."""
    rng = np.random.default_rng(seed)
    stays, events, truth = [], [], []
    stay_counter = 0
    for p in range(n_patients):
        pid = f"P{p:04d}"
        age = float(np.clip(rng.normal(63, 15), 18, 95))
        sex = int(rng.random() < 0.55)
        emergency = int(rng.random() < 0.6)
        diabetic_like = bool(rng.random() < 0.3)
        ckd_like = bool(rng.random() < 0.15 + 0.1 * (age > 70))
        offset_days = 0.0
        for _ in range(1 + int(rng.random() < 0.12)):
            stay_counter += 1
            sid = f"S{stay_counter:05d}"
            intime = BASE_TIME + pd.Timedelta(days=p * 20 + offset_days)
            offset_days += 8
            los_h = float(np.clip(rng.lognormal(np.log(52), 0.55), 14, 180))
            censored = bool(rng.random() < 0.15)
            followup_h = los_h * rng.uniform(0.4, 0.8) if censored else los_h
            followup_h = max(followup_h, 8.0)
            rows, t_row = _simulate_stay(rng, sid, intime, los_h, diabetic_like, ckd_like, age, followup_h)
            events.extend(rows)
            stays.append({
                "stay_id": sid, "patient_id": pid, "intime": intime,
                "followup_end": intime + pd.Timedelta(hours=followup_h),
                "age": round(age, 1), "sex_male": sex, "emergency": emergency,
            })
            truth.append({"stay_id": sid, "patient_id": pid, "followup_h": followup_h,
                          "incomplete_followup": censored, **t_row})
    return pd.DataFrame(stays), pd.DataFrame(events), pd.DataFrame(truth)


def _simulate_stay(rng, sid, intime, los_h, diabetic_like, ckd_like, age, followup_h):
    hours = np.arange(0, int(np.ceil(los_h)) + 1)
    n = len(hours)
    base = rng.normal(165 if diabetic_like else 115, 14)
    stress = np.zeros(n)
    glucose = np.zeros(n)
    g = base + rng.normal(0, 8)
    ramp_left, ramp_slope = 0, 0.0
    start_high = bool(rng.random() < 0.06)  # historical high-value exclusion cases
    for k in range(n):
        if ramp_left == 0 and rng.random() < (0.025 if diabetic_like else 0.012):
            ramp_left = int(rng.integers(4, 9))
            ramp_slope = rng.uniform(7, 15)
        if ramp_left > 0:
            g += ramp_slope + rng.normal(0, 3)
            stress[k] = min(1.0, stress[max(k - 1, 0)] + 0.2)
            ramp_left -= 1
        else:
            g += 0.25 * (base - g) + rng.normal(0, 5)
            stress[k] = max(0.0, stress[max(k - 1, 0)] - 0.12)
        if start_high and k < 4:
            g = max(g, 195 + rng.normal(0, 5))
        glucose[k] = g
    rows: list[dict] = []

    def add(var: str, t_chart_h: float, value: float, lag_min: float) -> None:
        ct = intime + pd.Timedelta(hours=float(t_chart_h))
        rows.append({"stay_id": sid, "variable": var, "charttime": ct,
                     "availtime": ct + pd.Timedelta(minutes=float(lag_min)),
                     "value": round(_clip(var, value), 2)})

    vit_missing_until = rng.uniform(0, 14) if rng.random() < 0.2 else 0.0
    vit_absent = bool(rng.random() < 0.05)
    for k in range(n):
        if vit_absent or hours[k] < vit_missing_until or hours[k] > followup_h:
            continue
        s = stress[k]
        add("hr", hours[k], 78 + 18 * s + rng.normal(0, 7) - 0.1 * (age - 60), 0)
        add("sbp", hours[k], 122 - 12 * s + rng.normal(0, 9), 0)
        add("rr", hours[k], 16 + 5 * s + rng.normal(0, 2), 0)
        add("spo2", hours[k], 97 - 2 * s + rng.normal(0, 1.2), 0)
        add("temp_c", hours[k], 36.8 + 0.6 * s + rng.normal(0, 0.25), 0)

    lab_absent = bool(rng.random() < 0.08)
    last_g_draw = -99.0
    for k in range(n):
        if hours[k] > followup_h:
            continue
        interval = 2.0 if glucose[k] > 150 else 6.0  # selective testing
        if hours[k] - last_g_draw >= interval and rng.random() < 0.85:
            lag = rng.uniform(20, 90) if rng.random() > 0.12 else rng.uniform(180, 360)
            t_draw = max(0.0, hours[k] + rng.uniform(-0.3, 0.3))
            add("glucose", t_draw, glucose[k] + rng.normal(0, 4), lag)
            last_g_draw = hours[k]
        if lab_absent or k % 12 != 5:
            continue
        lag = rng.uniform(40, 140)
        add("creatinine", hours[k], (2.1 if ckd_like else 0.95) + rng.normal(0, 0.12) + 0.05 * k / 24, lag)
        add("sodium", hours[k], 139 + rng.normal(0, 2.5), lag)
        add("potassium", hours[k], 4.2 + rng.normal(0, 0.4), lag)
        add("wbc", hours[k], 9 + 4 * stress[k] + rng.normal(0, 2), lag)
        add("bun", hours[k], (32 if ckd_like else 17) + rng.normal(0, 4), lag)

    df = pd.DataFrame(rows)
    g_rows = df[df.variable == "glucose"].sort_values("charttime")
    first6 = intime + pd.Timedelta(hours=6)
    hist = g_rows[(g_rows.charttime <= first6) & (g_rows.value > THRESHOLD)]
    after = g_rows[(g_rows.charttime > first6) & (g_rows.value > THRESHOLD)]
    ev_t = np.nan
    if len(after) and len(hist) == 0:
        cand = float((after.iloc[0].charttime - intime) / pd.Timedelta(hours=1))
        ev_t = cand if cand <= followup_h else np.nan  # excursion after follow-up ended is unobserved
    return rows, {"historical_high": bool(len(hist)), "event_time_h": ev_t}
