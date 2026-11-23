"""One alert-matching and cooldown policy shared by every model.

Policy: at each scored landmark, alert if risk >= threshold and at least ``cooldown_h`` hours have
passed since the stay's previous alert. Scoring stops after the first qualifying event (landmarks
beyond it do not exist), so no alert can follow an event.

Matching: the alert at landmark ``t`` is a TRUE alert iff its window label is 1, i.e. an observed
qualifying glucose has charttime in (t, t+H]. It is a FALSE alert iff label is 0. Alerts on
unknown-label windows are reported separately as UNCONFIRMED (never silently counted as true or false).

Warning lead time for a detected event = event_time - time of the first TRUE alert, in hours.
An event is DETECTED if at least one true alert fires. Cooldown can suppress an in-horizon alert after
an earlier alert, which is deliberate: an alert fired too early to count still spends the cooldown.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def apply_policy(df: pd.DataFrame, score: np.ndarray, thr: float, cooldown_h: float) -> np.ndarray:
    """Boolean alert vector aligned with ``df`` rows (any row order)."""
    alert = np.zeros(len(df), dtype=bool)
    order = np.lexsort((df.t_h.to_numpy(), df.stay_id.to_numpy()))
    last_stay, last_t = None, -np.inf
    sid, t = df.stay_id.to_numpy(), df.t_h.to_numpy()
    for i in order:
        if sid[i] != last_stay:
            last_stay, last_t = sid[i], -np.inf
        if score[i] >= thr and t[i] - last_t >= cooldown_h:
            alert[i] = True
            last_t = t[i]
    return alert


def evaluate_alerts(df: pd.DataFrame, score: np.ndarray, thr: float, cooldown_h: float,
                    cadence_h: float = 1.0) -> dict[str, Any]:
    """Event-level and burden metrics for one model on one cohort slice (all scored landmarks)."""
    alert = apply_policy(df, score, thr, cooldown_h)
    lab = df.label.to_numpy()
    ev_stays = df[df.event_time_h.notna()].drop_duplicates("stay_id")
    detected, lead = {}, {}
    for sid, e in zip(ev_stays.stay_id, ev_stays.event_time_h, strict=True):
        m = (df.stay_id.to_numpy() == sid) & alert & (lab == 1)
        if m.any():
            t_first = float(df.t_h.to_numpy()[m].min())
            detected[sid], lead[sid] = True, float(e - t_first)
    n_ev = len(ev_stays)
    days = len(df) * cadence_h / 24.0
    leads = np.array(list(lead.values()), float)
    return {
        "eligible_events": n_ev, "detected_events": len(detected), "missed_events": n_ev - len(detected),
        "detection_rate": len(detected) / n_ev if n_ev else np.nan,
        "lead_time_median_h": float(np.median(leads)) if len(leads) else np.nan,
        "lead_time_q25_h": float(np.percentile(leads, 25)) if len(leads) else np.nan,
        "lead_time_q75_h": float(np.percentile(leads, 75)) if len(leads) else np.nan,
        "n_alerts": int(alert.sum()), "true_alerts": int((alert & (lab == 1)).sum()),
        "false_alerts": int((alert & (lab == 0)).sum()), "unconfirmed_alerts": int((alert & (lab == -1)).sum()),
        "monitored_patient_days": days,
        "false_alerts_per_patient_day": float((alert & (lab == 0)).sum() / days) if days else np.nan,
        "unconfirmed_alerts_per_patient_day": float((alert & (lab == -1)).sum() / days) if days else np.nan,
        "threshold": float(thr), "cooldown_h": float(cooldown_h),
        "detected_stay_ids": sorted(detected), "lead_times_h": {k: lead[k] for k in sorted(lead)},
    }


def choose_threshold(df: pd.DataFrame, score: np.ndarray, budget: float, cooldown_h: float,
                     cadence_h: float = 1.0) -> tuple[float, dict[str, Any]]:
    """Validation-only search: among thresholds meeting the false-alert budget, maximise detection.

    Ties are broken toward the higher (more conservative) threshold. If none meets the budget the
    threshold is set above every score (no alerts), which is reported honestly as zero detection.
    """
    cands = np.unique(np.quantile(score, np.linspace(0.5, 0.999, 120)))
    best_thr, best_res = float(score.max()) + 1e-9, None
    for thr in cands[::-1]:
        res = evaluate_alerts(df, score, float(thr), cooldown_h, cadence_h)
        if res["false_alerts_per_patient_day"] <= budget and (
                best_res is None or res["detected_events"] > best_res["detected_events"]):
            best_thr, best_res = float(thr), res
    if best_res is None:
        best_res = evaluate_alerts(df, score, best_thr, cooldown_h, cadence_h)
    return best_thr, best_res


def select_policy(val_by_model: dict[str, tuple[pd.DataFrame, np.ndarray]], budget: float,
                  cooldown_grid: list[float], cadence_h: float = 1.0) -> tuple[float, dict[str, float]]:
    """Pick ONE shared cooldown on validation data, then each model's threshold.

    The cooldown maximising mean detection rate across models wins; ties go to the smaller cooldown.
    """
    best = None
    for cd in cooldown_grid:
        thr, rates = {}, []
        for name, (df, sc) in val_by_model.items():
            t, res = choose_threshold(df, sc, budget, cd, cadence_h)
            thr[name] = t
            rates.append(0.0 if np.isnan(res["detection_rate"]) else res["detection_rate"])
        key = float(np.mean(rates))
        if best is None or key > best[0] + 1e-12:
            best = (key, cd, thr)
    assert best is not None
    return best[1], best[2]


def paired_detection(det_a: set[str], det_b: set[str], all_events: set[str]) -> dict[str, int]:
    """Which events both detected, only A, only B, or neither."""
    return {"both": len(det_a & det_b), "only_a": len(det_a - det_b), "only_b": len(det_b - det_a),
            "neither": len(all_events - det_a - det_b)}
