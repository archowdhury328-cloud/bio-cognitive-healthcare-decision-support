"""Versioned, auditable cognitive-reasoning rule layer (a deterministic rule engine).

This is NOT a cognitive architecture. It is a small set of explicit rules, each with a stable id,
a documented trigger and a documented effect, so every change to a feature, confidence flag or
review decision can be traced to a rule.

  R1_TREND_MIN_OBS    fewer than ``min_obs_trend`` biomarker values in the trend window
                      -> trend is UNAVAILABLE (feature set to 0 with trend_avail=0).
                      Without the rule a naive slope of 0.0 would silently look like "stable".
  R2_STALE_RESULT     observations older than ``stale_hours[var]`` are removed from the model's
                      sequence input (value and mask zeroed); result age is exposed as a feature.
  R3_NO_GLUCOSE_HISTORY  no glucose result is available yet -> confidence flag and REVIEW decision.
  R4_VITALS_ABSENT    no vital sign in the last ``vitals_absent_window_h`` hours -> REVIEW decision.

R1 and R2 change model inputs. R3 and R4 change only the review decision (they never change risk).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .features import Features
from .schema import BIO_IDX, GLUCOSE_IDX, VARS, VITAL_IDX

RULESET_VERSION = "rules-1.0"
RULE_DOCS = {
    "R1_TREND_MIN_OBS": "trend unavailable when < min_obs_trend biomarker values in trend window (changes feature)",
    "R2_STALE_RESULT": "observations older than stale_hours[var] dropped from sequence input; age exposed (changes feature)",
    "R3_NO_GLUCOSE_HISTORY": "no glucose result available -> low-confidence flag, review decision (no feature change)",
    "R4_VITALS_ABSENT": "no vitals in recent window -> low-confidence flag, review decision (no feature change)",
}
BIO_NAMES = [VARS[i] for i in BIO_IDX]
RULE_COLS = [f"{n}_{k}" for n in BIO_NAMES for k in ("trend", "trend_avail", "stale", "age_frac")]


def apply_rules(f: Features, cfg: dict[str, Any]) -> Features:
    """Return a new Features with ``seq_r``, ``rule_vec`` and ``flags`` populated."""
    rc = cfg["rules"]
    cap = float(cfg["features"]["gap_cap_h"])
    T = f.seq.shape[1]
    seq_r = f.seq.copy()
    n = len(f)
    cols: list[np.ndarray] = []
    stale_names: list[list[str]] = [[] for _ in range(n)]
    n_trend_unavail = np.zeros(n, dtype=int)
    for vi in BIO_IDX:
        name = VARS[vi]
        avail = f.n_trend[:, vi] >= rc["min_obs_trend"]
        trend = np.where(avail, f.slope[:, vi], 0.0)
        limit = float(rc["stale_hours"][name])
        age = f.age_h[:, vi]
        stale = np.isnan(age) | (age > limit)
        n_trend_unavail += (~avail).astype(int)
        for r in np.where(stale & ~np.isnan(age))[0]:
            stale_names[r].append(name)
        # bins are chronological; bin j is (T-1-j .. T-j] hours before t -> drop bins older than limit
        old = np.array([(T - 1 - j) >= limit for j in range(T)])
        seq_r[:, old, vi, 0] = 0.0
        seq_r[:, old, vi, 1] = 0.0
        cols += [trend, avail.astype(float), stale.astype(float),
                 np.where(np.isnan(age), 1.0, np.minimum(age, cap) / cap)]
    recent = int(rc["vitals_absent_window_h"])
    vit_recent = f.seq[:, T - recent:, :, 1][:, :, list(VITAL_IDX)].sum((1, 2))
    vitals_absent = (vit_recent == 0)
    no_glu = f.n_obs[:, GLUCOSE_IDX] == 0
    flags = pd.DataFrame({
        "R1_trend_unavailable_vars": n_trend_unavail,
        "R2_stale_vars": [",".join(s) for s in stale_names],
        "R3_no_glucose_history": no_glu,
        "R4_vitals_absent": vitals_absent,
        "glucose_result_age_h": f.age_h[:, GLUCOSE_IDX],
    })
    flags["needs_review"] = flags.R3_no_glucose_history | flags.R4_vitals_absent
    return replace(f, seq_r=seq_r, rule_vec=np.stack(cols, axis=1), flags=flags)


def active_rules(flags_row: pd.Series) -> list[str]:
    """Human-readable list of rules that fired for one landmark."""
    out = []
    if flags_row.R1_trend_unavailable_vars > 0:
        out.append(f"R1_TREND_MIN_OBS ({int(flags_row.R1_trend_unavailable_vars)} biomarker trends unavailable)")
    if flags_row.R2_stale_vars:
        out.append(f"R2_STALE_RESULT ({flags_row.R2_stale_vars})")
    if flags_row.R3_no_glucose_history:
        out.append("R3_NO_GLUCOSE_HISTORY")
    if flags_row.R4_vitals_absent:
        out.append("R4_VITALS_ABSENT")
    return out
