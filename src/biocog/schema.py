"""Canonical variable lists, units and plausibility ranges shared by every adapter."""

from __future__ import annotations

VITALS: tuple[str, ...] = ("hr", "sbp", "rr", "spo2", "temp_c")
BIOMARKERS: tuple[str, ...] = ("glucose", "creatinine", "sodium", "potassium", "wbc", "bun")
VARS: tuple[str, ...] = VITALS + BIOMARKERS
N_VARS = len(VARS)
VITAL_IDX = tuple(range(len(VITALS)))
BIO_IDX = tuple(range(len(VITALS), N_VARS))
GLUCOSE = "glucose"
GLUCOSE_IDX = VARS.index(GLUCOSE)
STATIC: tuple[str, ...] = ("age", "sex_male", "emergency")

UNITS: dict[str, str] = {
    "hr": "bpm", "sbp": "mmHg", "rr": "breaths/min", "spo2": "%", "temp_c": "degC",
    "glucose": "mg/dL", "creatinine": "mg/dL", "sodium": "mEq/L", "potassium": "mEq/L",
    "wbc": "K/uL", "bun": "mg/dL",
}

# Values outside these ranges are treated as recording errors and dropped by adapters.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "hr": (20, 250), "sbp": (40, 300), "rr": (4, 70), "spo2": (50, 100), "temp_c": (30, 43),
    "glucose": (20, 1500), "creatinine": (0.1, 20), "sodium": (100, 180), "potassium": (1.5, 9),
    "wbc": (0.1, 200), "bun": (1, 250),
}

STAY_COLUMNS = ["stay_id", "patient_id", "intime", "followup_end", "age", "sex_male", "emergency"]
EVENT_COLUMNS = ["stay_id", "variable", "charttime", "availtime", "value"]
