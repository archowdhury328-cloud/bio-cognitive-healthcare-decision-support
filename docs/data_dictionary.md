# Data dictionary (canonical tables)

Every dataset adapter, including the synthetic generator, writes the same two tables. Timestamps are ISO 8601.

## `stays` (one row per ICU stay)

| column | type | meaning |
|---|---|---|
| `stay_id` | string | unique ICU stay id |
| `patient_id` | string | patient id; the unit of splitting (repeated stays share it) |
| `intime` | datetime | ICU admission; hour 0 of every time axis |
| `followup_end` | datetime | end of observed follow-up (ICU discharge/transfer/censoring). Nothing after it is observed |
| `age` | float | years |
| `sex_male` | 0/1 | |
| `emergency` | 0/1 | emergency/urgent admission type |

## `events` (long format, one row per observation)

| column | type | meaning |
|---|---|---|
| `stay_id` | string | |
| `variable` | string | one of `hr sbp rr spo2 temp_c glucose creatinine sodium potassium wbc bun` |
| `charttime` | datetime | when the measurement was taken / specimen drawn |
| `availtime` | datetime | when the value became **known**. Models may only use a value at landmark `t` if `availtime <= t`. Floored at `charttime` |
| `value` | float | in the unit below; out-of-range values are dropped by adapters |

Units: hr bpm; sbp mmHg; rr breaths/min; spo2 %; temp_c degC; glucose, creatinine, bun mg/dL; sodium, potassium mEq/L; wbc K/uL. Glucose comes from a **single** source (MIMIC itemid 50931, chemistry, mg/dL).

## `truth.csv` (synthetic fixture only)

Generator ground truth, never read by the cohort builder; used by tests only: `stay_id, patient_id, followup_h, incomplete_followup, historical_high, event_time_h` (`NaN` = event-free or excluded).

## Derived: landmark windows

One row per (stay, hourly landmark `t`): `stay_id, patient_id, split, t_h, label, unknown_reason, event_time_h, followup_h`.

- `label = 1`: an observed glucose > 180 has `charttime` in `(t, t+6h]`.
- `label = 0`: the full horizon is under follow-up, at least one valid glucose was measured in it, none > 180.
- `label = -1`: unknown. `unknown_reason` is `incomplete_followup` or `no_glucose`. Never trained on, never counted as true or false in metrics; alerts on such windows are reported separately as "unconfirmed".

## Fixture (`data/fixture/`)

120 synthetic patients (seed 7): repeated observations, absent vitals/biomarkers, results released after the draw (delays of ~20 min to 6 h), event-free stays, stays with an excursion, stays excluded for an early high value, and stays with incomplete follow-up. Regenerate with `python -m biocog generate-synthetic` then `python scripts/update_manifest.py`.
