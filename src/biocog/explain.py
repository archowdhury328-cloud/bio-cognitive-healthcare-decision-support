"""Risk estimate + explanation + uncertainty + review decision for scored landmarks.

Three different quantities are reported separately, because they answer different questions:
  * calibrated risk  : P(recorded glucose excursion in the next H hours), Platt-calibrated on
                       validation patients. It is a probability under the training population.
  * model disagreement: std of the probabilities of the seeded ensemble members. A large value means
                       the members disagree about this input; it says nothing about the true risk.
  * missing-input / rule flags: which modalities are absent, which rules fired. These describe the
                       INPUT, not the model.

Explanation method: feature-group occlusion. For each group the group's inputs are replaced by the
neutral value (training mean in normalised space; modality marked absent) and the change in
calibrated risk is reported: contribution = risk(full) - risk(group removed). It is a local,
model-dependent attribution, not a causal effect.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import Features
from .nets import NetEnsemble, predict_logits
from .rules import active_rules
from .schema import BIO_IDX, GLUCOSE_IDX, VITAL_IDX

GROUPS = ("static", "vitals", "glucose", "other_biomarkers", "rule_features")


def _occlude(d: dict[str, np.ndarray], group: str) -> dict[str, np.ndarray]:
    d2 = {k: v.copy() for k, v in d.items()}
    other_bio = [i for i in BIO_IDX if i != GLUCOSE_IDX]
    if group == "static":
        d2["static"][:] = 0.0
    elif group == "rule_features":
        d2["rule"][:] = 0.0
    elif group == "vitals":
        d2["seq"][:, :, list(VITAL_IDX), :] = 0.0
    elif group == "glucose":
        d2["seq"][:, :, GLUCOSE_IDX, :] = 0.0
    elif group == "other_biomarkers":
        d2["seq"][:, :, other_bio, :] = 0.0
    else:
        raise ValueError(group)
    d2["avail"] = np.stack([(d2["seq"][:, :, list(VITAL_IDX), 1].sum((1, 2)) > 0),
                            (d2["seq"][:, :, list(BIO_IDX), 1].sum((1, 2)) > 0)], axis=1).astype(np.float32)
    return d2


def _risk(ens: NetEnsemble, calib, d: dict[str, np.ndarray]) -> np.ndarray:
    raw = np.mean([predict_logits(ens.kind, s, d, ens.cfg) for s in ens.states], axis=0)
    return calib.predict(raw)


def group_contributions(ens: NetEnsemble, calib, f: Features) -> pd.DataFrame:
    """Per-row, per-group change in calibrated risk when the group is occluded (rows aligned with ``f``)."""
    d = ens.tensors(f)
    full = _risk(ens, calib, d)
    cols = {g: full - _risk(ens, calib, _occlude(d, g)) for g in GROUPS}
    return pd.DataFrame(cols)


def explain_landmark(ens: NetEnsemble, calib, f: Features, row: int, horizon_h: float,
                     review_disagreement: float) -> dict[str, Any]:
    """Structured output for one landmark: estimate, horizon, modalities, rules, contributions, uncertainty."""
    one = f.take(np.array([row]))
    d = ens.tensors(one)
    p = float(_risk(ens, calib, d)[0])
    dis = float(ens.disagreement(one)[0])
    flags = one.flags.iloc[0]
    contrib = group_contributions(ens, calib, one).iloc[0].to_dict()
    vit_obs = bool(d["avail"][0, 0] > 0)
    bio_obs = bool(d["avail"][0, 1] > 0)
    return {
        "stay_id": str(one.index.stay_id.iloc[0]), "landmark_h": float(one.index.t_h.iloc[0]),
        "event_definition": "first recorded glucose > 180 mg/dL (laboratory excursion, not a diabetes diagnosis)",
        "horizon_h": horizon_h,
        "calibrated_risk": round(p, 4),
        "model_disagreement": round(dis, 4),
        "modalities_observed": {"static_ehr": True, "vitals": vit_obs, "biomarkers": bio_obs},
        "modalities_missing": [m for m, ok in (("vitals", vit_obs), ("biomarkers", bio_obs)) if not ok],
        "glucose_result_age_h": None if pd.isna(flags.glucose_result_age_h) else round(float(flags.glucose_result_age_h), 2),
        "rules_activated": active_rules(flags),
        "needs_review": bool(flags.needs_review),
        "low_agreement": bool(dis > review_disagreement),
        "feature_group_contributions": {k: round(float(v), 4) for k, v in contrib.items()},
        "contribution_method": "group occlusion: risk(full) - risk(group removed); local, not causal",
    }


def decide(risk: float, thr: float, cooldown_ok: bool, needs_review: bool, low_agreement: bool) -> str:
    """Simulated alert decision. Nothing here is clinical advice or automated care."""
    if risk < thr or not cooldown_ok:
        return "NONE"
    return "QUEUE_REVIEW" if (needs_review or low_agreement) else "ALERT"


def trajectory(ens: NetEnsemble, calib, f: Features, stay_id: str, thr: float, cooldown_h: float,
               review_disagreement: float) -> pd.DataFrame:
    """Risk trajectory for one stay with the simulated decision at every landmark."""
    sel = np.where(f.index.stay_id.to_numpy() == stay_id)[0]
    sub = f.take(sel[np.argsort(f.index.t_h.to_numpy()[sel])])
    p = calib.predict(ens.predict_raw(sub))
    dis = ens.disagreement(sub)
    rows, last = [], -np.inf
    for i in range(len(sub)):
        t = float(sub.index.t_h.iloc[i])
        dec = decide(p[i], thr, t - last >= cooldown_h, bool(sub.flags.needs_review.iloc[i]),
                     bool(dis[i] > review_disagreement))
        if dec in ("ALERT", "QUEUE_REVIEW"):
            last = t
        rows.append({"t_h": t, "risk": p[i], "disagreement": dis[i], "label": int(sub.index.label.iloc[i]),
                     "needs_review": bool(sub.flags.needs_review.iloc[i]), "decision": dec,
                     "event_time_h": float(sub.index.event_time_h.iloc[i])})
    return pd.DataFrame(rows)
