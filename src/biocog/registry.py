"""Versioned offline model updates with evaluation before promotion.

``offline_update`` warm-starts a candidate from the promoted (incumbent) ensemble on the original
training windows PLUS a new batch of patients, then compares candidate and incumbent on the
VALIDATION patients only. The candidate is promoted only if it beats the incumbent by the required
margin on validation average precision without worsening calibrated Brier beyond a tolerance.
Test patients are never read here. Every decision (promoted or rejected) is appended to
``registry.json`` with its reason; the incumbent stays live on rejection.

Offline means: a human or scheduled job runs it on a batch of data; nothing updates itself in
deployment, and nothing here is cleared for clinical use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from .calibrate import PlattCalibrator
from .features import Features
from .nets import NetEnsemble, load_ensemble
from .pipeline import Bundle, _member_seeds


class ModelRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.json"
        self.state: dict[str, Any] = (json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists()
                                      else {"promoted": None, "history": []})

    def _flush(self) -> None:
        self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def next_version(self) -> str:
        return f"v{len(self.state['history']) + 1:04d}"

    def register(self, ens: NetEnsemble, calib: PlattCalibrator, val_metrics: dict, status: str,
                 reason: str, parent: str | None) -> str:
        v = self.next_version()
        d = self.root / v
        ens.save(d)
        calib.save(d / "calibrator.json")
        self.state["history"].append({"version": v, "status": status, "reason": reason, "parent": parent,
                                      "val_metrics": val_metrics})
        if status == "promoted":
            self.state["promoted"] = v
        self._flush()
        return v

    def load(self, version: str, b: Bundle) -> tuple[NetEnsemble, PlattCalibrator]:
        d = self.root / version
        return load_ensemble(d, b.cfg, b.pre), PlattCalibrator.load(d / "calibrator.json")


def _val_metrics(ens: NetEnsemble, b: Bundle) -> tuple[dict[str, float], PlattCalibrator]:
    va = b.feats.split("val").labelled()
    y = va.index.label.to_numpy()
    raw = ens.predict_raw(va)
    calib = PlattCalibrator().fit(raw, y)
    p = calib.predict(raw)
    return {"val_average_precision": float(average_precision_score(y, p)),
            "val_brier": float(brier_score_loss(y, p)), "n_val_windows": len(y)}, calib


def offline_update(reg: ModelRegistry, b: Bundle, incumbent: NetEnsemble, new_batch: Features,
                   min_ap_gain: float = 0.0, brier_tolerance: float = 0.005, epochs: int = 6,
                   lr_scale: float = 0.3) -> dict[str, Any]:
    """Fine-tune on old train + ``new_batch`` (all labelled windows), then gate on validation."""
    old_patients = set(b.windows.patient_id)
    overlap = old_patients & set(new_batch.index.patient_id)
    if overlap:
        raise ValueError(f"new batch shares {len(overlap)} patients with the existing cohort")
    tr = b.feats.split("train").labelled()
    both = _concat(tr, new_batch.labelled())
    va = b.feats.split("val").labelled()
    cand = NetEnsemble(incumbent.kind, b.cfg, incumbent.rules_on, incumbent.drop, incumbent.name)
    cand.fit(both, both.index.label.to_numpy(), va, va.index.label.to_numpy(), incumbent.pre,
             _member_seeds(b.cfg), init_states=[dict(s) for s in incumbent.states], epochs=epochs,
             lr=float(b.cfg["models"]["lr"]) * lr_scale)
    inc_m, inc_cal = _val_metrics(incumbent, b)
    cand_m, cand_cal = _val_metrics(cand, b)
    if reg.state["promoted"] is None:
        reg.register(incumbent, inc_cal, inc_m, "promoted", "initial incumbent", None)
    parent = reg.state["promoted"]
    ok_ap = cand_m["val_average_precision"] >= inc_m["val_average_precision"] + min_ap_gain
    ok_brier = cand_m["val_brier"] <= inc_m["val_brier"] + brier_tolerance
    promote = ok_ap and ok_brier
    reason = (f"AP {inc_m['val_average_precision']:.4f}->{cand_m['val_average_precision']:.4f} "
              f"(need gain >= {min_ap_gain}); Brier {inc_m['val_brier']:.4f}->{cand_m['val_brier']:.4f} "
              f"(tolerance +{brier_tolerance})")
    v = reg.register(cand, cand_cal, cand_m, "promoted" if promote else "rejected", reason, parent)
    return {"candidate_version": v, "promoted": promote, "reason": reason, "incumbent": inc_m,
            "candidate": cand_m, "serving_version": reg.state["promoted"]}


def _concat(a: Features, b: Features) -> Features:
    from dataclasses import replace
    cat = np.concatenate
    return replace(
        a, index=pd.concat([a.index, b.index], ignore_index=True), static=cat([a.static, b.static]),
        seq=cat([a.seq, b.seq]), tab=cat([a.tab, b.tab]), age_h=cat([a.age_h, b.age_h]),
        n_obs=cat([a.n_obs, b.n_obs]), n_trend=cat([a.n_trend, b.n_trend]), slope=cat([a.slope, b.slope]),
        last_val=cat([a.last_val, b.last_val]), seq_r=cat([a.seq_r, b.seq_r]),
        rule_vec=cat([a.rule_vec, b.rule_vec]), flags=pd.concat([a.flags, b.flags], ignore_index=True))
