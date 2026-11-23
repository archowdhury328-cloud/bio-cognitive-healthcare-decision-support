"""One-command offline demo on the committed SYNTHETIC fixture.

timestamped observations -> cohort windows -> risk trajectory -> explanation -> simulated alert decision.
Nothing is downloaded and no credentials are needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import cohort, explain, pipeline, report
from .config import load_config
from .data import load_manifest, verify_dataset
from .datasets import FIXTURE_DIR, load_dataset
from .pipeline import clean_json

DEMO_MODELS = ["last_value", "logistic", "gbt", "proposed"]
FAST = {"models": {"n_members": 2, "epochs": 14, "patience": 4}}


def run_demo(out_dir: Path, fast: bool = True, overrides: dict | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    check = verify_dataset("fixture", FIXTURE_DIR, load_manifest())
    if not check["ok"]:
        raise RuntimeError(f"fixture checksum failure: {check}")
    cfg = load_config(overrides=overrides or (FAST if fast else None))
    stays, events, _ = load_dataset("fixture")
    b = pipeline.build_bundle(stays, events, cfg)
    pipeline.save_bundle(b, out_dir)
    pipeline.train_all(b, out_dir, DEMO_MODELS)
    M = pipeline.evaluate_run(out_dir, cfg, DEMO_MODELS)
    pol = M["policy"]
    ens, calib = pipeline.load_model("proposed", out_dir, b)
    thr, cooldown = pol["thresholds"]["proposed"], pol["cooldown_h"]
    P = pd.read_csv(out_dir / "predictions.csv")
    te = P[(P.split == "test") & P.event_time_h.notna()]
    n_land = te.groupby("stay_id").size()
    # Pre-declared, outcome-blind rule: first test event stay (by id) with >= 12 scored landmarks, so the
    # trajectory is long enough to plot. It is NOT chosen for having a detected alert.
    sid = sorted(n_land[n_land >= 12].index)[0]
    dis_cut = float(cfg["rules"]["review_disagreement"])
    traj = explain.trajectory(ens, calib, b.feats, sid, thr, cooldown, dis_cut)
    fired = traj[traj.decision.isin(["ALERT", "QUEUE_REVIEW"])]
    row_t = float(fired.t_h.iloc[0]) if len(fired) else float(traj.t_h.iloc[-1])
    sel = np.where((b.feats.index.stay_id == sid).to_numpy() & (b.feats.index.t_h == row_t).to_numpy())[0][0]
    expl = explain.explain_landmark(ens, calib, b.feats, int(sel), float(cfg["task"]["horizon_h"]), dis_cut)
    expl["decision"] = str(traj[traj.t_h == row_t].decision.iloc[0])
    expl["event_time_h"] = float(traj.event_time_h.iloc[0])
    true_alerts = fired[fired.label == 1]
    expl["alert_fired"] = bool(len(fired))
    expl["shown_alert_in_horizon"] = bool(len(fired) and int(traj[traj.t_h == row_t].label.iloc[0]) == 1)
    expl["first_true_alert_h"] = float(true_alerts.t_h.iloc[0]) if len(true_alerts) else None
    expl["lead_time_h"] = float(traj.event_time_h.iloc[0] - true_alerts.t_h.iloc[0]) if len(true_alerts) else None
    expl["synthetic"] = True
    st, ev = cohort.to_hours(stays, events)
    g = ev[(ev.stay_id == sid) & (ev.variable == "glucose") & (ev.t_chart <= traj.t_h.max() + 6)]
    report.plot_trajectory(traj, g, thr, out_dir / "trajectory_SYNTHETIC.png",
                           "SYNTHETIC DATA - demonstration only, not a real patient")
    flow = b.flow
    summary = {
        "data_check": check, "cohort_flow": {f"{r.stage}[{r.split}]": int(r.n) for r in flow.itertuples()},
        "policy": pol, "example": expl,
        "proposed_test": {k: M["models"]["proposed"]["window"].get(k) for k in ("auroc", "average_precision", "brier")},
        "alerts_proposed_test": {k: v for k, v in M["models"]["proposed"]["alerts_test"].items()
                                 if k not in ("detected_stay_ids", "lead_times_h")},
    }
    (out_dir / "demo_output.json").write_text(json.dumps(clean_json(summary), indent=2), encoding="utf-8")
    (out_dir / "trajectory.csv").write_text(traj.to_csv(index=False), encoding="utf-8")
    return summary


def format_example(s: dict[str, Any]) -> str:
    e = s["example"]
    lines = [
        "SYNTHETIC DEMO (not real patients, not clinical advice)",
        f"stay {e['stay_id']}  landmark t={e['landmark_h']:.0f} h after ICU admission  horizon {e['horizon_h']:g} h",
        f"event: {e['event_definition']}",
        f"calibrated risk        : {e['calibrated_risk']:.3f}",
        f"model disagreement     : {e['model_disagreement']:.3f}  (ensemble std; not a risk)",
        f"modalities observed    : {', '.join(k for k, v in e['modalities_observed'].items() if v)}",
        f"modalities missing     : {', '.join(e['modalities_missing']) or 'none'}",
        f"latest glucose age     : {e['glucose_result_age_h']} h",
        f"rules activated        : {'; '.join(e['rules_activated']) or 'none'}",
        "feature-group contribution to risk (occlusion):",
    ]
    lines += [f"    {k:<18}{v:+.3f}" for k, v in e["feature_group_contributions"].items()]
    lines.append(f"simulated decision     : {e['decision']}")
    ev = e["event_time_h"]
    if not e["alert_fired"]:
        lines.append(f"no alert fired before the recorded excursion at t={ev:.1f} h (a MISSED event; the landmark "
                     "shown is the last one before it)")
    else:
        if not e["shown_alert_in_horizon"]:
            lines.append(f"this alert is {ev - e['landmark_h']:.1f} h before the excursion, outside the "
                         f"{e['horizon_h']:g} h horizon: the evaluation counts it as a FALSE alert")
        if e["lead_time_h"] is not None:
            lines.append(f"first TRUE alert (inside horizon) at t={e['first_true_alert_h']:.0f} h -> warning lead "
                         f"time {e['lead_time_h']:.1f} h before the recorded excursion at t={ev:.1f} h")
        else:
            lines.append(f"no alert fell inside the horizon: the event at t={ev:.1f} h counts as MISSED")
    return "\n".join(lines)

