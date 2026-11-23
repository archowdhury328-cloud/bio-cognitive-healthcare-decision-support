"""Experiment orchestration: cohort bundle -> train -> predictions -> evaluation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import alerts, cohort, metrics
from .baselines import (
    GradientBoostedBaseline,
    LastValueBaseline,
    LogisticBaseline,
    load_baseline,
)
from .calibrate import PlattCalibrator
from .features import Features, Preprocessor, build_features
from .nets import NetEnsemble, load_ensemble
from .rules import apply_rules

PRIMARY = ["last_value", "logistic", "gbt", "gru", "proposed"]
ABLATIONS = ["proposed_no_vitals", "proposed_no_biomarkers", "proposed_no_reasoning", "proposed_concat"]
BASE_SEED = 100


@dataclass
class Bundle:
    cfg: dict[str, Any]
    stay_tab: pd.DataFrame
    windows: pd.DataFrame
    feats: Features
    pre: Preprocessor
    flow: pd.DataFrame


def build_bundle(stays: pd.DataFrame, events: pd.DataFrame, cfg: dict[str, Any]) -> Bundle:
    st, ev = cohort.to_hours(stays, events)
    tab = cohort.stay_table(st, ev, cfg)
    split_map = cohort.assign_splits(tab, int(cfg["split"]["seed"]), cfg["split"]["fractions"])
    windows = cohort.make_windows(tab, ev, split_map, cfg)
    feats = apply_rules(build_features(st, ev, windows, cfg), cfg)
    pre = Preprocessor(cfg).fit(feats.split("train"))
    return Bundle(cfg, tab, windows, feats, pre, cohort.cohort_flow(tab, windows))


def save_bundle(b: Bundle, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(b, run_dir / "cohort.joblib")
    b.flow.to_csv(run_dir / "cohort_flow.csv", index=False)


def load_bundle(run_dir: Path) -> Bundle:
    """Bundles are joblib pickles written by this package; do not load files from untrusted sources."""
    return joblib.load(run_dir / "cohort.joblib")


def _member_seeds(cfg: dict[str, Any]) -> list[int]:
    return [BASE_SEED + i for i in range(int(cfg["models"]["n_members"]))]


def fit_model(name: str, b: Bundle, extra_seed: int = 0):
    """Fit one named model on training patients, using validation patients only for early stopping."""
    cfg = b.cfg
    tr, va = b.feats.split("train").labelled(), b.feats.split("val").labelled()
    y_tr, y_va = tr.index.label.to_numpy(), va.index.label.to_numpy()
    if name == "last_value":
        return LastValueBaseline().fit(tr, y_tr)
    if name == "logistic":
        return LogisticBaseline(cfg, BASE_SEED + extra_seed).fit(tr, y_tr, val=(va, y_va))
    if name == "gbt":
        return GradientBoostedBaseline(cfg, BASE_SEED + extra_seed).fit(tr, y_tr)
    spec = {"gru": ("gru", True, ()), "proposed": ("proposed", True, ()),
            "proposed_concat": ("proposed_concat", True, ()),
            "proposed_no_reasoning": ("proposed", False, ()),
            "proposed_no_vitals": ("proposed", True, ("vitals",)),
            "proposed_no_biomarkers": ("proposed", True, ("biomarkers",))}[name]
    ens = NetEnsemble(spec[0], cfg, rules_on=spec[1], drop=spec[2], name=name)
    seeds = [s + extra_seed for s in _member_seeds(cfg)]
    return ens.fit(tr, y_tr, va, y_va, b.pre, seeds)


def fit_calibrator(model, b: Bundle) -> PlattCalibrator:
    """Calibration uses validation patients ONLY."""
    va = b.feats.split("val").labelled()
    return PlattCalibrator().fit(model.predict_raw(va), va.index.label.to_numpy())


def save_model(name: str, model, calib: PlattCalibrator, run_dir: Path) -> None:
    d = run_dir / "models" / name
    d.mkdir(parents=True, exist_ok=True)
    if isinstance(model, NetEnsemble):
        model.save(d)
    else:
        model.save(d / "model.joblib")
    calib.save(d / "calibrator.json")


def load_model(name: str, run_dir: Path, b: Bundle):
    d = run_dir / "models" / name
    calib = PlattCalibrator.load(d / "calibrator.json")
    if (d / "meta.json").exists():
        return load_ensemble(d, b.cfg, b.pre), calib
    return load_baseline(d / "model.joblib"), calib


def time_inference(model, f: Features, n: int = 50) -> float:
    """Median single-landmark ``predict_raw`` latency in ms (excludes feature extraction)."""
    ts = []
    for i in range(min(n, len(f))):
        row = f.take(np.array([i]))
        t0 = time.perf_counter()
        model.predict_raw(row)
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts)) if ts else float("nan")


def train_all(b: Bundle, run_dir: Path, names: list[str]) -> dict[str, Any]:
    """Train ``names``, calibrate on validation, save models, write predictions and a budget log."""
    f_all, budget = b.feats, {}
    preds = b.windows[["stay_id", "patient_id", "split", "t_h", "label", "unknown_reason",
                       "event_time_h", "followup_h"]].copy()
    preds = pd.concat([preds, f_all.flags.reset_index(drop=True)], axis=1)
    test = f_all.split("test")
    for name in names:
        t0 = time.time()
        model = fit_model(name, b)
        calib = fit_calibrator(model, b)
        save_model(name, model, calib, run_dir)
        raw = model.predict_raw(f_all)
        preds[f"raw_{name}"], preds[f"p_{name}"] = raw, calib.predict(raw)
        if isinstance(model, NetEnsemble):
            preds[f"dis_{name}"] = model.disagreement(f_all)
        budget[name] = {"fit_seconds": round(time.time() - t0, 1), "latency_ms_median": time_inference(model, test),
                        "n_train_windows": int((b.windows.split.eq("train") & b.windows.label.ge(0)).sum())}
        if isinstance(model, NetEnsemble):
            budget[name].update({"n_members": len(model.states),
                                 "epochs_run": [lg["epochs_run"] for lg in model.logs],
                                 "max_epochs": int(b.cfg["models"]["epochs"]),
                                 "hidden": int(b.cfg["models"]["hidden"])})
    preds.to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "training_budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")
    return budget


def evaluate_run(run_dir: Path, cfg: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """Select policy on validation, evaluate on test. Reads only saved predictions."""
    P = pd.read_csv(run_dir / "predictions.csv")
    mc, ac = cfg["metrics"], cfg["alerts"]
    cad = float(cfg["task"]["cadence_h"])
    val, test = P[P.split == "val"].reset_index(drop=True), P[P.split == "test"].reset_index(drop=True)
    cooldown, thr = alerts.select_policy({n: (val, val[f"p_{n}"].to_numpy()) for n in names},
                                         float(ac["false_alerts_per_patient_day"]), ac["cooldown_grid_h"], cad)
    out: dict[str, Any] = {"policy": {"cooldown_h": cooldown, "thresholds": thr,
                                      "false_alert_budget_per_patient_day": ac["false_alerts_per_patient_day"]},
                           "models": {}}
    all_events = set(test[test.event_time_h.notna()].stay_id)
    lab = test[test.label >= 0]
    for n in names:
        p_t = lab[f"p_{n}"].to_numpy()
        wm = metrics.window_metrics(lab.label.to_numpy(), p_t, lab.patient_id.to_numpy(), thr[n],
                                    int(mc["min_event_patients_point"]))
        ci = metrics.patient_bootstrap(lab.label.to_numpy(), p_t, lab.patient_id.to_numpy(), thr[n],
                                       int(mc["n_bootstrap"]), 0, int(mc["min_event_patients_ci"]),
                                       int(mc["min_event_patients_point"]))
        ae = alerts.evaluate_alerts(test, test[f"p_{n}"].to_numpy(), thr[n], cooldown, cad)
        ae_val = alerts.evaluate_alerts(val, val[f"p_{n}"].to_numpy(), thr[n], cooldown, cad)
        cal = metrics.calibration_table(lab.label.to_numpy(), p_t, int(mc["calibration_bins"]))
        out["models"][n] = {"window": wm, "ci95": ci, "alerts_test": ae, "alerts_val": ae_val,
                            "calibration": cal.to_dict("records")}
    ref = "proposed" if "proposed" in names else names[-1]
    d_ref = set(out["models"][ref]["alerts_test"]["detected_stay_ids"])
    out["paired_detection_vs_" + ref] = {
        n: alerts.paired_detection(d_ref, set(out["models"][n]["alerts_test"]["detected_stay_ids"]), all_events)
        for n in names if n != ref}
    out = clean_json(out)
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def clean_json(o):
    """Recursively convert numpy scalars to Python and NaN/inf to None (strict JSON)."""
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def make_update_batch(stays: pd.DataFrame, events: pd.DataFrame, cfg: dict[str, Any]) -> Features:
    """Features for a NEW batch of patients, all assigned to 'train' (used by offline updates)."""
    st, ev = cohort.to_hours(stays, events)
    tab = cohort.stay_table(st, ev, cfg)
    split_map = {p: "train" for p in tab[tab.eligible].patient_id.unique()}
    windows = cohort.make_windows(tab, ev, split_map, cfg)
    return apply_rules(build_features(st, ev, windows, cfg), cfg)
