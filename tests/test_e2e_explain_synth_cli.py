"""Explanation structure, synthetic generator properties, CLI, and the clean-checkout end-to-end run."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from biocog import cli, cohort, datasets, explain, pipeline, report, synth
from biocog.demo import format_example, run_demo

ROOT = Path(__file__).resolve().parents[1]


# ---- synthetic generator ---------------------------------------------------------------------
def test_generator_is_deterministic_and_has_requested_properties():
    a = synth.generate(60, 3)
    b = synth.generate(60, 3)
    for x, y in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(x, y)
    stays, ev, truth = a
    lag = (ev.availtime - ev.charttime) / pd.Timedelta(hours=1)
    assert (lag[ev.variable == "glucose"] > 2.5).any()  # delayed lab results
    assert (lag[ev.variable.isin(["hr", "sbp"])] == 0).all()
    has_vit = ev[ev.variable == "hr"].stay_id.unique()
    assert len(set(stays.stay_id) - set(has_vit)) > 0  # absent vital-sign modality
    assert truth.incomplete_followup.any() and truth.event_time_h.isna().any() and truth.event_time_h.notna().any()
    assert truth.historical_high.any()
    assert ev.groupby(["stay_id", "variable"]).size().max() > 3  # repeated observations


def test_committed_fixture_is_labelled_and_usable():
    stays, ev = datasets.load_fixture()
    assert len(stays) > 100 and ev.variable.nunique() == 11
    assert (ev.groupby("stay_id").charttime.min() >= stays.set_index("stay_id").intime.reindex(
        ev.stay_id.unique()).values.min()).all()


# ---- explanation ------------------------------------------------------------------------------
def test_explain_landmark_reports_everything_requested(bundle, tiny_proposed, tiny_cfg):
    calib = pipeline.fit_calibrator(tiny_proposed, bundle)
    f = bundle.feats
    row = int(np.where(f.flags.R2_stale_vars.to_numpy() != "")[0][0])
    out = explain.explain_landmark(tiny_proposed, calib, f, row, 6.0, 0.15)
    assert 0 <= out["calibrated_risk"] <= 1 and out["horizon_h"] == 6.0 and "not a diabetes diagnosis" in out["event_definition"]
    assert set(out["feature_group_contributions"]) == set(explain.GROUPS)
    assert set(out["modalities_observed"]) == {"static_ehr", "vitals", "biomarkers"}
    assert out["rules_activated"] and isinstance(out["model_disagreement"], float)
    assert "not causal" in out["contribution_method"]


def test_absent_modality_has_zero_occlusion_contribution_and_is_reported_missing(bundle, tiny_proposed):
    import dataclasses
    calib = pipeline.fit_calibrator(tiny_proposed, bundle)
    f = bundle.feats.take(np.array([40]))
    seq, seq_r = f.seq.copy(), f.seq_r.copy()
    for a in (seq, seq_r):
        a[:, :, :5, 0] = 0
        a[:, :, :5, 1] = 0
    g = dataclasses.replace(f, seq=seq, seq_r=seq_r)
    contrib = explain.group_contributions(tiny_proposed, calib, g).iloc[0]
    assert contrib["vitals"] == 0.0
    out = explain.explain_landmark(tiny_proposed, calib, g, 0, 6.0, 0.15)
    assert "vitals" in out["modalities_missing"] and not out["modalities_observed"]["vitals"]


def test_trajectory_alerts_respect_cooldown_and_review_routing(bundle, tiny_proposed):
    calib = pipeline.fit_calibrator(tiny_proposed, bundle)
    sid = bundle.windows.stay_id.iloc[0]
    traj = explain.trajectory(tiny_proposed, calib, bundle.feats, sid, thr=0.0, cooldown_h=3.0, review_disagreement=1.0)
    fired = traj[traj.decision != "NONE"].t_h.to_numpy()
    assert len(fired) > 1 and (np.diff(fired) >= 3).all()
    assert (traj.t_h.diff().dropna() > 0).all()


# ---- CLI ---------------------------------------------------------------------------------------
def test_cli_verify_and_generate(tmp_path, capsys):
    assert cli.main(["verify-data", "--which", "fixture"]) == 0
    assert '"ok": true' in capsys.readouterr().out
    assert cli.main(["generate-synthetic", "--out", str(tmp_path), "--n-patients", "6", "--seed", "1"]) == 0
    assert {p.name for p in tmp_path.iterdir()} == {"stays.csv", "events.csv.gz", "truth.csv"}
    again = tmp_path / "b"
    cli.main(["generate-synthetic", "--out", str(again), "--n-patients", "6", "--seed", "1"])
    assert (tmp_path / "events.csv.gz").read_bytes() == (again / "events.csv.gz").read_bytes()  # byte-reproducible


def test_cli_cohort_train_evaluate_report_roundtrip(tmp_path, capsys, monkeypatch):
    cfgp = tmp_path / "cfg.yaml"
    import yaml

    from biocog.config import load_config
    from tests_support import TINY_OVERRIDES  # noqa: F401  (defined below via conftest path)
    cfgp.write_text(yaml.safe_dump(load_config(overrides=TINY_OVERRIDES)), encoding="utf-8")
    run = tmp_path / "run"
    base = ["--dataset", "fixture", "--run", str(run), "--config", str(cfgp)]
    assert cli.main(["cohort", *base]) == 0
    assert cli.main(["train", *base, "--models", "last_value", "logistic", "gbt", "proposed"]) == 0
    assert cli.main(["evaluate", *base]) == 0
    assert cli.main(["report", *base, "--out", str(tmp_path / "res"), "--title", "T", "--tag", "SYNTHETIC"]) == 0
    fx = datasets.FIXTURE_DIR
    sc = ["score", *base, "--stays", str(fx / "stays.csv"), "--events", str(fx / "events.csv.gz"),
          "--out", str(tmp_path / "scored.csv")]
    assert cli.main(sc) == 2  # refuses without the synthetic confirmation
    assert cli.main([*sc, "--confirm-synthetic"]) == 0
    assert {"stay_id", "risk", "decision"} <= set(pd.read_csv(tmp_path / "scored.csv").columns)
    bad = tmp_path / "bad.csv"
    pd.read_csv(fx / "stays.csv").assign(name="x").to_csv(bad, index=False)
    assert cli.main(["score", *base, "--stays", str(bad), "--events", str(fx / "events.csv.gz"),
                     "--confirm-synthetic"]) == 2  # extra columns refused
    assert (tmp_path / "res" / "results.md").exists() and (tmp_path / "res" / "pr_calibration.png").exists()
    man = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert {"git_commit", "config_hash", "split_seed", "packages", "python", "platform", "cohort_exclusions"} <= set(man)
    md = (tmp_path / "res" / "results.md").read_text(encoding="utf-8")
    assert "Cohort flow" in md and "Training budget" in md
    published = json.loads((tmp_path / "res" / "metrics.json").read_text(encoding="utf-8"))
    assert all("detected_stay_ids" not in m["alerts_test"] for m in published["models"].values())  # aggregate only


def test_package_entrypoint_runs():
    out = subprocess.run([sys.executable, "-m", "biocog", "verify-data", "--which", "fixture"], cwd=ROOT,
                         capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""} | _sysenv())
    assert out.returncode == 0 and '"ok": true' in out.stdout


def _sysenv():
    import os
    return {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}


# ---- clean-checkout end to end ----------------------------------------------------------------
def test_end_to_end_raw_fixture_to_trajectory_explanation_alert_and_report(tmp_path, tiny_cfg):
    summary = run_demo(tmp_path, overrides=tiny_cfg)
    ex = summary["example"]
    assert ex["synthetic"] and ex["decision"] in {"NONE", "ALERT", "QUEUE_REVIEW"}
    assert ex["stay_id"] and 0 <= ex["calibrated_risk"] <= 1
    traj = pd.read_csv(tmp_path / "trajectory.csv")
    assert {"risk", "disagreement", "decision", "t_h"} <= set(traj.columns) and len(traj) >= 12
    assert traj.t_h.max() < traj.event_time_h.iloc[0]  # scoring stopped before the event
    assert (tmp_path / "trajectory_SYNTHETIC.png").stat().st_size > 5000
    text = format_example(summary)
    assert "SYNTHETIC DEMO" in text and "calibrated risk" in text and "simulated decision" in text
    # same artifacts feed the report generator
    names = pipeline_names(tmp_path)
    report.write_tables(tmp_path, tmp_path / "rep", names, "demo")
    assert "Alert-level evaluation" in (tmp_path / "rep" / "results.md").read_text(encoding="utf-8")
    # cohort numbers in the demo equal an independent rebuild from raw rows
    b = pipeline.load_bundle(tmp_path)
    st, ev = cohort.to_hours(*datasets.load_fixture())
    assert len(cohort.stay_table(st, ev, tiny_cfg)) == int(b.flow[b.flow.stage == "all_stays"].n.iloc[0])


def pipeline_names(run):
    return [c[2:] for c in pd.read_csv(run / "predictions.csv", nrows=0).columns if c.startswith("p_")]
