"""Command-line interface: ``python -m biocog <command>`` (or ``biocog <command>``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import data, datasets, pipeline, provenance, report, routing
from .config import load_config

DEFAULT_RUNS = Path("runs_local")


def _run_dir(args) -> Path:
    d = Path(args.run) if args.run else DEFAULT_RUNS / args.dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def _names(args, include_ablations: bool) -> list[str]:
    return pipeline.PRIMARY + (pipeline.ABLATIONS if include_ablations else [])


def cmd_fetch_data(args) -> int:
    res = data.fetch_mimic_demo()
    print(json.dumps(res, indent=2))
    return 0 if res["ok"] else 1


def cmd_verify_data(args) -> int:
    root = datasets.FIXTURE_DIR if args.which == "fixture" else data.MIMIC_DIR
    name = "fixture" if args.which == "fixture" else "mimic_iv_demo_2.2"
    res = data.verify_dataset(name, root)
    print(json.dumps(res, indent=2))
    return 0 if res["ok"] else 1


def cmd_generate_synthetic(args) -> int:
    datasets.write_fixture(Path(args.out), args.n_patients, args.seed)
    print(f"wrote synthetic fixture ({args.n_patients} patients, seed {args.seed}) to {args.out}")
    return 0


def cmd_cohort(args) -> int:
    cfg = load_config(args.config)
    run = _run_dir(args)
    stays, events, info = datasets.load_dataset(args.dataset)
    b = pipeline.build_bundle(stays, events, cfg)
    pipeline.save_bundle(b, run)
    man = provenance.collect(cfg, info["dataset"], "2.2" if args.dataset == "mimic" else "generated", b.flow)
    man["dataset_info"] = info
    (run / "run_manifest.json").write_text(json.dumps(pipeline.clean_json(man), indent=2), encoding="utf-8")
    print(b.flow.to_string(index=False))
    return 0


def cmd_train(args) -> int:
    run = _run_dir(args)
    b = pipeline.load_bundle(run)
    names = args.models or _names(args, args.ablations)
    print(json.dumps(pipeline.train_all(b, run, names), indent=2))
    return 0


def cmd_evaluate(args) -> int:
    run, cfg = _run_dir(args), load_config(args.config)
    names = [c[2:] for c in pd.read_csv(run / "predictions.csv", nrows=0).columns if c.startswith("p_")]
    out = pipeline.evaluate_run(run, cfg, names)
    print(json.dumps({n: out["models"][n]["alerts_test"]["detected_events"] for n in names}, indent=2))
    return 0


def cmd_ablate(args) -> int:
    args.models = pipeline.PRIMARY + pipeline.ABLATIONS
    cmd_train(args)
    return cmd_evaluate(args)


def cmd_federated(args) -> int:
    from .federated import run_federated
    run = _run_dir(args)
    b = pipeline.load_bundle(run)
    res = run_federated(b, seed=args.seed)
    (run / "federated.json").write_text(json.dumps(pipeline.clean_json(res), indent=2), encoding="utf-8")
    report.write_federated_md(run / "federated.json", run, run.name)
    print(json.dumps({"federated": res["federated_final"], "pooled": res["pooled"]}, indent=2))
    return 0


def cmd_route(args) -> int:
    cfg = load_config(args.config)
    res = routing.run_experiment(int(cfg["routing"]["n_train_episodes"]), int(cfg["routing"]["n_eval_patients"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "routing.json").write_text(json.dumps(pipeline.clean_json(res), indent=2), encoding="utf-8")
    report.write_routing_md(out / "routing.json", out)
    print(json.dumps({"fixed": res["fixed_threshold"]["heldout"], "learned": res["learned_q_policy"]["aggregate"]},
                     indent=2))
    return 0


def cmd_update(args) -> int:
    from . import synth
    from .registry import ModelRegistry, offline_update
    run = _run_dir(args)
    b = pipeline.load_bundle(run)
    ens, _ = pipeline.load_model("proposed", run, b)
    stays, events, _ = synth.generate(args.new_patients, args.seed)
    stays["patient_id"] = "N" + stays.patient_id.astype(str)  # guarantee disjoint patient ids
    stays["stay_id"] = "N" + stays.stay_id.astype(str)
    events["stay_id"] = "N" + events.stay_id.astype(str)
    batch = pipeline.make_update_batch(stays, events, b.cfg)
    res = offline_update(ModelRegistry(run / "registry"), b, ens, batch)
    report.write_registry_md(run / "registry" / "registry.json", run / "registry")
    print(json.dumps(pipeline.clean_json(res), indent=2))
    return 0


def cmd_report(args) -> int:
    run, cfg = _run_dir(args), load_config(args.config)
    names = [c[2:] for c in pd.read_csv(run / "predictions.csv", nrows=0).columns if c.startswith("p_")]
    out = Path(args.out)
    report.write_tables(run, out, names, args.title, aggregate_only=True)
    tag = args.tag
    for src, writer in (("federated.json", report.write_federated_md), ("registry/registry.json", report.write_registry_md)):
        if (run / src).exists():
            writer(run / src, out, *([args.title] if src == "federated.json" else []))
    report.plot_pr_calibration(run, out, [n for n in names if n in pipeline.PRIMARY], tag,
                               int(cfg["metrics"]["calibration_bins"]))
    print(f"wrote tables and figures to {out}")
    return 0


def cmd_score(args) -> int:
    """Research interface: score SYNTHETIC canonical tables with a trained run. Not for patient data."""
    if not args.confirm_synthetic:
        print("refusing: pass --confirm-synthetic. This interface is for synthetic examples only; do not upload "
              "identifiable patient data, and do not use outputs for care.")
        return 2
    from . import cohort, explain
    from .features import build_features
    from .rules import apply_rules
    run = _run_dir(args)
    b = pipeline.load_bundle(run)
    stays = pd.read_csv(args.stays, parse_dates=["intime", "followup_end"])
    events = pd.read_csv(args.events, parse_dates=["charttime", "availtime"])
    from .schema import EVENT_COLUMNS, STAY_COLUMNS
    if list(stays.columns) != STAY_COLUMNS or list(events.columns) != EVENT_COLUMNS:
        print(f"refusing: tables must have exactly the canonical columns {STAY_COLUMNS} and {EVENT_COLUMNS}")
        return 2
    if len(stays) > 200:
        print("refusing: at most 200 stays per call")
        return 2
    st, ev = cohort.to_hours(stays, events)
    tab = cohort.stay_table(st, ev, b.cfg)
    w = cohort.make_windows(tab, ev, {p: "test" for p in tab[tab.eligible].patient_id}, b.cfg)
    if w.empty:
        print("no eligible windows in the supplied tables")
        return 1
    f = apply_rules(build_features(st, ev, w, b.cfg), b.cfg)
    ens, calib = pipeline.load_model("proposed", run, b)
    pol = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["policy"]
    out = []
    for sid in sorted(set(w.stay_id)):
        t = explain.trajectory(ens, calib, f, sid, pol["thresholds"]["proposed"], pol["cooldown_h"],
                               float(b.cfg["rules"]["review_disagreement"]))
        t.insert(0, "stay_id", sid)
        out.append(t)
    res = pd.concat(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(res.groupby("stay_id").agg(max_risk=("risk", "max"), alerts=("decision", lambda d: int((d != "NONE").sum()))))
    print(f"wrote trajectories to {args.out}")
    return 0


def cmd_demo(args) -> int:
    from .demo import format_example, run_demo
    s = run_demo(Path(args.out), fast=not args.full)
    print(format_example(s))
    print(f"\nfiles written to {args.out} (cohort flow, predictions, trajectory figure, demo_output.json)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="biocog", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        sp = sub.add_parser(name, **kw)
        sp.set_defaults(fn=fn)
        return sp

    def common(sp, dataset=True):
        if dataset:
            sp.add_argument("--dataset", choices=["fixture", "synthetic", "mimic"], default="fixture")
        sp.add_argument("--run", help="run directory (default runs_local/<dataset>)")
        sp.add_argument("--config", default=None)

    add("fetch-data", cmd_fetch_data, help="download + checksum-verify the open MIMIC-IV demo v2.2")
    sp = add("verify-data", cmd_verify_data, help="verify checksums against data/manifest.json")
    sp.add_argument("--which", choices=["fixture", "mimic"], default="fixture")
    sp = add("generate-synthetic", cmd_generate_synthetic, help="regenerate the synthetic fixture")
    sp.add_argument("--out", default=str(datasets.FIXTURE_DIR))
    sp.add_argument("--n-patients", type=int, default=datasets.FIXTURE_PATIENTS)
    sp.add_argument("--seed", type=int, default=datasets.FIXTURE_SEED)
    sp = add("cohort", cmd_cohort, help="build cohort, split, windows and features")
    common(sp)
    sp = add("train", cmd_train, help="train + calibrate models; write predictions")
    common(sp)
    sp.add_argument("--models", nargs="*")
    sp.add_argument("--ablations", action="store_true")
    sp = add("evaluate", cmd_evaluate, help="select policy on validation, evaluate on test")
    common(sp)
    sp = add("ablate", cmd_ablate, help="train all models incl. ablations, then evaluate")
    common(sp)
    sp = add("federated", cmd_federated, help="FedAvg simulation vs pooled training")
    common(sp)
    sp.add_argument("--seed", type=int, default=0)
    sp = add("route", cmd_route, help="offline alert-routing experiment (synthetic)")
    sp.add_argument("--config", default=None)
    sp.add_argument("--out", default="runs_local/routing")
    sp = add("update", cmd_update, help="offline model update with promotion gate")
    common(sp)
    sp.add_argument("--new-patients", type=int, default=60)
    sp.add_argument("--seed", type=int, default=99)
    sp = add("report", cmd_report, help="tables + figures from saved predictions")
    common(sp)
    sp.add_argument("--out", required=True)
    sp.add_argument("--title", default="Results")
    sp.add_argument("--tag", default="")
    sp = add("score", cmd_score, help="score SYNTHETIC canonical tables with a trained run")
    common(sp)
    sp.add_argument("--stays", required=True)
    sp.add_argument("--events", required=True)
    sp.add_argument("--out", default="score_output.csv")
    sp.add_argument("--confirm-synthetic", action="store_true")
    sp = add("demo", cmd_demo, help="one-command offline demo on the synthetic fixture")
    sp.add_argument("--out", default="demo_output")
    sp.add_argument("--full", action="store_true", help="use the full training budget")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
