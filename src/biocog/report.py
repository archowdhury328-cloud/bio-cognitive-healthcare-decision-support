"""Tables and figures built ONLY from saved predictions, metrics.json and run logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .pipeline import ABLATIONS, PRIMARY

LABELS = {"last_value": "Last glucose value", "logistic": "Logistic regression", "gbt": "Gradient-boosted trees",
          "gru": "GRU comparator", "proposed": "Proposed gated multimodal",
          "proposed_no_vitals": "Proposed - no vitals", "proposed_no_biomarkers": "Proposed - no biomarkers",
          "proposed_no_reasoning": "Proposed - no reasoning layer", "proposed_concat": "Proposed - concat fusion"}


def _f(x: Any, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _ci(m: dict, key: str, ci: dict) -> str:
    v = _f(m.get(key))
    if key in ci:
        v += f" [{ci[key][0]:.3f}, {ci[key][1]:.3f}]"
    return v


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def performance_table(M: dict, names: list[str]) -> pd.DataFrame:
    rows = []
    for n in names:
        m, ci = M["models"][n]["window"], M["models"][n]["ci95"]
        rows.append({"model": LABELS[n], "AUROC [95% CI]": _ci(m, "auroc", ci),
                     "AP [95% CI]": _ci(m, "average_precision", ci), "Brier [95% CI]": _ci(m, "brier", ci),
                     "accuracy": _f(m.get("accuracy")), "sensitivity": _f(m.get("sensitivity")),
                     "specificity": _f(m.get("specificity")), "precision": _f(m.get("precision")),
                     "threshold": _f(m.get("threshold")), "estimable": m["estimable"]})
    return pd.DataFrame(rows)


def alert_table(M: dict, names: list[str]) -> pd.DataFrame:
    rows = []
    for n in names:
        a = M["models"][n]["alerts_test"]
        lt = (f"{_f(a['lead_time_median_h'], 1)} [{_f(a['lead_time_q25_h'], 1)}, {_f(a['lead_time_q75_h'], 1)}]")
        rows.append({"model": LABELS[n], "detected / eligible events": f"{a['detected_events']} / {a['eligible_events']}",
                     "missed": a["missed_events"], "median lead h [IQR]": lt,
                     "false alerts / patient-day": _f(a["false_alerts_per_patient_day"], 2),
                     "unconfirmed alerts / patient-day": _f(a["unconfirmed_alerts_per_patient_day"], 2),
                     "threshold": _f(a["threshold"]), "cooldown h": a["cooldown_h"],
                     "monitored patient-days": _f(a["monitored_patient_days"], 1)})
    return pd.DataFrame(rows)


def paired_table(M: dict) -> pd.DataFrame:
    key = next(k for k in M if k.startswith("paired_detection_vs_"))
    ref = key.replace("paired_detection_vs_", "")
    rows = [{"comparison": f"{LABELS[ref]} vs {LABELS[n]}", "both detected": v["both"],
             f"only {ref}": v["only_a"], "only comparator": v["only_b"], "neither": v["neither"]}
            for n, v in M[key].items()]
    return pd.DataFrame(rows)


def budget_table(budget: dict, names: list[str]) -> pd.DataFrame:
    rows = []
    for n in names:
        if n not in budget:
            continue
        b = budget[n]
        rows.append({"model": LABELS[n], "fit seconds": b["fit_seconds"],
                     "train windows": b["n_train_windows"],
                     "members x epochs run": "-" if "n_members" not in b else
                     f"{b['n_members']} x {b['epochs_run']} (max {b['max_epochs']})",
                     "median inference ms / landmark": _f(b["latency_ms_median"], 2)})
    return pd.DataFrame(rows)


def rule_impact(run_dir: Path, M: dict, review_disagreement: float = 0.15) -> pd.DataFrame:
    """Effect of review-only rules (R3, R4) and ensemble disagreement on coverage and alert burden (test patients)."""
    from .alerts import apply_policy
    P = pd.read_csv(run_dir / "predictions.csv")
    te = P[P.split == "test"].reset_index(drop=True)
    pol = M["policy"]
    alert = apply_policy(te, te["p_proposed"].to_numpy(), pol["thresholds"]["proposed"], pol["cooldown_h"])
    flag = te.needs_review.astype(bool).to_numpy()
    low = (te["dis_proposed"] > review_disagreement).to_numpy()
    days = len(te) / 24.0
    rows = []
    for name, m in (("R3/R4 review flag", flag), ("ensemble disagreement > cutoff", low), ("either", flag | low)):
        routed = alert & m
        rows.append({"routing signal": name, "landmarks flagged": f"{m.mean():.1%}", "alerts total": int(alert.sum()),
                     "alerts routed to review": int(routed.sum()),
                     "of which true / false / unconfirmed": f"{int((routed & (te.label == 1)).sum())} / "
                     f"{int((routed & (te.label == 0)).sum())} / {int((routed & (te.label == -1)).sum())}",
                     "automatic alerts / patient-day": _f((alert & ~m).sum() / days, 2),
                     "all alerts / patient-day": _f(alert.sum() / days, 2)})
    return pd.DataFrame(rows)


def write_tables(run_dir: Path, out_dir: Path, names: list[str], title: str, aggregate_only: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    M = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    flow = pd.read_csv(run_dir / "cohort_flow.csv")
    budget = json.loads((run_dir / "training_budget.json").read_text(encoding="utf-8"))
    prim = [n for n in names if n in PRIMARY]
    abl = [n for n in names if n in ABLATIONS]
    parts = [f"# {title}\n", "Every number below was computed from saved predictions or run logs by "
             "`biocog report`.\n", "## Cohort flow\n", _md(flow), "\n## Test-set window metrics (labelled windows)\n",
             "Threshold = validation-selected alert threshold on calibrated risk. `n/a` = unestimable "
             "(too few positive or negative patients) - not zero.\n", _md(performance_table(M, prim)),
             "\n## Alert-level evaluation (test patients)\n", f"Shared policy: cooldown {M['policy']['cooldown_h']} h; "
             f"validation false-alert budget {M['policy']['false_alert_budget_per_patient_day']} per patient-day.\n",
             _md(alert_table(M, prim)), "\n## Paired event detection (same events, which model caught them)\n",
             _md(paired_table(M))]
    if abl:
        parts += ["\n## Ablations of the proposed model (same cohort, same budget)\n",
                  _md(performance_table(M, ["proposed"] + abl)), "\n", _md(alert_table(M, ["proposed"] + abl))]
    if "proposed" in names:
        parts += ["\n## Review-routing impact (R3/R4 rules change review decisions only)\n",
                  "Alerts of the proposed model under the shared policy; routed alerts become QUEUE_REVIEW instead of "
                  "ALERT. Risk itself is unchanged by these signals.\n", _md(rule_impact(run_dir, M))]
    parts += ["\n## Training budget and inference latency\n",
              "Latency = median single-landmark `predict_raw` call on this machine; excludes feature extraction.\n",
              _md(budget_table(budget, names))]
    (out_dir / "results.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    flow.to_csv(out_dir / "cohort_flow.csv", index=False)
    performance_table(M, names).to_csv(out_dir / "performance.csv", index=False)
    alert_table(M, names).to_csv(out_dir / "alerts.csv", index=False)
    slim = json.loads(json.dumps(M))
    if aggregate_only:  # remove per-stay identifiers from the published copy
        for m in slim["models"].values():
            for k in ("alerts_test", "alerts_val"):
                m[k].pop("detected_stay_ids", None)
                m[k].pop("lead_times_h", None)
    (out_dir / "metrics.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    (out_dir / "training_budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")


def plot_pr_calibration(run_dir: Path, out_dir: Path, names: list[str], tag: str, n_bins: int = 5) -> None:
    P = pd.read_csv(run_dir / "predictions.csv")
    te = P[(P.split == "test") & (P.label >= 0)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for n in names:
        pr, rc, _ = precision_recall_curve(te.label, te[f"p_{n}"])
        ax[0].plot(rc, pr, label=LABELS[n], lw=2 if n == "proposed" else 1.2)
        edges = np.linspace(0, 1, n_bins + 1)
        b = np.clip(np.digitize(te[f"p_{n}"], edges[1:-1]), 0, n_bins - 1)
        xs = [te[f"p_{n}"][b == k].mean() for k in range(n_bins) if (b == k).sum() >= 5]
        ys = [te.label[b == k].mean() for k in range(n_bins) if (b == k).sum() >= 5]
        ax[1].plot(xs, ys, marker="o", ms=3, label=LABELS[n])
    ax[0].axhline(te.label.mean(), ls=":", c="grey", label=f"prevalence {te.label.mean():.3f}")
    ax[0].set(xlabel="recall", ylabel="precision", title="Precision-recall (test windows)")
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[1].set(xlabel="mean predicted risk", ylabel="observed event rate",
              title=f"Calibration ({n_bins} bins, bins with <5 windows omitted)")
    ax[0].legend(fontsize=7)
    fig.suptitle(tag, fontsize=9, color="firebrick")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "pr_calibration.png", dpi=130)
    plt.close(fig)


def plot_trajectory(traj: pd.DataFrame, glucose: pd.DataFrame, thr: float, out_path: Path, tag: str,
                    horizon_h: float = 6.0) -> None:
    """One stay: risk vs threshold, decisions, glucose results at their AVAILABILITY time, event time."""
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})
    ax[0].plot(traj.t_h, traj.risk, c="tab:blue", label="calibrated risk")
    ax[0].fill_between(traj.t_h, traj.risk - traj.disagreement, traj.risk + traj.disagreement, alpha=0.2,
                       label="ensemble disagreement (+/- std)")
    ax[0].axhline(thr, c="tab:red", ls="--", lw=1, label=f"alert threshold {thr:.2f}")
    for dec, colour in (("ALERT", "tab:red"), ("QUEUE_REVIEW", "tab:orange")):
        m = traj.decision == dec
        ax[0].scatter(traj.t_h[m], traj.risk[m], c=colour, zorder=5, label=dec.lower().replace("_", " "))
    ev = float(traj.event_time_h.iloc[0])
    if np.isfinite(ev):
        for a in ax:
            a.axvline(ev, c="black", lw=1.5)
        ax[0].text(ev, ax[0].get_ylim()[1] * 0.92, " event (first glucose > 180)", ha="right", fontsize=8)
    ax[0].set(ylabel=f"P(excursion within {horizon_h:g} h)", ylim=(0, 1.0))
    ax[0].legend(fontsize=7, loc="upper left")
    ax[1].axhline(180, c="tab:red", ls="--", lw=1)
    ax[1].scatter(glucose.t_avail, glucose.value, c="tab:green", s=22, label="glucose at availability time")
    ax[1].scatter(glucose.t_chart, glucose.value, facecolors="none", edgecolors="grey", s=22,
                  label="glucose at draw time")
    for r in glucose.itertuples():
        ax[1].plot([r.t_chart, r.t_avail], [r.value, r.value], c="grey", lw=0.6)
    ax[1].set(xlabel="hours since ICU admission", ylabel="glucose, mg/dL")
    ax[1].set_xlim(left=max(0.0, float(glucose.t_chart.min()) - 1) if len(glucose) else 0.0)
    ax[1].legend(fontsize=7)
    fig.suptitle(tag, fontsize=9, color="firebrick")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_federated_md(fed_json: Path, out_dir: Path, label: str) -> None:
    r = json.loads(fed_json.read_text(encoding="utf-8"))
    clients = pd.DataFrame(r["clients"]).rename(columns={"id": "client", "n": "train windows", "events": "positive windows"})
    cmp_ = pd.DataFrame([
        {"training": f"FedAvg ({r['n_clients']} clients, {r['rounds']} rounds x {r['local_epochs']} local epochs)",
         **{k: _f(r["federated_final"][k]) for k in ("auroc", "average_precision", "brier")}},
        {"training": f"Pooled (same architecture, {r['pooled_epochs_matched']} epochs)",
         **{k: _f(r["pooled"][k]) for k in ("auroc", "average_precision", "brier")}}])
    hist = pd.DataFrame([{"round": h["round"], "test AUROC": _f(h["auroc"]), "test AP": _f(h["average_precision"])}
                         for h in r["federated_history"]])
    s = r["shared_per_client_per_round"]
    text = "\n".join([
        f"# Federated averaging simulation - {label}\n",
        "**Simulation only.** Clients are patient-disjoint partitions of the training patients on one machine. "
        f"Each client shares {s['parameters']:,} model parameters (~{s['approx_megabytes']} MB) per round plus its "
        "training-window count (the averaging weight); raw windows are not exchanged in the simulation. "
        "FedAvg alone gives no formal privacy guarantee.\n",
        "## Clients\n", _md(clients), "\n## Same test patients: federated vs pooled\n", _md(cmp_),
        f"\nTest windows: {r['federated_final']['n_test_windows']} ({r['federated_final']['n_test_positive']} positive). "
        "One training seed; differences between the rows are within what seed noise can produce on a cohort this small.\n",
        "## Federated test performance by round\n", _md(hist)])
    (out_dir / "federated.md").write_text(text + "\n", encoding="utf-8")


def write_routing_md(route_json: Path, out_dir: Path) -> None:
    r = json.loads(route_json.read_text(encoding="utf-8"))
    fx, lq = r["fixed_threshold"], r["learned_q_policy"]["aggregate"]
    rows = [{"policy": f"fixed threshold >= {fx['threshold']:.2f} -> {fx['action']} (tuned on training episodes)",
             "mean return": _f(fx["heldout"]["mean_return"], 2),
             "timely detection": _f(fx["heldout"]["timely_detection_rate"], 2),
             "reviews / patient-day": _f(fx["heldout"]["reviews_per_patient_day"], 2),
             "median lead after review h": _f(fx["heldout"]["median_lead_after_review_h"], 1)},
            {"policy": f"tabular Q-learning (mean +/- sd over {r['learned_q_policy']['n_seeds']} seeds)",
             "mean return": f"{lq['mean_return']['mean']:.2f} +/- {lq['mean_return']['sd']:.2f}",
             "timely detection": f"{lq['timely_detection_rate']['mean']:.2f} +/- {lq['timely_detection_rate']['sd']:.2f}",
             "reviews / patient-day": f"{lq['reviews_per_patient_day']['mean']:.2f} +/- {lq['reviews_per_patient_day']['sd']:.2f}",
             "median lead after review h": _f(lq["median_lead_after_review_h"]["mean"], 1)}]
    rw = r["reward"]
    text = "\n".join([
        "# Alert-routing simulation (SYNTHETIC trajectories)\n",
        "Review prioritisation demonstration. Rewards and costs are invented release choices; nothing here supports "
        "a claim of patient benefit or treatment selection.\n",
        f"Actions: wait / queue_review (cost {rw['cost']['queue_review']}, completes in "
        f"{rw['review_delay_h']['queue_review']} h) / "
        f"flag_earlier_review (cost {rw['cost']['flag_earlier_review']}, completes in "
        f"{rw['review_delay_h']['flag_earlier_review']} h). Reward +{rw['detect']} if a review completes 0 < lead <= "
        f"{rw['horizon_h']:g} h before the event; {rw['miss']} if the event is not preceded by a completed review. "
        "Feedback is delayed until the review completes.\n",
        f"Held-out synthetic patients: {r['n_eval_patients']}; training episodes per seed: {r['n_train_episodes']}.\n",
        _md(pd.DataFrame(rows))])
    (out_dir / "routing.md").write_text(text + "\n", encoding="utf-8")


def write_registry_md(reg_json: Path, out_dir: Path) -> None:
    r = json.loads(reg_json.read_text(encoding="utf-8"))
    rows = [{"version": h["version"], "status": h["status"], "parent": h["parent"] or "-",
             "val AP": _f(h["val_metrics"]["val_average_precision"]), "val Brier": _f(h["val_metrics"]["val_brier"]),
             "reason": h["reason"]} for h in r["history"]]
    text = ("# Offline update log (promotion gate on validation patients only)\n\n"
            f"Serving version: **{r['promoted']}**\n\n" + _md(pd.DataFrame(rows)))
    (out_dir / "registry.md").write_text(text + "\n", encoding="utf-8")
