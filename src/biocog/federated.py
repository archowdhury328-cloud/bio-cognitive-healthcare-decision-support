"""Federated averaging simulation on patient-disjoint clients, compared with pooled training.

THIS IS A SIMULATION on one machine. "Clients" are random patient-disjoint partitions of the
training patients; no network, no real sites. What each simulated client shares per round: the full
set of model parameters after local training (and its training-window count, used as the averaging
weight). Raw windows never leave a client in the simulation.

Limits (read before citing): FedAvg alone gives NO formal privacy guarantee. Model updates can leak
information about training data (membership inference, gradient inversion); formal guarantees would
need e.g. differential privacy and secure aggregation, neither of which is implemented here.
Normalisation statistics are fitted on the pooled training patients and shared with every client,
which a real deployment would have to replace with secure aggregation of statistics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .calibrate import PlattCalibrator
from .nets import fedavg, predict_logits, train_member
from .pipeline import BASE_SEED, Bundle


def partition_patients(patients: list[str], has_event: dict[str, bool], k: int, seed: int) -> list[list[str]]:
    """Patient-disjoint partition: event patients dealt round-robin first so every client sees some."""
    rng = np.random.default_rng(seed)
    ev = [p for p in sorted(patients) if has_event.get(p)]
    non = [p for p in sorted(patients) if not has_event.get(p)]
    ev = [ev[i] for i in rng.permutation(len(ev))]
    non = [non[i] for i in rng.permutation(len(non))]
    parts: list[list[str]] = [[] for _ in range(k)]
    for i, p in enumerate(ev + non):
        parts[i % k].append(p)
    return parts


def _eval(state, kind, b: Bundle, pre) -> dict[str, Any]:
    va, te = b.feats.split("val").labelled(), b.feats.split("test").labelled()
    d_va, d_te = pre.tensors(va), pre.tensors(te)
    calib = PlattCalibrator().fit(predict_logits(kind, state, d_va, b.cfg), va.index.label.to_numpy())
    raw = predict_logits(kind, state, d_te, b.cfg)
    y = te.index.label.to_numpy()
    p = calib.predict(raw)
    return {"auroc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "n_test_windows": len(y), "n_test_positive": int(y.sum())}


def run_federated(b: Bundle, kind: str = "proposed", seed: int = 0) -> dict[str, Any]:
    cfg, fc = b.cfg, b.cfg["federated"]
    tr = b.feats.split("train").labelled()
    pat = tr.index.patient_id.to_numpy()
    y_all = tr.index.label.to_numpy()
    has_event = {p: bool(y_all[pat == p].max() == 1) for p in set(pat)}
    parts = partition_patients(sorted(set(pat)), has_event, int(fc["n_clients"]), cfg["split"]["seed"] + seed)
    d_all = b.pre.tensors(tr)
    clients = []
    for k, ids in enumerate(parts):
        idx = np.where(np.isin(pat, ids))[0]
        clients.append({"id": k, "patients": len(ids), "idx": idx, "n": len(idx), "events": int(y_all[idx].sum())})
    assert not any(set(a["idx"]) & set(c["idx"]) for i, a in enumerate(clients) for c in clients[i + 1:])
    local_epochs, rounds = int(fc["local_epochs"]), int(fc["rounds"])
    # global initialisation: one throw-away epoch-0 member gives the shared starting weights
    g_state, _ = train_member(kind, {k: v[:8] for k, v in d_all.items()}, y_all[:8], None, None, cfg,
                              BASE_SEED + seed, epochs=0)
    history = []
    for r in range(rounds):
        states, weights = [], []
        for c in clients:
            if c["n"] == 0:
                continue
            sub = {k: v[c["idx"]] for k, v in d_all.items()}
            st, _ = train_member(kind, sub, y_all[c["idx"]], None, None, cfg, BASE_SEED + 1000 * r + c["id"],
                                 epochs=local_epochs, init_state=g_state)
            states.append(st)
            weights.append(c["n"])
        g_state = fedavg(states, weights)
        history.append({"round": r + 1, **_eval(g_state, kind, b, b.pre)})
    pooled, _ = train_member(kind, d_all, y_all, None, None, cfg, BASE_SEED + seed, epochs=rounds * local_epochs)
    n_params = int(sum(v.numel() for v in g_state.values()))
    return {
        "simulated": True, "kind": kind, "n_clients": len(clients),
        "clients": [{k: v for k, v in c.items() if k != "idx"} for c in clients],
        "rounds": rounds, "local_epochs": local_epochs,
        "pooled_epochs_matched": rounds * local_epochs,
        "federated_final": history[-1], "federated_history": history,
        "pooled": _eval(pooled, kind, b, b.pre),
        "shared_per_client_per_round": {"parameters": n_params, "approx_megabytes": round(n_params * 4 / 1e6, 4),
                                        "also_shares": "training-window count (averaging weight)"},
        "privacy_note": "simulation only; FedAvg provides no formal privacy guarantee",
    }
