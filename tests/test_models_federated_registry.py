"""Saved-model reloads, federated averaging, offline updates with a promotion gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from biocog import federated, nets, pipeline, synth
from biocog.registry import ModelRegistry, offline_update


@pytest.mark.parametrize("name", ["last_value", "logistic", "gbt"])
def test_baseline_reload_gives_identical_predictions(bundle, tmp_path, name):
    f = bundle.feats.split("test")
    model = pipeline.fit_model(name, bundle)
    calib = pipeline.fit_calibrator(model, bundle)
    pipeline.save_model(name, model, calib, tmp_path)
    m2, c2 = pipeline.load_model(name, tmp_path, bundle)
    assert np.array_equal(model.predict_raw(f), m2.predict_raw(f))
    assert np.allclose(calib.predict(model.predict_raw(f)), c2.predict(m2.predict_raw(f)))


def test_neural_ensemble_reload_is_exact_and_carries_metadata(bundle, tiny_proposed, tmp_path):
    f = bundle.feats.split("test")
    calib = pipeline.fit_calibrator(tiny_proposed, bundle)
    pipeline.save_model("proposed", tiny_proposed, calib, tmp_path)
    ens, c2 = pipeline.load_model("proposed", tmp_path, bundle)
    assert np.allclose(tiny_proposed.predict_raw(f), ens.predict_raw(f), atol=1e-6)
    assert np.allclose(tiny_proposed.disagreement(f), ens.disagreement(f), atol=1e-6)
    assert ens.kind == "proposed" and ens.rules_on and ens.drop == () and (c2.a, c2.b) == (calib.a, calib.b)


def test_ensemble_is_seed_deterministic(bundle):
    a, b = pipeline.fit_model("proposed", bundle), pipeline.fit_model("proposed", bundle)
    f = bundle.feats.split("val")
    assert np.allclose(a.predict_raw(f), b.predict_raw(f), atol=1e-6)


def test_all_neural_variants_train_and_ablations_change_inputs(bundle):
    f = bundle.feats.split("test").take(np.arange(30))
    raw = {}
    for name in ("gru", "proposed_concat", "proposed_no_reasoning", "proposed_no_vitals", "proposed_no_biomarkers"):
        m = pipeline.fit_model(name, bundle)
        raw[name] = m.predict_raw(f)
        assert np.isfinite(raw[name]).all() and len(raw[name]) == 30
    d_nv = bundle.pre.tensors(f, drop=("vitals",))
    d_nb = bundle.pre.tensors(f, drop=("biomarkers",))
    assert d_nv["seq"][:, :, :5].sum() == 0 and (d_nv["avail"][:, 0] == 0).all()
    assert d_nb["seq"][:, :, 5:].sum() == 0 and (d_nb["avail"][:, 1] == 0).all() and np.abs(d_nb["rule"]).sum() == 0


def test_fedavg_is_sample_count_weighted():
    a = {"w": torch.tensor([1.0, 3.0]), "n": torch.tensor(5)}
    b = {"w": torch.tensor([5.0, 7.0]), "n": torch.tensor(9)}
    out = nets.fedavg([a, b], [1, 3])
    assert torch.allclose(out["w"], torch.tensor([4.0, 6.0]))  # 0.25*a + 0.75*b
    assert int(out["n"]) == 5  # integer buffers are not averaged
    assert torch.allclose(nets.fedavg([a, b], [1, 1])["w"], torch.tensor([3.0, 5.0]))


def test_partition_is_patient_disjoint_complete_and_spreads_events():
    pats = [f"P{i}" for i in range(40)]
    ev = {p: i % 4 == 0 for i, p in enumerate(pats)}
    parts = federated.partition_patients(pats, ev, 4, seed=0)
    flat = [p for part in parts for p in part]
    assert sorted(flat) == sorted(pats) and len(flat) == len(set(flat))
    assert all(any(ev[p] for p in part) for part in parts)
    assert federated.partition_patients(pats, ev, 4, seed=0) == parts


def test_federated_simulation_runs_and_labels_itself(bundle):
    res = federated.run_federated(bundle)
    assert res["simulated"] and "no formal privacy guarantee" in res["privacy_note"]
    assert sum(c["patients"] for c in res["clients"]) == bundle.windows[bundle.windows.split == "train"].patient_id.nunique()
    assert res["pooled_epochs_matched"] == res["rounds"] * res["local_epochs"]
    for k in ("auroc", "average_precision", "brier"):
        assert np.isfinite(res["federated_final"][k]) and np.isfinite(res["pooled"][k])
    assert res["federated_final"]["n_test_windows"] == res["pooled"]["n_test_windows"]  # same test patients


def _new_batch(bundle, n=15, seed=5):
    stays, events, _ = synth.generate(n, seed)
    for df in (stays,):
        df["patient_id"] = "N" + df.patient_id
    for df in (stays, events):
        df["stay_id"] = "N" + df.stay_id
    return pipeline.make_update_batch(stays, events, bundle.cfg)


def test_offline_update_rejects_when_gate_not_met_and_incumbent_stays(bundle, tiny_proposed, tmp_path):
    reg = ModelRegistry(tmp_path)
    res = offline_update(reg, bundle, tiny_proposed, _new_batch(bundle), min_ap_gain=10.0)
    assert res["promoted"] is False and res["serving_version"] == "v0001"
    hist = reg.state["history"]
    assert [h["status"] for h in hist] == ["promoted", "rejected"] and "AP" in hist[1]["reason"]
    assert ModelRegistry(tmp_path).state["promoted"] == "v0001"  # persisted


def test_offline_update_promotes_when_gate_met_and_reloads(bundle, tiny_proposed, tmp_path):
    reg = ModelRegistry(tmp_path)
    res = offline_update(reg, bundle, tiny_proposed, _new_batch(bundle), min_ap_gain=-1.0, brier_tolerance=1.0)
    assert res["promoted"] and res["serving_version"] == res["candidate_version"] == "v0002"
    ens, calib = reg.load("v0002", bundle)
    assert len(ens.predict_raw(bundle.feats.split("test"))) == len(bundle.feats.split("test"))


def test_update_batch_with_overlapping_patients_is_refused(bundle, tiny_proposed, tmp_path):
    stays, events, _ = synth.generate(5, 7)  # same ids as the original cohort
    batch = pipeline.make_update_batch(stays, events, bundle.cfg)
    with pytest.raises(ValueError, match="shares"):
        offline_update(ModelRegistry(tmp_path), bundle, tiny_proposed, batch)


def test_training_never_reads_test_windows(bundle):
    """Early stopping / calibration / thresholds use val only: dropping test rows leaves fits unchanged."""
    a = pipeline.fit_model("logistic", bundle)
    keep = np.where(bundle.feats.index.split.to_numpy() != "test")[0]
    import dataclasses
    sub = dataclasses.replace(bundle, feats=bundle.feats.take(keep), windows=bundle.windows.iloc[keep].reset_index(drop=True))
    b = pipeline.fit_model("logistic", sub)
    f = bundle.feats.split("val")
    assert np.allclose(a.predict_raw(f), b.predict_raw(f))
    assert isinstance(bundle.flow, pd.DataFrame)
