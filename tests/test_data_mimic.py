"""Manifest verification, guarded fetch, and the MIMIC adapter on a tiny hand-made tree (no credentials)."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import zipfile

import pandas as pd
import pytest

from biocog import data, datasets, mimic


def _gz(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        df.to_csv(fh, index=False)


@pytest.fixture
def mini_tree(tmp_path):
    r = tmp_path / "mimic"
    _gz(r / "icu/icustays.csv.gz", pd.DataFrame({
        "subject_id": [1, 2], "hadm_id": [10, 20], "stay_id": [100, 200], "first_careunit": "MICU",
        "last_careunit": "MICU", "intime": ["2150-03-01 08:00:00", "2150-04-01 08:00:00"],
        "outtime": ["2150-03-03 08:00:00", "2150-04-02 08:00:00"], "los": [2.0, 1.0]}))
    _gz(r / "hosp/patients.csv.gz", pd.DataFrame({
        "subject_id": [1, 2], "gender": ["M", "F"], "anchor_age": [60, 40], "anchor_year": [2148, 2150],
        "anchor_year_group": "2008 - 2010", "dod": None}))
    _gz(r / "hosp/admissions.csv.gz", pd.DataFrame({
        "subject_id": [1, 2], "hadm_id": [10, 20], "admittime": ["2150-03-01 06:00:00", "2150-04-01 06:00:00"],
        "admission_type": ["EMERGENCY", "ELECTIVE"]}))
    chart = pd.DataFrame({
        "stay_id": [100, 100, 100, 100, 100],
        "itemid": [220045, 223761, 220045, 220179, 999999],
        "charttime": ["2150-03-01 09:00:00"] * 5,
        "storetime": ["2150-03-01 08:50:00", "2150-03-01 09:05:00", "2150-03-01 09:30:00",
                      "2150-03-01 09:10:00", "2150-03-01 09:00:00"],
        "valuenum": [80.0, 98.6, 400.0, 120.0, 5.0]})
    _gz(r / "icu/chartevents.csv.gz", chart)
    lab = pd.DataFrame({
        "hadm_id": [10, 10, 10, 10, None],
        "itemid": [50931, 50931, 50809, 50912, 50931],
        "charttime": ["2150-03-01 10:00:00"] * 5,
        "storetime": ["2150-03-01 12:30:00", "2150-03-01 12:30:00", "2150-03-01 12:00:00",
                      "2150-03-01 13:00:00", "2150-03-01 12:00:00"],
        "valuenum": [200.0, 11.1, 150.0, 1.1, 99.0],
        "valueuom": ["mg/dL", "mmol/L", "mg/dL", "mg/dL", "mg/dL"]})
    _gz(r / "hosp/labevents.csv.gz", lab)
    return r


def test_adapter_units_availability_and_exclusions(mini_tree):
    stays, ev, rep = mimic.to_canonical(mini_tree)
    s = stays.set_index("stay_id")
    assert s.loc["100", "age"] == 62 and s.loc["100", "sex_male"] == 1 and s.loc["100", "emergency"] == 1
    assert s.loc["200", "emergency"] == 0 and s.loc["200", "age"] == 40
    temp = ev[ev.variable == "temp_c"].iloc[0]
    assert temp.value == pytest.approx(37.0, abs=1e-6)  # 98.6 F -> C
    hr = ev[ev.variable == "hr"]
    assert list(hr.value) == [80.0]  # 400 bpm dropped as out of range
    assert rep["chart_rows_dropped_out_of_range"] == 1
    # storetime (08:50) precedes charttime (09:00): availability is floored at the draw time
    assert hr.iloc[0].availtime == pd.Timestamp("2150-03-01 09:00:00")
    # temperature stored 5 minutes after it was measured: available at storetime
    assert temp.availtime == pd.Timestamp("2150-03-01 09:05:00")
    glu = ev[ev.variable == "glucose"]
    assert list(glu.value) == [200.0]  # single source 50931; 50809 ignored; mmol/L row dropped; no-hadm row excluded
    assert glu.iloc[0].availtime == pd.Timestamp("2150-03-01 12:30:00")  # lab availability = storetime
    assert rep["lab_rows_dropped_nonconforming_unit"] == 1 and rep["lab_rows_without_hadm_id_excluded"] == 1
    assert "sbp" in set(ev.variable) and 999999 not in set(ev.get("itemid", []))


def test_adapter_output_feeds_cohort_builder(mini_tree, cfg):
    from biocog import cohort
    stays, ev, _ = mimic.to_canonical(mini_tree)
    st, e = cohort.to_hours(stays, ev)
    tab = cohort.stay_table(st, e, cfg)
    assert set(tab.stay_id) == {"100", "200"} and e.t_avail.ge(e.t_chart).all()


def test_manifest_covers_fixture_and_full_mimic_demo():
    man = data.load_manifest()
    assert len(man["datasets"]["mimic_iv_demo_2.2"]["files"]) == 34
    assert man["datasets"]["mimic_iv_demo_2.2"]["license"].startswith("ODC Open Database License")
    assert data.verify_dataset("fixture", datasets.FIXTURE_DIR)["ok"]


def test_verify_detects_tampering_and_missing(tmp_path):
    dst = tmp_path / "fx"
    shutil.copytree(datasets.FIXTURE_DIR, dst)
    assert data.verify_dataset("fixture", dst)["ok"]
    with open(dst / "stays.csv", "ab") as fh:
        fh.write(b"x")
    (dst / "truth.csv").unlink()
    res = data.verify_dataset("fixture", dst)
    assert not res["ok"] and res["mismatch"] == ["stays.csv"] and res["missing"] == ["truth.csv"]


def _fake_manifest(files):
    return {"datasets": {"mimic_iv_demo_2.2": {"files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}}}


def _zip_bytes(files, root=data.ZIP_ROOT):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, v in files.items():
            z.writestr(f"{root}/{k}", v)
    return buf.getvalue()


def test_fetch_verifies_then_extracts(monkeypatch, tmp_path):
    files = {"hosp/a.csv.gz": b"aaa", "icu/b.csv.gz": b"bbb"}
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(_zip_bytes(files)))
    man = _fake_manifest(files)
    dest = tmp_path / "raw" / "mimic"
    monkeypatch.setitem(man["datasets"], "mimic_iv_demo_2.2", man["datasets"]["mimic_iv_demo_2.2"])
    monkeypatch.setattr(data, "load_manifest", lambda *a, **k: man)
    res = data.fetch_mimic_demo(dest, man)
    assert res["ok"] and (dest / "hosp/a.csv.gz").read_bytes() == b"aaa"


def test_fetch_refuses_on_checksum_mismatch(monkeypatch, tmp_path):
    files = {"hosp/a.csv.gz": b"aaa"}
    bad = _zip_bytes({"hosp/a.csv.gz": b"EVIL"})
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(bad))
    dest = tmp_path / "raw" / "mimic"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        data.fetch_mimic_demo(dest, _fake_manifest(files))
    assert not (dest / "hosp/a.csv.gz").exists()


def test_manifest_json_is_valid_and_fixture_has_no_clinical_claim():
    man = json.loads(data.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert man["datasets"]["fixture"]["synthetic"] is True
