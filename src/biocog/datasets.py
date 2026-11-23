"""Dataset loaders returning canonical ``(stays, events)``; fixture writing with stable bytes."""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import pandas as pd

from . import synth
from .data import MIMIC_DIR, ROOT

FIXTURE_DIR = ROOT / "data" / "fixture"
FIXTURE_PATIENTS, FIXTURE_SEED = 120, 7
BENCH_PATIENTS, BENCH_SEED = 400, 11


def write_fixture(out: Path, n_patients: int = FIXTURE_PATIENTS, seed: int = FIXTURE_SEED) -> None:
    """Write stays.csv, events.csv.gz (mtime=0 so the bytes are reproducible) and truth.csv."""
    out.mkdir(parents=True, exist_ok=True)
    stays, events, truth = synth.generate(n_patients, seed)
    eol = chr(10)
    stays.to_csv(out / "stays.csv", index=False, lineterminator=eol)
    truth.to_csv(out / "truth.csv", index=False, lineterminator=eol)
    buf = io.StringIO()
    events.to_csv(buf, index=False, lineterminator=eol)
    with open(out / "events.csv.gz", "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(buf.getvalue().encode("utf-8"))


def load_fixture(path: Path = FIXTURE_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    stays = pd.read_csv(path / "stays.csv", parse_dates=["intime", "followup_end"])
    events = pd.read_csv(path / "events.csv.gz", parse_dates=["charttime", "availtime"])
    return stays, events


def load_dataset(name: str, mimic_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """``fixture`` (committed, synthetic), ``synthetic`` (larger generated benchmark) or ``mimic``."""
    if name == "fixture":
        s, e = load_fixture()
        return s, e, {"dataset": "fixture", "synthetic": True}
    if name == "synthetic":
        s, e, _ = synth.generate(BENCH_PATIENTS, BENCH_SEED)
        return s, e, {"dataset": "synthetic", "synthetic": True, "n_patients": BENCH_PATIENTS, "seed": BENCH_SEED}
    if name == "mimic":
        from .mimic import to_canonical
        s, e, rep = to_canonical(mimic_dir or MIMIC_DIR)
        return s, e, {"dataset": "mimic_iv_demo_2.2", "synthetic": False, "adapter_report": rep}
    raise ValueError(f"unknown dataset {name!r}")
