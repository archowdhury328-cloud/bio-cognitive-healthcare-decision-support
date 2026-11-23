"""Run provenance: data version, exclusions, split seed, config, dependencies, hardware, commit SHA."""

from __future__ import annotations

import datetime as dt
import platform
import subprocess
from pathlib import Path
from typing import Any

from .config import config_hash

ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=15,
                              check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def collect(cfg: dict[str, Any], dataset: str, data_version: str, flow=None) -> dict[str, Any]:
    import numpy
    import pandas
    import sklearn
    import torch

    exclusions = {}
    if flow is not None:
        exclusions = {r.stage: int(r.n) for r in flow.itertuples() if str(r.stage).startswith("excluded:")}
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset, "data_version": data_version,
        "git_commit": (_git("rev-parse", "--verify", "-q", "HEAD") or None),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config_hash": config_hash(cfg), "split_seed": cfg["split"]["seed"],
        "cohort_exclusions": exclusions,
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "cuda_gpu": gpu,
        "packages": {"torch": torch.__version__, "numpy": numpy.__version__, "pandas": pandas.__version__,
                     "scikit-learn": sklearn.__version__},
    }
