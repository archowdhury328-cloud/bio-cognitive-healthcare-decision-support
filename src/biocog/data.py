"""Data acquisition and checksum verification.

Nothing here invents a URL or a cohort. The public integration example is the openly available
MIMIC-IV Clinical Database Demo v2.2 on PhysioNet (100 de-identified patients, no free text).
The paper's own cohort is not identified in the information available to this release; it is
therefore NOT downloadable from here (see docs/paper-to-code.md).
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data" / "manifest.json"
MIMIC_DIR = ROOT / "data" / "raw" / "mimic-iv-demo-2.2"
MIMIC_ZIP_URL = "https://physionet.org/static/published-projects/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2.zip"
MIMIC_SUMS_URL = "https://physionet.org/files/mimic-iv-demo/2.2/SHA256SUMS.txt"
ZIP_ROOT = "mimic-iv-clinical-database-demo-2.2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_dataset(name: str, root: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Check every manifest file under ``root``. Returns {'ok': bool, 'missing': [...], 'mismatch': [...]}."""
    manifest = manifest or load_manifest()
    files = manifest["datasets"][name]["files"]
    missing, mismatch, ok = [], [], 0
    for rel, expected in files.items():
        p = root / rel
        if not p.exists():
            missing.append(rel)
        elif sha256_file(p) != expected:
            mismatch.append(rel)
        else:
            ok += 1
    return {"dataset": name, "checked": len(files), "verified": ok, "missing": missing,
            "mismatch": mismatch, "ok": not missing and not mismatch}


def fetch_mimic_demo(dest: Path = MIMIC_DIR, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Download the open demo zip, verify against the manifest checksums, extract to ``dest``.

    Requires internet access only; no credentials (this demo is openly licensed under ODbL 1.0).
    The extracted files are verified; on any mismatch the extraction is refused.
    """
    manifest = manifest or load_manifest()
    dest.parent.mkdir(parents=True, exist_ok=True)
    zpath = dest.parent / "mimic-iv-demo-2.2.zip"
    with urllib.request.urlopen(MIMIC_ZIP_URL, timeout=120) as resp, open(zpath, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    expected = manifest["datasets"]["mimic_iv_demo_2.2"]["files"]
    try:
        with zipfile.ZipFile(zpath) as z:
            for rel, digest in expected.items():
                got = hashlib.sha256(z.read(f"{ZIP_ROOT}/{rel}")).hexdigest()
                if got != digest:
                    raise RuntimeError(f"checksum mismatch for {rel}: refusing to extract")
            for rel in expected:
                target = (dest / rel).resolve()
                if dest.resolve() not in target.parents:
                    raise RuntimeError(f"unsafe path in archive: {rel}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(f"{ZIP_ROOT}/{rel}"))
    finally:
        zpath.unlink(missing_ok=True)
    return verify_dataset("mimic_iv_demo_2.2", dest, manifest)
