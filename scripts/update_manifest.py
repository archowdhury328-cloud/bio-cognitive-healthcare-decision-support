"""Recompute fixture checksums in data/manifest.json (run after `biocog generate-synthetic`).

MIMIC demo checksums are copied from PhysioNet's published SHA256SUMS and are never recomputed here.
"""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
mp = root / "data" / "manifest.json"
man = json.loads(mp.read_text(encoding="utf-8"))
man["datasets"]["fixture"]["files"] = {
    f: hashlib.sha256((root / "data" / "fixture" / f).read_bytes()).hexdigest()
    for f in ("stays.csv", "events.csv.gz", "truth.csv")}
mp.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
print(man["datasets"]["fixture"]["files"])
