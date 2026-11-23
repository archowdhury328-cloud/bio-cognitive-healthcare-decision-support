"""Fail if tracked files OR any commit in history contain tokens, private keys or patient-record files.

Run from the repository root:  python scripts/scan_repo.py
"""

from __future__ import annotations

import re
import subprocess
import sys

TOKEN_PATTERNS = {
    "github token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b|github_pat_[A-Za-z0-9_]{20,}"),
    "aws access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
}
# path fragments that indicate raw clinical records or patient-level outputs
FORBIDDEN_PATHS = (
    "data/raw/", "runs_local/", "local_outputs/", "mimic-iv-clinical-database-demo", "labevents", "chartevents",
)
ALLOWED_CSV_GZ = {"data/fixture/events.csv.gz"}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, encoding="utf-8").stdout


def main() -> int:
    problems: list[str] = []
    tracked = git("ls-files").splitlines()
    history_paths = set(git("log", "--all", "--name-only", "--pretty=format:").splitlines()) - {""}
    for p in sorted(set(tracked) | history_paths):
        low = p.lower()
        if any(f in low for f in FORBIDDEN_PATHS) or (low.endswith(".csv.gz") and p not in ALLOWED_CSV_GZ):
            problems.append(f"patient-record-like path in tree/history: {p}")
    revs = git("rev-list", "--all").split()
    diff = git("log", "--all", "-p", "--no-color", "--pretty=format:@@ %H", "--", ".", ":!*.gz", ":!*.png")
    commit = "?"
    for line in diff.splitlines():
        if line.startswith("@@ "):
            commit = line[3:11]
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for name, pat in TOKEN_PATTERNS.items():
            if pat.search(line):
                problems.append(f"{name} in history (commit {commit}): {line[:60]}...")
    for p in problems:
        print("FAIL", p)
    print(f"scanned {len(tracked)} tracked files and {len(revs)} commits: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
