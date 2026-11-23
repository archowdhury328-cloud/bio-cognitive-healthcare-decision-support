# Commands executed and measured coverage

Environment: Windows 11, Python 3.10.11, CPU only (PyTorch 2.13.0+cpu), numpy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2. No GPU, CUDA or Docker was exercised.

| command | outcome |
|---|---|
| `python -m biocog generate-synthetic` then `python scripts/update_manifest.py` | fixture written; `verify-data --which fixture` ok (3/3) |
| `python -m biocog fetch-data` | MIMIC-IV demo v2.2 downloaded, 34/34 files match PhysioNet SHA-256, extracted |
| `bash scripts/reproduce.sh all` | exit 0: cohort, ablate (5 models + 4 ablations), federated, update, report for `synthetic` and `mimic`; `route` for synthetic. About 1 h on this CPU |
| `python -m biocog demo --out demo_output` | succeeds; output in `docs/expected_demo_output.txt` |
| `ruff check src tests scripts` | all checks passed |
| `python -m pytest --cov=biocog` | **83 passed** in 237 s; **95% line coverage** (1794 statements, 89 missed; missed lines are mostly CLI error branches and the network-download path) |
| mutation check: replace `avail <= t` with `chart <= t` in `features.py` | 4 leakage tests fail; restored, all pass |
| `python scripts/scan_repo.py` | see final report for the result at release time |

Not exercised: GitHub Actions CI on a runner (the workflow is committed; its first run is the remote one), Linux, multi-GPU, any non-demo clinical data.
