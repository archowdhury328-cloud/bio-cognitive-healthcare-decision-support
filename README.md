# bio-cognitive-healthcare-decision-support

A small, tested **research reference implementation** of design ideas described at abstract level in
*Bio-Cognitive AI Systems for Predictive Healthcare Decision Support* (ICAFT 2025, IEEE,
[doi:10.1109/ICAFT66710.2025.11453175](https://doi.org/10.1109/ICAFT66710.2025.11453175)): gated multimodal fusion of vitals,
biomarkers and EHR context; an auditable rule layer; ensemble uncertainty; offline model updates; a federated-averaging
simulation; and an alert-routing simulation.

> **Not a medical device. Not for clinical use, patient triage or automated care.** It does **not** reproduce the paper's
> results: the paper's cohort, endpoint, architecture and numbers were not available to this release
> ([what was and was not verified](docs/paper-to-code.md)). The task here is a release-chosen engineering example, a
> **recorded laboratory glucose excursion** (first glucose > 180 mg/dL within 6 h of an hourly landmark), not a diabetes
> diagnosis.

## One-command offline demo (no downloads, no credentials)

```bash
pip install -e ".[dev]"            # CPU is enough; see requirements.lock for the versions used
python -m biocog demo --out demo_output     # about 1-2 minutes on a laptop CPU
```

Real output from this repository (synthetic patient; files in [`results/synthetic/`](results/synthetic/)):

```text
SYNTHETIC DEMO (not real patients, not clinical advice)
stay S00018  landmark t=9 h after ICU admission  horizon 6 h
event: first recorded glucose > 180 mg/dL (laboratory excursion, not a diabetes diagnosis)
calibrated risk        : 0.182
model disagreement     : 0.127  (ensemble std; not a risk)
modalities observed    : static_ehr, vitals, biomarkers
modalities missing     : none
latest glucose age     : 1.1 h
rules activated        : R1_TREND_MIN_OBS (5 biomarker trends unavailable)
feature-group contribution to risk (occlusion):
    static            +0.038
    vitals            +0.090
    glucose           +0.082
    other_biomarkers  +0.021
    rule_features     +0.021
simulated decision     : ALERT
this alert is 11.1 h before the excursion, outside the 6 h horizon: the evaluation counts it as a FALSE alert
first TRUE alert (inside horizon) at t=16 h -> warning lead time 4.1 h before the recorded excursion at t=20.1 h
```

It walks one stay from timestamped observations to a risk trajectory, an explanation and a simulated alert decision
([figure](results/synthetic/trajectory_SYNTHETIC.png)). The stay is chosen by a fixed, outcome-blind rule (first test event stay
with at least 12 scored landmarks), not because it looks good; here the first alert was too early to count.

## What is in here

| need | where |
|---|---|
| Proposed model | [`src/biocog/nets.py`](src/biocog/nets.py) (`ProposedNet`), rule layer [`rules.py`](src/biocog/rules.py) |
| Baselines (last value, logistic, gradient-boosted trees, GRU) and data setup | [`baselines.py`](src/biocog/baselines.py), [`nets.py`](src/biocog/nets.py), [`data.py`](src/biocog/data.py), [`mimic.py`](src/biocog/mimic.py), [`docs/DATA_CARD.md`](docs/DATA_CARD.md) |
| Cohort, split, labels, leakage controls | [`cohort.py`](src/biocog/cohort.py), [`features.py`](src/biocog/features.py) |
| Alert policy, lead time | [`alerts.py`](src/biocog/alerts.py) |
| Federated simulation, alert-routing RL, offline updates | [`federated.py`](src/biocog/federated.py), [`routing.py`](src/biocog/routing.py), [`registry.py`](src/biocog/registry.py) |
| Measured results | [`results/synthetic/`](results/synthetic/results.md), [`results/mimic_demo/`](results/mimic_demo/results.md) |
| Paper-to-code map, data card, model card, data dictionary | [`docs/`](docs/) |

## Commands (each verified; see [`docs/commands_executed.md`](docs/commands_executed.md))

```bash
python -m biocog fetch-data                              # MIMIC-IV demo v2.2, SHA-256 verified, no credentials
python -m biocog verify-data --which mimic               # or --which fixture
python -m biocog cohort   --dataset mimic                # cohort, patient split, windows, features
python -m biocog train    --dataset mimic                # baselines + GRU + proposed (calibrated on validation patients)
python -m biocog evaluate --dataset mimic                # validation-selected policy, test evaluation
python -m biocog ablate   --dataset mimic                # + four ablations, then evaluate
python -m biocog federated --dataset mimic               # FedAvg simulation vs pooled
python -m biocog route --out runs_local/routing          # offline alert-routing experiment (synthetic)
python -m biocog update   --dataset mimic                # offline update with validation promotion gate
python -m biocog report --run runs_local/mimic --out results/mimic_demo
python -m biocog score --run runs_local/fixture --stays S.csv --events E.csv --confirm-synthetic   # synthetic tables only
bash scripts/reproduce.sh all                            # everything above, both datasets (about 1 h on CPU)
```

`--dataset` is `fixture` (committed, synthetic, 120 patients), `synthetic` (generated benchmark, 400 patients) or `mimic`.
The `score` interface accepts only canonical synthetic tables, refuses without `--confirm-synthetic`, and rejects extra
columns; do not upload identifiable patient data anywhere.

## Measured results (read the caveats)

All numbers are test-patient measurements from the runs in `results/`; nothing is tuned to make the proposed model win.

**Synthetic benchmark (generated data; shows the software works, not anything about patients).** 400 patients, 22 test events, shared alert policy
(cooldown 6 h, validation budget 0.5 false alerts per patient-day):

| model | AUROC [95% CI] | AP | events detected | median lead h |
|---|---|---|---|---|
| Last glucose value | 0.834 [0.767, 0.889] | 0.204 | 21/22 | 2.2 |
| Logistic regression | 0.878 [0.823, 0.927] | 0.240 | 21/22 | 2.2 |
| Gradient-boosted trees | 0.847 [0.777, 0.908] | 0.170 | 18/22 | 2.6 |
| GRU comparator | 0.903 [0.854, 0.943] | 0.326 | 20/22 | 2.6 |
| Proposed gated multimodal | 0.891 [0.836, 0.937] | 0.310 | 21/22 | 3.0 |

The proposed model **does not beat the GRU comparator** (overlapping intervals); removing biomarkers hurts most (AUROC 0.737,
14/22 detected); the concat-fusion and no-reasoning ablations are within noise. FedAvg (4 clients) scored AUROC 0.879 vs 0.786
for pooled training with a matched epoch budget, one seed: do not read that as federated being better. The tabular-Q routing
policy ties the tuned fixed threshold on return (-1.22 vs -1.23) at a different detection/burden operating point.

**MIMIC-IV Clinical Database Demo (real data, 100 patients).** The cohort is tiny: 98 eligible stays, 20 event patients, **4 test event patients**.
Confidence intervals are therefore withheld (fewer than 10 positive patients) and every number below is anecdote, not evidence:

| model | AUROC | events detected |
|---|---|---|
| Last glucose value | 0.919 | 2/4 |
| Logistic regression | 0.783 | 4/4 |
| Gradient-boosted trees | 0.383 | 2/4 |
| GRU comparator | 0.538 | 0/4 |
| Proposed gated multimodal | 0.399 | 0/4 |

On this sample the neural models do **worse** than the one-line last-value baseline. That is retained as observed; with 4 event
patients no conclusion about any model is supportable. Most hourly windows are unknown (glucose not measured), because of
selective testing; see the [data card](docs/DATA_CARD.md).

## Leakage controls and tests

83 tests, **95% line coverage** (CI fails below 80%). Notable ones: changing every post-landmark observation leaves features and
predictions identical; a lab drawn before a landmark but released after it is invisible (both tests fail if availability logic is
broken, which was checked by mutation); patient-disjoint splits assigned before windows; timestamp boundaries; unknown labels;
calibration fitted on validation patients only; lead-time arithmetic; missed-event denominators; cooldown suppression;
sample-weighted FedAvg; saved-model reloads; a promotion gate that never reads test patients; a clean-checkout end-to-end run.
Raw clinical files and patient-level outputs are git-ignored; `scripts/scan_repo.py` and CI scan tracked files and history for
tokens and record-like files.

## Provenance and authorship

- **Paper authors (verified from the Crossref record, 2026-11-24):** Syed Mohammed Muhive Uddin, Md Asikur Rahman Chy, Tawfiqur Rahman Sikder,
  Md Imtiaz Faruk, Muhammad Adnan, Md Jubayar Hossain, Mia Md. Tofayel Gonee Manik. IEEE Xplore listing date supplied by the requester: 31 March 2026.
- **Maintainer:** Md Asikur Rahman Chy ([GitHub](https://github.com/archowdhury328-cloud),
  [Google Scholar](https://scholar.google.com/citations?user=mAM8u_UAAAAJ&hl=en)), repository maintainer and release owner; the only human
  contributor of commits in this repository.
- **Coding assistance disclosure:** the code, tests and documentation in this release were produced with AI coding assistance and committed
  on the maintainer's behalf. It is a new implementation written in November 2026; it is not code from the paper's authors and not a record of
  work done in 2025.
- **Dataset credit:** MIMIC-IV Clinical Database Demo v2.2, PhysioNet (ODbL 1.0); see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
- **Licence:** none granted yet; the maintainer has not chosen one.

Cite via [`CITATION.cff`](CITATION.cff).
