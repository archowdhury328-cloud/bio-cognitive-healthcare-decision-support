# Paper-to-code map

**Paper:** Bio-Cognitive AI Systems for Predictive Healthcare Decision Support
**Venue:** 2025 1st International Conference on Advancement in Futuristic Technologies (ICAFT), Belagavi, India, 19-21 December 2025. IEEE.
**DOI:** [10.1109/ICAFT66710.2025.11453175](https://doi.org/10.1109/ICAFT66710.2025.11453175)
**IEEE Xplore listing date:** 31 March 2026 (supplied by the requester; the Crossref record for this DOI was created 2026-03-31, which agrees).

## What was verified, and what was not

| Item | Status |
|---|---|
| DOI resolves to this exact title | Verified through the Crossref REST record (`api.crossref.org/works/10.1109/ICAFT66710.2025.11453175`), 2026-11-24. |
| Author list and order, venue, dates, 28 references, pages 1-9 | Verified from the same record (see `CITATION.cff`). |
| Abstract | **Not retrievable.** Crossref carries none and the DOI landing page is not machine-readable. The abstract-level description in this document is the one supplied with the task, not text read from the IEEE record. |
| Full text | **Not accessed.** No licensed copy was supplied. Every statement below about what the paper specifies is therefore limited to the abstract-level description. |

Other papers share similar titles. Nothing in this repository was taken from any source other than the DOI above and the cited public datasets.

## What the paper does and does not specify (as available here)

Described, without numerical estimates: multimodal physiological signals, EHR variables and longitudinal biomarkers; deep learning; cognitive reasoning; adaptive learning; uncertainty estimation; federated learning; reinforcement learning; early warnings and risk trajectories; improvements in accuracy, sensitivity and time-to-detection.

**Unspecified in what was available (recorded as unspecified, not guessed):** the original cohort and its source, disease targets and prediction horizon, train/validation/test split, network layers and sizes, comparator list, and every numerical result.

Consequence: **this repository does not reproduce, and makes no claim about, the paper's reported performance.** It is a small, tested, research reference implementation of the documented design ideas, on a task chosen by this release (see below), with a different target than any chronic-disease endpoint the paper may use.

## Release-chosen task (not the paper's endpoint)

> At hourly landmarks after six observed ICU hours, estimate whether the first subsequently recorded glucose value above 180 mg/dL will occur within the next six hours.

This is a **recorded laboratory excursion**, not a diabetes diagnosis, and it is used only because it can be built from the public demo data. The threshold (180 mg/dL, strict), horizon (6 h) and first landmark (6 h) are release choices stored in `configs/glucose_excursion.yaml`.

## Component map

Status key: **implemented** = code exists and is unit-tested; **choice** = the paper is silent, so this release made a documented design choice; **not claimed** = no evidence is offered for the paper-level claim.

| Paper-level component | Source in paper | Code | Config | Verification status |
|---|---|---|---|---|
| Multimodal inputs: physiological signals (vitals) | abstract only; sensors/rates unspecified | `src/biocog/schema.py` (`VITALS`), `src/biocog/mimic.py` (chart events), `src/biocog/synth.py` | `features.*` | **implemented**, **choice** (5 vitals: HR, SBP, RR, SpO2, temperature) |
| EHR variables | abstract only | static context `age, sex, emergency` in `features.py`; `mimic.py` (patients, admissions) | - | **implemented**, **choice** (3 static fields) |
| Longitudinal biomarkers | abstract only | `BIOMARKERS` in `schema.py` (glucose, creatinine, sodium, potassium, WBC, BUN) | `features.lookback_h` | **implemented**, **choice** |
| Missing / absent modalities | not specified | masks + time-gap channels (`features.py`), availability flags (`nets.py`) | - | **implemented**; tests `test_rules.py`, `test_e2e_explain_synth_cli.py` |
| Deep learning model | layers unspecified | `ProposedNet` in `src/biocog/nets.py` (static MLP + two masked GRUs + learned gates) | `models.*` | **implemented**, **choice** (architecture is this release's) |
| Gated multimodal fusion | abstract-level only | `ProposedNet(fusion="gated")`; concat ablation `proposed_concat` | `models.hidden` | **implemented**; ablation in `results/*/results.md` |
| Cognitive reasoning | abstract-level only | `src/biocog/rules.py`: versioned rules R1-R4 (a deterministic rule layer, **not** a cognitive architecture) | `rules.*` | **implemented**; per-rule tests in `tests/test_rules.py` |
| Adaptive learning | abstract-level only | `src/biocog/registry.py`: offline warm-start update + validation promotion gate + version log | `registry` args | **implemented** as offline updates only; nothing adapts online |
| Uncertainty estimation | abstract-level only | seeded ensemble disagreement + Platt calibration on validation patients (`nets.py`, `calibrate.py`, `explain.py`) | `models.n_members`, `rules.review_disagreement` | **implemented**; calibrated risk, disagreement and missing-input flags are reported separately |
| Federated learning | abstract-level only | `src/biocog/federated.py`: FedAvg simulation, patient-disjoint clients, sample-count weighting, pooled comparison | `federated.*` | **implemented** as a one-machine simulation; no formal privacy guarantee |
| Reinforcement learning / pathway selection | abstract-level only | `src/biocog/routing.py`: offline alert-routing simulator + tabular Q-learning | `routing.*` | **implemented** on synthetic trajectories as review prioritisation; **not** treatment selection |
| Early warnings / risk trajectories | abstract-level only | `explain.trajectory`, `alerts.py`, `report.plot_trajectory` | `alerts.*` | **implemented** |
| Accuracy / sensitivity / time-to-detection improvements | numbers not available | `metrics.py` (window metrics), `alerts.py` (lead time) | `metrics.*` | **not claimed** - this repository reports its own measurements only; see `results/` |
| Original cohort, disease targets, split, architecture, numerical results | **unspecified** | - | - | **not claimed** |

## Differences from the paper

All of them, because the paper's own specification could not be read: cohort, target, horizon, split, layer sizes, comparators, and metrics are this release's choices. If a full text is obtained, update this table before citing any number from this repository next to the paper.

## Where claims live

- `results/paper_claims/` - what the paper is said to claim, with no numbers (none were available). Nothing in it is a measurement.
- `results/mimic_demo/` - measurements on the MIMIC-IV Clinical Database Demo v2.2 (aggregates only). Real data, 100 patients, tiny event counts.
- `results/synthetic/` - measurements on generated data. Every file and figure is labelled SYNTHETIC. These show the software works; they are not evidence about patients.
