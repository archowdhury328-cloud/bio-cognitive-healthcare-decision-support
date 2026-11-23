# Model card

**Status: research reference implementation. Not a medical device. Not validated. Not for clinical use, patient triage or automated care.**

## Task

Estimate, at hourly ICU landmarks, the probability that a laboratory glucose value above 180 mg/dL will be recorded in the next 6 hours. A release-chosen engineering example; it is not a diabetes diagnosis and not the paper's endpoint (see `paper-to-code.md`).

## Models (every one has a config, `fit`, saved state, inference path and test)

| name | what it is | inputs |
|---|---|---|
| `last_value` | score = last observed glucose | glucose |
| `logistic` | L2 logistic regression; C chosen on validation patients | past-only summaries (last, 12 h mean, slope, count, age of last, static) |
| `gbt` | histogram gradient-boosted trees | same summaries |
| `gru` | temporal GRU over all variables with masks and time gaps | hourly grid, 24 h, static at the head |
| `proposed` | static MLP + masked vitals GRU + masked biomarker GRU, learned gates, explicit availability flags, rule-layer features; seeded ensemble | hourly grid + static + rules |
| ablations | `proposed_no_vitals`, `proposed_no_biomarkers`, `proposed_no_reasoning`, `proposed_concat` | as above, one change each |

All models see only observations with `availtime <= landmark`. Neural ensembles: `models.n_members` seeds, mean logit; calibration = Platt scaling fitted on validation patients only.

## Outputs

Per landmark: calibrated risk, the horizon, observed and missing modalities, activated rules, feature-group contributions, **model disagreement** (ensemble std of member probabilities) and a simulated decision (`NONE`, `ALERT`, `QUEUE_REVIEW`). Three different things are kept separate: *calibrated risk* is a probability under the training population; *disagreement* says the ensemble members differ on this input; *missing-input / rule flags* describe the input. Contributions come from group occlusion (risk with the group removed vs present): local, model-dependent, not causal.

## Intended use

Teaching, benchmarking, and as a starting point for studying leakage-safe landmark evaluation, calibration, and alert-burden accounting.

## Not intended / out of scope

Clinical decision-making; patient uploads (the research interface accepts only the synthetic fixture shape in public demos); treatment selection; claims about patient benefit; any claim about the published paper's numbers.

## Training data and splits

Synthetic fixture (demo, tests) and the MIMIC-IV demo (integration). Patients are assigned to train/validation/test **before** windows exist, stratified by whether a patient has an eligible event, so repeated admissions and overlapping windows from one patient stay in one split. Preprocessing statistics are fitted on training patients; thresholds, calibration, the shared cooldown and hyperparameters (logistic C, early stopping) use validation patients; test patients are reserved for the final evaluation.

## Evaluation protocol

Window metrics (accuracy, sensitivity, specificity, precision, AUROC, average precision, Brier, calibration bins) with patient-level cluster bootstrap where there are at least 10 positive patients. Alert metrics under one shared policy: threshold per model chosen on validation under the same false-alert budget; one shared cooldown; true alert = window label 1; lead time = event time - first true alert; missed events stay in the denominator; false alerts per monitored patient-day; alerts on unknown windows reported separately as unconfirmed. Latency is reported separately.

## Known limitations

- Tiny data: the MIMIC demo has about 20 independent event patients; synthetic data has by construction learnable structure. Neither supports a claim about patients.
- Selective testing shapes both labels and features.
- Platt calibration on few validation events is unstable; read the calibration plot's sparse bins as noise.
- The "cognitive reasoning" layer is four hand-written rules, not a cognitive model. R3/R4 change review routing only.
- The federated run is a single-machine simulation with no formal privacy guarantee; the routing run uses invented rewards on synthetic trajectories.
- Saved sklearn baselines and the cohort bundle are `joblib` pickles: load only files you created.
- If the proposed model underperforms a simpler baseline, the result tables say so; they are not tuned to prevent it.

## Provenance

Every run writes `run_manifest.json`: data version, cohort exclusions, split seed, config hash, package versions, hardware, commit SHA. Trained weights from the published runs are not distributed; `biocog` regenerates them from code and data in minutes on a CPU.
