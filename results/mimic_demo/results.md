# MIMIC-IV Clinical Database Demo v2.2: glucose-excursion engineering example

Every number below was computed from saved predictions or run logs by `biocog report`.

## Cohort flow

| stage | split | n |
|---|---|---|
| all_stays | all | 140 |
| all_patients | all | 100 |
| excluded:historical_high | all | 39 |
| excluded:followup_before_first_landmark | all | 3 |
| eligible_stays | all | 98 |
| eligible_patients | all | 80 |
| eligible_event_stays | all | 20 |
| eligible_event_patients | all | 20 |
| event_free_stays | all | 78 |
| windows_total | all | 6307 |
| windows_positive | all | 109 |
| windows_negative | all | 2105 |
| windows_unknown:incomplete_followup | all | 466 |
| windows_unknown:no_glucose | all | 3627 |
| eligible_stays | train | 60 |
| eligible_patients | train | 48 |
| eligible_event_stays | train | 12 |
| eligible_event_patients | train | 12 |
| event_free_stays | train | 48 |
| windows_total | train | 4147 |
| windows_positive | train | 61 |
| windows_negative | train | 1447 |
| windows_unknown:incomplete_followup | train | 286 |
| windows_unknown:no_glucose | train | 2353 |
| eligible_stays | val | 19 |
| eligible_patients | val | 16 |
| eligible_event_stays | val | 4 |
| eligible_event_patients | val | 4 |
| event_free_stays | val | 15 |
| windows_total | val | 862 |
| windows_positive | val | 24 |
| windows_negative | val | 275 |
| windows_unknown:incomplete_followup | val | 90 |
| windows_unknown:no_glucose | val | 473 |
| eligible_stays | test | 19 |
| eligible_patients | test | 16 |
| eligible_event_stays | test | 4 |
| eligible_event_patients | test | 4 |
| event_free_stays | test | 15 |
| windows_total | test | 1298 |
| windows_positive | test | 24 |
| windows_negative | test | 383 |
| windows_unknown:incomplete_followup | test | 90 |
| windows_unknown:no_glucose | test | 801 |

## Test-set window metrics (labelled windows)

Threshold = validation-selected alert threshold on calibrated risk. `n/a` = unestimable (too few positive or negative patients) - not zero.

| model | AUROC [95% CI] | AP [95% CI] | Brier [95% CI] | accuracy | sensitivity | specificity | precision | threshold | estimable |
|---|---|---|---|---|---|---|---|---|---|
| Last glucose value | 0.919 | 0.427 | 0.049 | 0.862 | 0.500 | 0.885 | 0.214 | 0.158 | True |
| Logistic regression | 0.783 | 0.121 | 0.056 | 0.730 | 0.917 | 0.718 | 0.169 | 0.098 | True |
| Gradient-boosted trees | 0.383 | 0.050 | 0.057 | 0.415 | 0.500 | 0.410 | 0.050 | 0.089 | True |
| GRU comparator | 0.538 | 0.089 | 0.056 | 0.941 | 0.000 | 1.000 | n/a | 0.168 | True |
| Proposed gated multimodal | 0.399 | 0.048 | 0.056 | 0.941 | 0.000 | 1.000 | n/a | 0.101 | True |

## Alert-level evaluation (test patients)

Shared policy: cooldown 6 h; validation false-alert budget 0.5 per patient-day.

| model | detected / eligible events | missed | median lead h [IQR] | false alerts / patient-day | unconfirmed alerts / patient-day | threshold | cooldown h | monitored patient-days |
|---|---|---|---|---|---|---|---|---|
| Last glucose value | 2 / 4 | 2 | 2.5 [2.3, 2.7] | 0.15 | 0.35 | 0.158 | 6.0 | 54.1 |
| Logistic regression | 4 / 4 | 0 | 5.3 [4.8, 5.6] | 0.57 | 1.50 | 0.098 | 6.0 | 54.1 |
| Gradient-boosted trees | 2 / 4 | 2 | 3.3 [2.7, 4.0] | 0.85 | 1.57 | 0.089 | 6.0 | 54.1 |
| GRU comparator | 0 / 4 | 4 | n/a [n/a, n/a] | 0.00 | 0.00 | 0.168 | 6.0 | 54.1 |
| Proposed gated multimodal | 0 / 4 | 4 | n/a [n/a, n/a] | 0.00 | 0.17 | 0.101 | 6.0 | 54.1 |

## Paired event detection (same events, which model caught them)

| comparison | both detected | only proposed | only comparator | neither |
|---|---|---|---|---|
| Proposed gated multimodal vs Last glucose value | 0 | 0 | 2 | 2 |
| Proposed gated multimodal vs Logistic regression | 0 | 0 | 4 | 0 |
| Proposed gated multimodal vs Gradient-boosted trees | 0 | 0 | 2 | 2 |
| Proposed gated multimodal vs GRU comparator | 0 | 0 | 0 | 4 |
| Proposed gated multimodal vs Proposed - no vitals | 0 | 0 | 0 | 4 |
| Proposed gated multimodal vs Proposed - no biomarkers | 0 | 0 | 1 | 3 |
| Proposed gated multimodal vs Proposed - no reasoning layer | 0 | 0 | 1 | 3 |
| Proposed gated multimodal vs Proposed - concat fusion | 0 | 0 | 0 | 4 |

## Ablations of the proposed model (same cohort, same budget)

| model | AUROC [95% CI] | AP [95% CI] | Brier [95% CI] | accuracy | sensitivity | specificity | precision | threshold | estimable |
|---|---|---|---|---|---|---|---|---|---|
| Proposed gated multimodal | 0.399 | 0.048 | 0.056 | 0.941 | 0.000 | 1.000 | n/a | 0.101 | True |
| Proposed - no vitals | 0.552 | 0.063 | 0.058 | 0.919 | 0.000 | 0.977 | 0.000 | 0.166 | True |
| Proposed - no biomarkers | 0.540 | 0.090 | 0.055 | 0.882 | 0.125 | 0.930 | 0.100 | 0.105 | True |
| Proposed - no reasoning layer | 0.554 | 0.143 | 0.055 | 0.909 | 0.167 | 0.956 | 0.190 | 0.110 | True |
| Proposed - concat fusion | 0.451 | 0.053 | 0.056 | 0.941 | 0.000 | 1.000 | n/a | 0.092 | True |


| model | detected / eligible events | missed | median lead h [IQR] | false alerts / patient-day | unconfirmed alerts / patient-day | threshold | cooldown h | monitored patient-days |
|---|---|---|---|---|---|---|---|---|
| Proposed gated multimodal | 0 / 4 | 4 | n/a [n/a, n/a] | 0.00 | 0.17 | 0.101 | 6.0 | 54.1 |
| Proposed - no vitals | 0 / 4 | 4 | n/a [n/a, n/a] | 0.04 | 0.31 | 0.166 | 6.0 | 54.1 |
| Proposed - no biomarkers | 1 / 4 | 3 | 4.8 [4.8, 4.8] | 0.15 | 0.28 | 0.105 | 6.0 | 54.1 |
| Proposed - no reasoning layer | 1 / 4 | 3 | 5.8 [5.8, 5.8] | 0.09 | 0.28 | 0.110 | 6.0 | 54.1 |
| Proposed - concat fusion | 0 / 4 | 4 | n/a [n/a, n/a] | 0.00 | 0.06 | 0.092 | 6.0 | 54.1 |

## Review-routing impact (R3/R4 rules change review decisions only)

Alerts of the proposed model under the shared policy; routed alerts become QUEUE_REVIEW instead of ALERT. Risk itself is unchanged by these signals.

| routing signal | landmarks flagged | alerts total | alerts routed to review | of which true / false / unconfirmed | automatic alerts / patient-day | all alerts / patient-day |
|---|---|---|---|---|---|---|
| R3/R4 review flag | 6.0% | 9 | 0 | 0 / 0 / 0 | 0.17 | 0.17 |
| ensemble disagreement > cutoff | 0.0% | 9 | 0 | 0 / 0 / 0 | 0.17 | 0.17 |
| either | 6.0% | 9 | 0 | 0 / 0 / 0 | 0.17 | 0.17 |

## Training budget and inference latency

Latency = median single-landmark `predict_raw` call on this machine; excludes feature extraction.

| model | fit seconds | train windows | members x epochs run | median inference ms / landmark |
|---|---|---|---|---|
| Last glucose value | 0.2 | 1508 | - | 0.03 |
| Logistic regression | 0.5 | 1508 | - | 1.60 |
| Gradient-boosted trees | 2.8 | 1508 | - | 7.50 |
| GRU comparator | 15.4 | 1508 | 3 x [9, 7, 7] (max 25) | 8.95 |
| Proposed gated multimodal | 8.4 | 1508 | 3 x [8, 7, 7] (max 25) | 10.84 |
| Proposed - no vitals | 9.9 | 1508 | 3 x [14, 7, 7] (max 25) | 10.64 |
| Proposed - no biomarkers | 20.1 | 1508 | 3 x [10, 7, 7] (max 25) | 31.16 |
| Proposed - no reasoning layer | 18.8 | 1508 | 3 x [10, 7, 7] (max 25) | 30.69 |
| Proposed - concat fusion | 18.3 | 1508 | 3 x [7, 7, 7] (max 25) | 28.22 |
