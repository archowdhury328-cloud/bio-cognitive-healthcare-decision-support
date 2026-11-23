# Synthetic benchmark (400 generated patients)

Every number below was computed from saved predictions or run logs by `biocog report`.

## Cohort flow

| stage | split | n |
|---|---|---|
| all_stays | all | 445 |
| all_patients | all | 400 |
| excluded:historical_high | all | 61 |
| eligible_stays | all | 384 |
| eligible_patients | all | 350 |
| eligible_event_stays | all | 116 |
| eligible_event_patients | all | 111 |
| event_free_stays | all | 268 |
| windows_total | all | 14888 |
| windows_positive | all | 642 |
| windows_negative | all | 11925 |
| windows_unknown:incomplete_followup | all | 1590 |
| windows_unknown:no_glucose | all | 731 |
| eligible_stays | train | 229 |
| eligible_patients | train | 210 |
| eligible_event_stays | train | 72 |
| eligible_event_patients | train | 67 |
| event_free_stays | train | 157 |
| windows_total | train | 8312 |
| windows_positive | train | 386 |
| windows_negative | train | 6594 |
| windows_unknown:incomplete_followup | train | 932 |
| windows_unknown:no_glucose | train | 400 |
| eligible_stays | val | 79 |
| eligible_patients | val | 70 |
| eligible_event_stays | val | 22 |
| eligible_event_patients | val | 22 |
| event_free_stays | val | 57 |
| windows_total | val | 3165 |
| windows_positive | val | 128 |
| windows_negative | val | 2543 |
| windows_unknown:incomplete_followup | val | 338 |
| windows_unknown:no_glucose | val | 156 |
| eligible_stays | test | 76 |
| eligible_patients | test | 70 |
| eligible_event_stays | test | 22 |
| eligible_event_patients | test | 22 |
| event_free_stays | test | 54 |
| windows_total | test | 3411 |
| windows_positive | test | 128 |
| windows_negative | test | 2788 |
| windows_unknown:incomplete_followup | test | 320 |
| windows_unknown:no_glucose | test | 175 |

## Test-set window metrics (labelled windows)

Threshold = validation-selected alert threshold on calibrated risk. `n/a` = unestimable (too few positive or negative patients) - not zero.

| model | AUROC [95% CI] | AP [95% CI] | Brier [95% CI] | accuracy | sensitivity | specificity | precision | threshold | estimable |
|---|---|---|---|---|---|---|---|---|---|
| Last glucose value | 0.834 [0.767, 0.889] | 0.204 [0.122, 0.341] | 0.039 [0.026, 0.055] | 0.849 | 0.648 | 0.858 | 0.174 | 0.121 | True |
| Logistic regression | 0.878 [0.823, 0.927] | 0.240 [0.136, 0.447] | 0.039 [0.026, 0.054] | 0.880 | 0.562 | 0.895 | 0.197 | 0.139 | True |
| Gradient-boosted trees | 0.847 [0.777, 0.908] | 0.170 [0.098, 0.339] | 0.041 [0.027, 0.058] | 0.890 | 0.406 | 0.912 | 0.175 | 0.175 | True |
| GRU comparator | 0.903 [0.854, 0.943] | 0.326 [0.194, 0.497] | 0.036 [0.024, 0.050] | 0.905 | 0.570 | 0.920 | 0.247 | 0.208 | True |
| Proposed gated multimodal | 0.891 [0.836, 0.937] | 0.310 [0.169, 0.494] | 0.037 [0.025, 0.051] | 0.877 | 0.625 | 0.888 | 0.205 | 0.176 | True |

## Alert-level evaluation (test patients)

Shared policy: cooldown 6 h; validation false-alert budget 0.5 per patient-day.

| model | detected / eligible events | missed | median lead h [IQR] | false alerts / patient-day | unconfirmed alerts / patient-day | threshold | cooldown h | monitored patient-days |
|---|---|---|---|---|---|---|---|---|
| Last glucose value | 21 / 22 | 1 | 2.2 [1.0, 3.9] | 0.59 | 0.07 | 0.121 | 6.0 | 142.1 |
| Logistic regression | 21 / 22 | 1 | 2.2 [1.2, 3.3] | 0.60 | 0.11 | 0.139 | 6.0 | 142.1 |
| Gradient-boosted trees | 18 / 22 | 4 | 2.6 [1.1, 3.6] | 0.50 | 0.06 | 0.175 | 6.0 | 142.1 |
| GRU comparator | 20 / 22 | 2 | 2.6 [1.6, 3.2] | 0.53 | 0.08 | 0.208 | 6.0 | 142.1 |
| Proposed gated multimodal | 21 / 22 | 1 | 3.0 [1.7, 3.3] | 0.58 | 0.08 | 0.176 | 6.0 | 142.1 |

## Paired event detection (same events, which model caught them)

| comparison | both detected | only proposed | only comparator | neither |
|---|---|---|---|---|
| Proposed gated multimodal vs Last glucose value | 20 | 1 | 1 | 0 |
| Proposed gated multimodal vs Logistic regression | 20 | 1 | 1 | 0 |
| Proposed gated multimodal vs Gradient-boosted trees | 17 | 4 | 1 | 0 |
| Proposed gated multimodal vs GRU comparator | 20 | 1 | 0 | 1 |
| Proposed gated multimodal vs Proposed - no vitals | 20 | 1 | 1 | 0 |
| Proposed gated multimodal vs Proposed - no biomarkers | 14 | 7 | 0 | 1 |
| Proposed gated multimodal vs Proposed - no reasoning layer | 21 | 0 | 0 | 1 |
| Proposed gated multimodal vs Proposed - concat fusion | 21 | 0 | 0 | 1 |

## Ablations of the proposed model (same cohort, same budget)

| model | AUROC [95% CI] | AP [95% CI] | Brier [95% CI] | accuracy | sensitivity | specificity | precision | threshold | estimable |
|---|---|---|---|---|---|---|---|---|---|
| Proposed gated multimodal | 0.891 [0.836, 0.937] | 0.310 [0.169, 0.494] | 0.037 [0.025, 0.051] | 0.877 | 0.625 | 0.888 | 0.205 | 0.176 | True |
| Proposed - no vitals | 0.812 [0.742, 0.876] | 0.149 [0.093, 0.262] | 0.040 [0.027, 0.057] | 0.845 | 0.625 | 0.855 | 0.165 | 0.136 | True |
| Proposed - no biomarkers | 0.737 [0.633, 0.849] | 0.208 [0.112, 0.360] | 0.038 [0.025, 0.054] | 0.949 | 0.297 | 0.978 | 0.388 | 0.209 | True |
| Proposed - no reasoning layer | 0.872 [0.811, 0.926] | 0.204 [0.108, 0.403] | 0.042 [0.028, 0.057] | 0.874 | 0.539 | 0.890 | 0.183 | 0.200 | True |
| Proposed - concat fusion | 0.894 [0.842, 0.940] | 0.302 [0.168, 0.475] | 0.037 [0.025, 0.052] | 0.890 | 0.617 | 0.902 | 0.224 | 0.195 | True |


| model | detected / eligible events | missed | median lead h [IQR] | false alerts / patient-day | unconfirmed alerts / patient-day | threshold | cooldown h | monitored patient-days |
|---|---|---|---|---|---|---|---|---|
| Proposed gated multimodal | 21 / 22 | 1 | 3.0 [1.7, 3.3] | 0.58 | 0.08 | 0.176 | 6.0 | 142.1 |
| Proposed - no vitals | 21 / 22 | 1 | 2.0 [1.1, 3.9] | 0.65 | 0.08 | 0.136 | 6.0 | 142.1 |
| Proposed - no biomarkers | 14 / 22 | 8 | 3.1 [1.2, 4.1] | 0.20 | 0.03 | 0.209 | 6.0 | 142.1 |
| Proposed - no reasoning layer | 21 / 22 | 1 | 2.8 [1.7, 3.2] | 0.56 | 0.08 | 0.200 | 6.0 | 142.1 |
| Proposed - concat fusion | 21 / 22 | 1 | 3.0 [1.7, 3.3] | 0.55 | 0.08 | 0.195 | 6.0 | 142.1 |

## Review-routing impact (R3/R4 rules change review decisions only)

Alerts of the proposed model under the shared policy; routed alerts become QUEUE_REVIEW instead of ALERT. Risk itself is unchanged by these signals.

| routing signal | landmarks flagged | alerts total | alerts routed to review | of which true / false / unconfirmed | automatic alerts / patient-day | all alerts / patient-day |
|---|---|---|---|---|---|---|
| R3/R4 review flag | 12.3% | 116 | 1 | 0 / 1 / 0 | 0.81 | 0.82 |
| ensemble disagreement > cutoff | 0.3% | 116 | 0 | 0 / 0 / 0 | 0.82 | 0.82 |
| either | 12.5% | 116 | 1 | 0 / 1 / 0 | 0.81 | 0.82 |

## Training budget and inference latency

Latency = median single-landmark `predict_raw` call on this machine; excludes feature extraction.

| model | fit seconds | train windows | members x epochs run | median inference ms / landmark |
|---|---|---|---|---|
| Last glucose value | 0.5 | 6980 | - | 0.03 |
| Logistic regression | 1.4 | 6980 | - | 1.40 |
| Gradient-boosted trees | 3.4 | 6980 | - | 9.06 |
| GRU comparator | 49.4 | 6980 | 3 x [10, 8, 9] (max 25) | 14.30 |
| Proposed gated multimodal | 90.1 | 6980 | 3 x [11, 10, 10] (max 25) | 28.95 |
| Proposed - no vitals | 81.4 | 6980 | 3 x [10, 9, 8] (max 25) | 27.30 |
| Proposed - no biomarkers | 102.6 | 6980 | 3 x [14, 10, 11] (max 25) | 28.68 |
| Proposed - no reasoning layer | 99.5 | 6980 | 3 x [11, 10, 10] (max 25) | 27.59 |
| Proposed - concat fusion | 98.7 | 6980 | 3 x [11, 9, 10] (max 25) | 28.72 |
