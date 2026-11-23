# Federated averaging simulation - MIMIC-IV Clinical Database Demo v2.2: glucose-excursion engineering example

**Simulation only.** Clients are patient-disjoint partitions of the training patients on one machine. Each client shares 6,497 model parameters (~0.026 MB) per round plus its training-window count (the averaging weight); raw windows are not exchanged in the simulation. FedAvg alone gives no formal privacy guarantee.

## Clients

| client | patients | train windows | positive windows |
|---|---|---|---|
| 0 | 11 | 286 | 18 |
| 1 | 11 | 567 | 15 |
| 2 | 11 | 228 | 15 |
| 3 | 11 | 427 | 13 |

## Same test patients: federated vs pooled

| training | auroc | average_precision | brier |
|---|---|---|---|
| FedAvg (4 clients, 6 rounds x 4 local epochs) | 0.634 | 0.122 | 0.055 |
| Pooled (same architecture, 24 epochs) | 0.460 | 0.052 | 0.056 |

Test windows: 407 (24 positive). One training seed; differences between the rows are within what seed noise can produce on a cohort this small.

## Federated test performance by round

| round | test AUROC | test AP |
|---|---|---|
| 1 | 0.410 | 0.080 |
| 2 | 0.396 | 0.078 |
| 3 | 0.391 | 0.073 |
| 4 | 0.619 | 0.168 |
| 5 | 0.627 | 0.140 |
| 6 | 0.634 | 0.122 |
