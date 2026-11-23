# Federated averaging simulation - Synthetic benchmark (400 generated patients)

**Simulation only.** Clients are patient-disjoint partitions of the training patients on one machine. Each client shares 6,497 model parameters (~0.026 MB) per round plus its training-window count (the averaging weight); raw windows are not exchanged in the simulation. FedAvg alone gives no formal privacy guarantee.

## Clients

| client | patients | train windows | positive windows |
|---|---|---|---|
| 0 | 53 | 1748 | 91 |
| 1 | 52 | 1931 | 96 |
| 2 | 52 | 1679 | 108 |
| 3 | 52 | 1622 | 91 |

## Same test patients: federated vs pooled

| training | auroc | average_precision | brier |
|---|---|---|---|
| FedAvg (4 clients, 6 rounds x 4 local epochs) | 0.879 | 0.293 | 0.037 |
| Pooled (same architecture, 24 epochs) | 0.786 | 0.153 | 0.040 |

Test windows: 2916 (128 positive). One training seed; differences between the rows are within what seed noise can produce on a cohort this small.

## Federated test performance by round

| round | test AUROC | test AP |
|---|---|---|
| 1 | 0.873 | 0.299 |
| 2 | 0.898 | 0.296 |
| 3 | 0.895 | 0.312 |
| 4 | 0.892 | 0.318 |
| 5 | 0.887 | 0.317 |
| 6 | 0.879 | 0.293 |
