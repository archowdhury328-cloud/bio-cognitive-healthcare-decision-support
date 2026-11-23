# Offline update log (promotion gate on validation patients only)

Serving version: **v0001**

| version | status | parent | val AP | val Brier | reason |
|---|---|---|---|---|---|
| v0001 | promoted | - | 0.253 | 0.073 | initial incumbent |
| v0002 | rejected | v0001 | 0.233 | 0.074 | AP 0.2525->0.2329 (need gain >= 0.0); Brier 0.0735->0.0736 (tolerance +0.005) |
