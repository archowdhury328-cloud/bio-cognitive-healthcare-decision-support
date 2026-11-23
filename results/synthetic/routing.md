# Alert-routing simulation (SYNTHETIC trajectories)

Review prioritisation demonstration. Rewards and costs are invented release choices; nothing here supports a claim of patient benefit or treatment selection.

Actions: wait / queue_review (cost 1.0, completes in 4 h) / flag_earlier_review (cost 2.5, completes in 1 h). Reward +10.0 if a review completes 0 < lead <= 12 h before the event; -5.0 if the event is not preceded by a completed review. Feedback is delayed until the review completes.

Held-out synthetic patients: 400; training episodes per seed: 12000.

| policy | mean return | timely detection | reviews / patient-day | median lead after review h |
|---|---|---|---|---|
| fixed threshold >= 0.33 -> queue_review (tuned on training episodes) | -1.23 | 0.85 | 2.00 | 6.0 |
| tabular Q-learning (mean +/- sd over 5 seeds) | -1.22 +/- 0.17 | 0.27 +/- 0.14 | 0.53 +/- 0.27 | 6.0 |
