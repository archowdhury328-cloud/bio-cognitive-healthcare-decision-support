"""Alert-routing simulator: reward mechanics, determinism, learning loop."""

from __future__ import annotations

import numpy as np

from biocog import routing
from biocog.routing import Episode


def _ep(event=10, n=11, hi_at=()):
    risk = np.full(n, 0.1)
    for t in hi_at:
        risk[t] = 0.95
    return Episode(risk, np.zeros(n, int), float(event))


def _policy(action):
    return lambda s: action if s[0] == routing.N_BUCKETS - 1 else 0


def test_queue_review_completing_before_event_earns_detection_minus_cost():
    r = routing.run_episode(_ep(hi_at=(3,)), _policy(1))  # completes at 7, event at 10
    assert r["return"] == routing.R_DETECT - routing.COST[1] and r["detected"] and r["lead_h"] == 3.0


def test_flag_completing_at_event_time_is_too_late_and_misses():
    r = routing.run_episode(_ep(hi_at=(9,)), _policy(2))  # completes at 10 == event
    assert not r["detected"] and r["return"] == -routing.COST[2] + routing.R_MISS


def test_no_review_on_event_episode_pays_miss_penalty_and_event_free_pays_nothing():
    assert routing.run_episode(_ep(), _policy(1))["return"] == routing.R_MISS
    free = Episode(np.full(20, 0.1), np.zeros(20, int), np.nan)
    assert routing.run_episode(free, _policy(1))["return"] == 0.0
    assert routing.run_episode(free, _policy(1))["detected"] is None


def test_actions_ignored_while_review_pending():
    calls = []
    ep = _ep(hi_at=(1, 2, 3))

    def choose(s):
        calls.append(s[2])
        return 1 if s[0] == routing.N_BUCKETS - 1 else 0

    r = routing.run_episode(ep, choose)
    assert r["reviews"] == 1  # t=1 queued; 2,3 forced waits (pending until t=5)
    assert 1 not in calls  # choose() is never consulted while pending


def test_simulation_is_deterministic_and_event_episodes_truncate_at_event():
    a, b = routing.simulate_patients(50, 3), routing.simulate_patients(50, 3)
    assert all(np.array_equal(x.risk, y.risk) for x, y in zip(a, b, strict=True))
    for ep in a:
        if not np.isnan(ep.event_t):
            assert len(ep.risk) == int(ep.event_t) + 1


def test_q_learning_updates_values_and_experiment_reports_both_policies():
    eps = routing.simulate_patients(200, 0)
    Q = routing.train_q(eps, 300, 0)
    assert Q.shape == (routing.N_BUCKETS, 2, 2, 3) and np.abs(Q).sum() > 0
    res = routing.run_experiment(200, 60, n_seeds=2, seed=0)
    assert res["synthetic"] and "no clinical claim" in res["note"]
    assert {"fixed_threshold", "learned_q_policy"} <= set(res)
    assert res["learned_q_policy"]["n_seeds"] == 2 and len(res["learned_q_policy"]["per_seed"]) == 2


def test_fixed_threshold_tuned_on_training_episodes_only():
    train = routing.simulate_patients(80, 1)
    theta, action, res = routing.best_fixed_threshold(train)
    assert 0 < theta < 1 and action in (1, 2) and res["events"] > 0
