"""Offline alert-routing simulator and tabular Q-learning policy. SYNTHETIC trajectories only.

This demonstrates REVIEW PRIORITISATION, not treatment selection. Nothing here supports a claim of
patient benefit: reward and cost values are invented release choices, and the "patients" are drawn
from a generator, not from any real cohort.

Environment (hourly steps)
  observable state : (risk bucket 0..5, unreliable-input flag u, review_pending)
  actions          : 0 wait | 1 queue_review (cost 1.0, completes after 4 h) |
                     2 flag_earlier_review (cost 2.5, completes after 1 h)
  delayed feedback : a review's outcome arrives only when it COMPLETES. If the patient's event time e
                     satisfies 0 < e - completion <= H, reward +10 (timely detection); otherwise 0.
  miss penalty     : -5 at the event if no review completed in the H hours before it.
While a review is pending, new actions are ignored (forced wait). Episodes end at the event or at the
end of follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTIONS = ("wait", "queue_review", "flag_earlier_review")
COST = (0.0, 1.0, 2.5)
DELAY = (0, 4, 1)
H = 12.0
R_DETECT, R_MISS = 10.0, -5.0
N_BUCKETS = 6


@dataclass(frozen=True)
class Episode:
    risk: np.ndarray        # observed (noisy) risk score per hour
    unreliable: np.ndarray  # 0/1 per hour: input flagged missing/stale/low-agreement
    event_t: float          # np.nan if event-free


def simulate_patients(n: int, seed: int) -> list[Episode]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        length = int(rng.integers(24, 73))
        has_event = rng.random() < 0.3
        e = float(rng.integers(10, length)) if has_event else np.nan
        t = np.arange(length, dtype=float)
        u = (rng.random(length) < 0.2).astype(int)
        signal = np.zeros(length)
        if has_event:
            signal = 0.6 * np.clip(1 - (e - t) / 8.0, 0, 1)
        spurious = (rng.random(length) < 0.03) * 0.4
        spurious = np.maximum.reduce([np.roll(spurious, k) for k in range(3)])
        noise = rng.normal(0, 0.12, length) * (1 + 1.2 * u)
        risk = np.clip(0.1 + signal + spurious + noise, 0, 1)
        if has_event:
            risk, u = risk[: int(e) + 1], u[: int(e) + 1]
        out.append(Episode(risk, u, e))
    return out


def bucket(r: float) -> int:
    return int(min(r * N_BUCKETS, N_BUCKETS - 1))


def run_episode(ep: Episode, choose, learn=None):
    """Play one episode. ``choose(state)->action``; ``learn(s,a,r,s2,done)`` optional.

    Reward for a transition includes any review that completes at the boundary to the next hour, so a
    completion reward is credited to the transition in which it is earned (the forced-wait chain after a
    queue action carries it back to the action that started the review).
    """
    pending, completes_at = False, np.inf
    n_reviews, detected, lead, total = 0, False, np.nan, 0.0
    has_event, e, T = not np.isnan(ep.event_t), ep.event_t, len(ep.risk)
    for t in range(T):
        s = (bucket(ep.risk[t]), int(ep.unreliable[t]), int(pending))
        a = 0 if pending else choose(s)
        r = 0.0
        if a:
            r -= COST[a]
            pending, completes_at = True, t + DELAY[a]
            n_reviews += 1
        if pending and completes_at <= t + 1:
            pending = False
            if has_event and not detected and 0 < e - completes_at <= H:
                r += R_DETECT
                detected, lead = True, float(e - completes_at)
        done = t == T - 1
        if done and has_event and not detected:
            r += R_MISS
        total += r
        if learn is not None:
            s2 = s if done else (bucket(ep.risk[t + 1]), int(ep.unreliable[t + 1]), int(pending))
            learn(s, a, r, s2, done)
    return {"return": total, "reviews": n_reviews, "hours": T, "has_event": has_event,
            "detected": detected if has_event else None, "lead_h": lead}


def train_q(episodes: list[Episode], n_episodes: int, seed: int, alpha=0.05, gamma=0.97) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_BUCKETS, 2, 2, len(ACTIONS)))
    for k in range(n_episodes):
        eps = max(0.05, 1.0 - k / (0.6 * n_episodes))
        ep = episodes[int(rng.integers(len(episodes)))]

        def choose(s, eps=eps):
            return int(rng.integers(len(ACTIONS))) if rng.random() < eps else int(np.argmax(Q[s]))

        def learn(s, a, r, s2, done):
            target = r + (0.0 if done else gamma * Q[s2].max())
            Q[s][a] += alpha * (target - Q[s][a])

        run_episode(ep, choose, learn)
    return Q


def summarise(results: list[dict]) -> dict:
    ev = [r for r in results if r["has_event"]]
    hours = sum(r["hours"] for r in results)
    leads = [r["lead_h"] for r in ev if r["detected"]]
    return {"mean_return": float(np.mean([r["return"] for r in results])),
            "events": len(ev), "timely_detected": int(sum(bool(r["detected"]) for r in ev)),
            "timely_detection_rate": float(np.mean([bool(r["detected"]) for r in ev])) if ev else float("nan"),
            "median_lead_after_review_h": float(np.median(leads)) if leads else float("nan"),
            "reviews_per_patient_day": float(sum(r["reviews"] for r in results) / (hours / 24.0))}


def fixed_threshold_policy(theta: float, action: int):
    return lambda s: action if (s[0] / N_BUCKETS >= theta) else 0


def evaluate_policy(episodes: list[Episode], choose) -> dict:
    return summarise([run_episode(ep, choose) for ep in episodes])


def best_fixed_threshold(train_eps: list[Episode]) -> tuple[float, int, dict]:
    """Tune the fixed-threshold baseline on TRAINING episodes (same data the learner sees)."""
    best = None
    for action in (1, 2):
        for theta in np.arange(1, N_BUCKETS) / N_BUCKETS:
            res = evaluate_policy(train_eps, fixed_threshold_policy(float(theta), action))
            if best is None or res["mean_return"] > best[2]["mean_return"]:
                best = (float(theta), action, res)
    assert best is not None
    return best


def run_experiment(n_train_episodes: int, n_eval_patients: int, n_seeds: int = 5, seed: int = 0) -> dict:
    train_eps = simulate_patients(1000, seed)
    eval_eps = simulate_patients(n_eval_patients, seed + 10_000)  # held-out synthetic patients
    theta, action, fixed_train = best_fixed_threshold(train_eps)
    fixed = evaluate_policy(eval_eps, fixed_threshold_policy(theta, action))
    learned = []
    for s in range(n_seeds):
        Q = train_q(train_eps, n_train_episodes, seed + s)
        learned.append(evaluate_policy(eval_eps, lambda st, Q=Q: int(np.argmax(Q[st]))))
    keys = ["mean_return", "timely_detection_rate", "reviews_per_patient_day", "median_lead_after_review_h"]
    agg = {k: {"mean": float(np.nanmean([r[k] for r in learned])), "sd": float(np.nanstd([r[k] for r in learned]))}
           for k in keys}
    return {"synthetic": True, "note": "review-prioritisation demonstration; invented rewards; no clinical claim",
            "actions": list(ACTIONS), "reward": {"detect": R_DETECT, "miss": R_MISS, "cost": dict(zip(ACTIONS, COST, strict=True)),
                                                 "review_delay_h": dict(zip(ACTIONS, DELAY, strict=True)), "horizon_h": H},
            "n_train_episodes": n_train_episodes, "n_eval_patients": n_eval_patients,
            "fixed_threshold": {"threshold": theta, "action": ACTIONS[action], "train": fixed_train, "heldout": fixed},
            "learned_q_policy": {"per_seed": learned, "aggregate": agg, "n_seeds": n_seeds}}
