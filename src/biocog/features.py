"""Leak-safe feature construction at landmark times.

The only rule that matters: a feature at landmark ``t`` may use an observation only if its
``t_avail <= t`` (the time the result became known), never ``t_chart`` alone. A lab drawn
before ``t`` but released after ``t`` is therefore invisible. Nothing after ``t`` is read.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .schema import BIO_IDX, N_VARS, STATIC, VARS, VITAL_IDX

N_CH = 3  # value, mask, gap


@dataclass(frozen=True)
class Features:
    index: pd.DataFrame
    static: np.ndarray          # [n, 3] age, sex_male, emergency
    seq: np.ndarray             # [n, T, N, 3] chronological hourly grid: value, mask, gap_h
    tab: np.ndarray             # [n, F] past-only summaries, NaN where unavailable
    tab_names: tuple[str, ...]
    age_h: np.ndarray           # [n, N] hours since last available observation, NaN if none
    n_obs: np.ndarray           # [n, N] observations available inside the lookback
    n_trend: np.ndarray         # [n, N] observations inside the trend window
    slope: np.ndarray           # [n, N] naive per-hour slope (0.0 when fewer than 2 points)
    last_val: np.ndarray        # [n, N]
    seq_r: np.ndarray | None = None      # sequence after the rule layer
    rule_vec: np.ndarray | None = None   # [n, R] rule-layer features
    flags: pd.DataFrame | None = field(default=None)

    def __len__(self) -> int:
        return len(self.index)

    def take(self, sel: np.ndarray) -> Features:
        sel = np.asarray(sel)
        def pick(a):
            return None if a is None else a[sel]
        return replace(
            self, index=self.index.iloc[sel].reset_index(drop=True), static=self.static[sel],
            seq=self.seq[sel], tab=self.tab[sel], age_h=self.age_h[sel], n_obs=self.n_obs[sel],
            n_trend=self.n_trend[sel], slope=self.slope[sel], last_val=self.last_val[sel],
            seq_r=pick(self.seq_r), rule_vec=pick(self.rule_vec),
            flags=None if self.flags is None else self.flags.iloc[sel].reset_index(drop=True))

    def split(self, name: str) -> Features:
        return self.take(np.where(self.index.split.to_numpy() == name)[0])

    def labelled(self) -> Features:
        return self.take(np.where(self.index.label.to_numpy() >= 0)[0])


def _var_summary(chart: np.ndarray, avail: np.ndarray, val: np.ndarray, t: float, T: int,
                 trend_w: float, cap: float):
    m = avail <= t  # availability, not draw time, decides visibility
    c, v = chart[m], val[m]
    if len(c) == 0:
        empty = np.zeros((T, N_CH))
        empty[:, 2] = cap
        return empty, np.nan, np.nan, 0, 0, 0.0, np.nan
    edges = t - np.arange(T + 1, dtype=float)
    pos = np.searchsorted(c, edges, side="right")
    counts = pos[:-1] - pos[1:]
    cs = np.concatenate([[0.0], np.cumsum(v)])
    sums = cs[pos[:-1]] - cs[pos[1:]]
    vals = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    idx = pos[:-1] - 1
    gaps = np.where(idx >= 0, edges[:-1] - c[np.maximum(idx, 0)], cap)
    grid = np.stack([vals, (counts > 0).astype(float), np.clip(gaps, 0, cap)], axis=-1)[::-1]
    if len(c) == 0:
        return grid, np.nan, np.nan, 0, 0, 0.0, np.nan
    tw = c > t - trend_w
    ntr = int(tw.sum())
    mean12 = float(v[tw].mean()) if ntr else np.nan
    slope = 0.0
    if ntr >= 2:
        x, y = c[tw] - c[tw].mean(), v[tw]
        denom = float((x * x).sum())
        slope = float((x * (y - y.mean())).sum() / denom) if denom > 0 else 0.0
    return grid, float(v[-1]), float(t - c[-1]), int((c > t - T).sum()), ntr, slope, mean12


def build_features(st: pd.DataFrame, ev: pd.DataFrame, windows: pd.DataFrame, cfg: dict[str, Any]) -> Features:
    """Build all arrays for every landmark in ``windows`` (one row per landmark)."""
    T = int(cfg["features"]["lookback_h"])
    trend_w, cap = float(cfg["features"]["trend_window_h"]), float(cfg["features"]["gap_cap_h"])
    n = len(windows)
    seq = np.zeros((n, T, N_VARS, N_CH))
    seq[..., 2] = cap
    age_h = np.full((n, N_VARS), np.nan)
    n_obs = np.zeros((n, N_VARS))
    n_trend = np.zeros((n, N_VARS))
    slope = np.zeros((n, N_VARS))
    last_val = np.full((n, N_VARS), np.nan)
    mean12 = np.full((n, N_VARS), np.nan)
    ev_by = {sid: {v: (g.t_chart.to_numpy(), g.t_avail.to_numpy(), g.value.to_numpy())
                   for v, g in grp.groupby("variable")} for sid, grp in ev.groupby("stay_id")}
    st_i = st.set_index("stay_id")
    win = windows.reset_index(drop=True)
    for sid, rows in win.groupby("stay_id").indices.items():
        per_var = ev_by.get(sid, {})
        for vi, name in enumerate(VARS):
            if name not in per_var:
                continue
            chart, avail, val = per_var[name]
            for r in rows:
                g, lv, ag, no, ntr, sl, m12 = _var_summary(chart, avail, val, float(win.t_h[r]), T, trend_w, cap)
                seq[r, :, vi, :] = g
                last_val[r, vi], age_h[r, vi], n_obs[r, vi] = lv, ag, no
                n_trend[r, vi], slope[r, vi], mean12[r, vi] = ntr, sl, m12
    srow = st_i.loc[win.stay_id]
    static = np.stack([srow.age.to_numpy(float), srow.sex_male.to_numpy(float),
                       srow.emergency.to_numpy(float)], axis=1)
    tab_cols, tab_names = [], []
    for vi, name in enumerate(VARS):
        age_fill = np.where(np.isnan(age_h[:, vi]), cap, np.minimum(age_h[:, vi], cap))
        for suffix, col in (("last", last_val[:, vi]), ("mean12", mean12[:, vi]), ("slope", slope[:, vi]),
                            ("n", n_obs[:, vi]), ("age", age_fill)):
            tab_cols.append(col)
            tab_names.append(f"{name}_{suffix}")
    tab = np.concatenate([np.stack(tab_cols, axis=1), static], axis=1)
    return Features(index=win, static=static, seq=seq, tab=tab, tab_names=tuple(tab_names) + STATIC,
                    age_h=age_h, n_obs=n_obs, n_trend=n_trend, slope=slope, last_val=last_val)


class Preprocessor:
    """Normalisation statistics fitted on training patients only."""

    def __init__(self, cfg: dict[str, Any]):
        self.cap = float(cfg["features"]["gap_cap_h"])
        self.stats: dict[str, np.ndarray] = {}

    def fit(self, f: Features) -> Preprocessor:
        mask = f.seq[..., 1] > 0
        vals = f.seq[..., 0]
        mu, sd = np.zeros(N_VARS), np.ones(N_VARS)
        for i in range(N_VARS):
            x = vals[..., i][mask[..., i]]
            if len(x) > 1:
                mu[i], sd[i] = x.mean(), max(x.std(), 1e-6)
        s_mu, s_sd = f.static.mean(0), np.maximum(f.static.std(0), 1e-6)
        r_src = f.rule_vec if f.rule_vec is not None else np.zeros((len(f), 1))
        self.stats = {"mu": mu, "sd": sd, "s_mu": s_mu, "s_sd": s_sd,
                      "r_mu": r_src.mean(0), "r_sd": np.maximum(r_src.std(0), 1e-6)}
        return self

    def tensors(self, f: Features, rules_on: bool = True, drop: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
        """Model-ready float32 arrays. ``drop`` removes whole modalities ('vitals', 'biomarkers')."""
        s = self.stats
        seq = f.seq_r if (rules_on and f.seq_r is not None) else f.seq
        val = (seq[..., 0] - s["mu"]) / s["sd"] * seq[..., 1]
        x = np.stack([val, seq[..., 1], seq[..., 2] / self.cap], axis=-1)
        rule = np.zeros((len(f), len(s["r_mu"])))
        if rules_on and f.rule_vec is not None:
            rule = (f.rule_vec - s["r_mu"]) / s["r_sd"]
        for name, idx in (("vitals", VITAL_IDX), ("biomarkers", BIO_IDX)):
            if name in drop:
                x[:, :, list(idx), :] = 0.0
                if name == "biomarkers":
                    rule = np.zeros_like(rule)
        avail = np.stack([(x[:, :, list(VITAL_IDX), 1].sum((1, 2)) > 0),
                          (x[:, :, list(BIO_IDX), 1].sum((1, 2)) > 0)], axis=1).astype(np.float32)
        static = (f.static - s["s_mu"]) / s["s_sd"]
        return {"seq": x.astype(np.float32), "static": static.astype(np.float32),
                "rule": rule.astype(np.float32), "avail": avail}
