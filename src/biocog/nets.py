"""Neural models: GRU comparator and the proposed gated multimodal network, plus ensembles.

Proposed model (``ProposedNet``)
  static EHR context (+ rule-layer features)  -> MLP            -> h_s
  vital-sign history  (value, mask, gap)      -> masked GRU     -> h_v
  biomarker history   (value, mask, gap)      -> masked GRU     -> h_b
  explicit modality availability a_v, a_b (static is always available)
  fusion 'gated':  g = sigmoid(W [h_s, h_v, h_b, a_v, a_b]) split in 3;  z = sum_m g_m * (a_m * h_m)
  fusion 'concat': z = relu(W [h_s, a_v*h_v, a_b*h_b])           (ablation)
  head: z -> logit of event risk within the horizon.
An absent modality contributes exactly zero to the fused vector; the model cannot read the zeros
as data because the availability flag is an input to the gates.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .schema import BIO_IDX, VITAL_IDX

Tensors = dict[str, np.ndarray]


class MaskedGRU(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x [B,T,in]
        _, h = self.gru(x)
        return h[-1]


class GRUComparator(nn.Module):
    """Temporal GRU over all variables with masks and time-gap features; static joins at the head."""

    def __init__(self, n_vars: int, n_static: int, hidden: int, n_rule: int = 0):
        super().__init__()
        self.gru = MaskedGRU(n_vars * 3, hidden)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, seq, static, rule, avail):
        b, t = seq.shape[:2]
        return self.head(torch.cat([self.gru(seq.reshape(b, t, -1)), static], 1)).squeeze(-1)


class ProposedNet(nn.Module):
    def __init__(self, n_static: int, n_rule: int, hidden: int, fusion: str = "gated"):
        super().__init__()
        self.fusion, self.d = fusion, hidden
        self.static_enc = nn.Sequential(nn.Linear(n_static + n_rule, hidden), nn.ReLU())
        self.vit = MaskedGRU(len(VITAL_IDX) * 3, hidden)
        self.bio = MaskedGRU(len(BIO_IDX) * 3, hidden)
        if fusion == "gated":
            self.gate = nn.Linear(3 * hidden + 2, 3 * hidden)
        else:
            self.mix = nn.Linear(3 * hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, seq, static, rule, avail):
        b, t = seq.shape[:2]
        h_s = self.static_enc(torch.cat([static, rule], 1))
        h_v = self.vit(seq[:, :, list(VITAL_IDX), :].reshape(b, t, -1)) * avail[:, :1]
        h_b = self.bio(seq[:, :, list(BIO_IDX), :].reshape(b, t, -1)) * avail[:, 1:2]
        if self.fusion == "gated":
            g = torch.sigmoid(self.gate(torch.cat([h_s, h_v, h_b, avail], 1))).view(b, 3, self.d)
            z = g[:, 0] * h_s + g[:, 1] * h_v + g[:, 2] * h_b
        else:
            z = torch.relu(self.mix(torch.cat([h_s, h_v, h_b], 1)))
        return self.head(z).squeeze(-1)


def build_net(kind: str, n_vars: int, n_static: int, n_rule: int, hidden: int) -> nn.Module:
    if kind == "gru":
        return GRUComparator(n_vars, n_static, hidden, n_rule)
    if kind in ("proposed", "proposed_concat"):
        return ProposedNet(n_static, n_rule, hidden, "concat" if kind == "proposed_concat" else "gated")
    raise ValueError(f"unknown net kind {kind!r}")


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def _forward(net: nn.Module, d: Tensors, idx: np.ndarray) -> torch.Tensor:
    return net(_t(d["seq"][idx]), _t(d["static"][idx]), _t(d["rule"][idx]), _t(d["avail"][idx]))


def train_member(kind: str, d_tr: Tensors, y_tr: np.ndarray, d_va: Tensors | None, y_va: np.ndarray | None,
                 cfg: dict[str, Any], seed: int, epochs: int | None = None, lr: float | None = None,
                 init_state: dict | None = None) -> tuple[dict, dict]:
    """Train one net. Early stopping watches validation loss when ``d_va`` is given.

    Returns ``(state_dict, log)``. The log records the training budget actually used.
    """
    mc = cfg["models"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_vars, n_static, n_rule = d_tr["seq"].shape[2], d_tr["static"].shape[1], d_tr["rule"].shape[1]
    net = build_net(kind, n_vars, n_static, n_rule, int(mc["hidden"]))
    if init_state is not None:
        net.load_state_dict(init_state)
    opt = torch.optim.Adam(net.parameters(), lr=lr or float(mc["lr"]), weight_decay=float(mc["weight_decay"]))
    pos = max(float(y_tr.sum()), 1.0)
    pos_w = torch.tensor(max((len(y_tr) - pos) / pos, 1.0))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best, best_state, bad, log = np.inf, copy.deepcopy(net.state_dict()), 0, []
    n_epochs, bs = int(mc["epochs"] if epochs is None else epochs), int(mc["batch_size"])
    y_t = _t(y_tr)
    for ep in range(n_epochs):
        net.train()
        order = rng.permutation(len(y_tr))
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(_forward(net, d_tr, idx), y_t[idx])
            loss.backward()
            opt.step()
        if d_va is not None and y_va is not None and len(y_va):
            net.eval()
            with torch.no_grad():
                v = float(loss_fn(_forward(net, d_va, np.arange(len(y_va))), _t(y_va)))
            log.append({"epoch": ep + 1, "val_loss": v})
            if v < best - 1e-5:
                best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= int(mc["patience"]):
                    break
        else:
            best_state = copy.deepcopy(net.state_dict())
    return best_state, {"epochs_run": len(log) if log else n_epochs, "seed": seed, "history": log}


def predict_logits(kind: str, state: dict, d: Tensors, cfg: dict[str, Any]) -> np.ndarray:
    n_vars, n_static, n_rule = d["seq"].shape[2], d["static"].shape[1], d["rule"].shape[1]
    net = build_net(kind, n_vars, n_static, n_rule, int(cfg["models"]["hidden"]))
    net.load_state_dict(state)
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(d["seq"]), 512):
            out.append(_forward(net, d, np.arange(i, min(i + 512, len(d["seq"])))).numpy())
    return np.concatenate(out) if out else np.zeros(0)


class NetEnsemble:
    """Seeded ensemble. ``raw`` = mean member logit; ``disagreement`` = std of member probabilities."""

    def __init__(self, kind: str, cfg: dict[str, Any], rules_on: bool = True, drop: tuple[str, ...] = (),
                 name: str | None = None):
        self.kind, self.cfg, self.rules_on, self.drop = kind, cfg, rules_on, tuple(drop)
        self.name = name or kind
        self.states: list[dict] = []
        self.logs: list[dict] = []
        self.pre = None

    def tensors(self, f) -> Tensors:
        return self.pre.tensors(f, rules_on=self.rules_on, drop=self.drop)

    def fit(self, f_tr, y_tr, f_va, y_va, pre, seeds: list[int], init_states: list[dict] | None = None,
            epochs: int | None = None, lr: float | None = None) -> NetEnsemble:
        self.pre = pre
        d_tr, d_va = self.tensors(f_tr), self.tensors(f_va)
        self.states, self.logs = [], []
        for k, s in enumerate(seeds):
            st, log = train_member(self.kind, d_tr, y_tr, d_va, y_va, self.cfg, s, epochs, lr,
                                   None if init_states is None else init_states[k])
            self.states.append(st)
            self.logs.append(log)
        return self

    def member_logits(self, f) -> np.ndarray:
        d = self.tensors(f)
        return np.stack([predict_logits(self.kind, s, d, self.cfg) for s in self.states])

    def predict_raw(self, f) -> np.ndarray:
        return self.member_logits(f).mean(0)

    def disagreement(self, f) -> np.ndarray:
        return (1 / (1 + np.exp(-self.member_logits(f)))).std(0)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(self.states):
            torch.save(s, path / f"member{i}.pt")
        meta = {"kind": self.kind, "name": self.name, "rules_on": self.rules_on, "drop": list(self.drop),
                "n_members": len(self.states), "logs": self.logs}
        (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_ensemble(path: Path, cfg: dict[str, Any], pre) -> NetEnsemble:
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    ens = NetEnsemble(meta["kind"], cfg, meta["rules_on"], tuple(meta["drop"]), meta["name"])
    ens.pre, ens.logs = pre, meta["logs"]
    ens.states = [torch.load(path / f"member{i}.pt", map_location="cpu", weights_only=True)
                  for i in range(meta["n_members"])]
    return ens


def fedavg(states: list[dict], weights: list[float]) -> dict:
    """Sample-count weighted average of float parameters (federated averaging)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    out = {}
    for key in states[0]:
        if states[0][key].dtype.is_floating_point:
            out[key] = sum(float(wi) * s[key] for wi, s in zip(w, states, strict=True))
        else:
            out[key] = states[0][key].clone()
    return out
