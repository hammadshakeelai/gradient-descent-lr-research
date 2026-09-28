"""Experiment 4: a small neural network, all methods side by side.

Data : 3-arm spiral (2-D points, 3 classes), generated locally, no download.
Model: MLP 2 -> 64 -> 64 -> 3, minibatch 64, 1500 steps.

Methods
  built-in PyTorch : SGD+momentum with constant / cosine / WSD schedules
  pip packages     : Prodigy (prodigyopt), Schedule-Free SGD (schedulefree)
  prototypes here  : HD-SGD (Baydin 2018), HD-WSD (ours)

Each LR-based method is swept over initial LRs 1e-3..1 with 3 seeds.
We report (a) the best test accuracy after tuning and (b) the median
over the whole LR sweep, which shows how much tuning you need.
Output: figures/04_mlp_benchmark.png and results/04_mlp_benchmark.csv
"""
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lrlab import HDSGD, HDWSD, make_schedule

ROOT = Path(__file__).resolve().parent.parent
STEPS, BATCH, SEEDS = 1500, 64, 3
LR_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
torch.set_num_threads(1)


def spirals(n, seed):
    g = torch.Generator().manual_seed(seed)
    k = torch.randint(0, 3, (n,), generator=g)
    r = torch.rand(n, generator=g)
    t = 4 * r + k * 2 * math.pi / 3 + 0.25 * torch.randn(n, generator=g)
    x = torch.stack([r * torch.cos(t), r * torch.sin(t)], 1)
    return x, k


XTR, YTR = spirals(3000, 0)
XTE, YTE = spirals(3000, 1)


def build(method, params, lr):
    if method.startswith("SGD+"):
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
        return opt, make_schedule(method[4:], opt, STEPS)
    if method == "HD-SGD":
        return HDSGD(params, lr=lr, hyper_lr=1e-4, momentum=0.9), None
    if method == "HD-WSD (ours)":
        return HDWSD(params, lr=lr, momentum=0.9, hyper_lr=0.02, total_steps=STEPS,
                     warmup_steps=STEPS // 20), None
    if method == "Schedule-Free SGD":
        import schedulefree
        return schedulefree.SGDScheduleFree(params, lr=lr, momentum=0.9, warmup_steps=STEPS // 20), None
    if method == "Prodigy":
        from prodigyopt import Prodigy
        return Prodigy(params, lr=lr), None       # lr is a multiplier; 1.0 is the default
    raise ValueError(method)


def train(method, lr, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
    opt, sched = build(method, model.parameters(), lr)
    if hasattr(opt, "train"):
        opt.train()                                   # Schedule-Free needs train/eval mode switches
    g = torch.Generator().manual_seed(seed)
    for _ in range(STEPS):
        idx = torch.randint(0, len(XTR), (BATCH,), generator=g)
        loss = nn.functional.cross_entropy(model(XTR[idx]), YTR[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched:
            sched.step()
        if not math.isfinite(loss.item()):
            return float("nan"), 1 / 3
    if hasattr(opt, "eval"):
        opt.eval()
    with torch.no_grad():
        train_loss = nn.functional.cross_entropy(model(XTR), YTR).item()
        acc = (model(XTE).argmax(1) == YTE).float().mean().item()
    return train_loss, acc


if __name__ == "__main__":
    methods = ["SGD+constant", "SGD+cosine", "SGD+wsd", "Schedule-Free SGD", "HD-SGD", "HD-WSD (ours)", "Prodigy"]
    rows, acc_by = [], {}
    for m in methods:
        grid = [1.0] if m == "Prodigy" else LR_GRID
        accs = []
        for lr in grid:
            res = [train(m, lr, s) for s in range(SEEDS)]
            acc = float(np.mean([a for _, a in res]))
            loss = float(np.nanmean([l for l, _ in res])) if any(np.isfinite(l) for l, _ in res) else float("nan")
            rows.append(dict(method=m, lr=lr, test_acc=acc, train_loss=loss))
            accs.append(acc)
            print(f"{m:18s} lr={lr:<7g} test_acc={acc:.3f}  train_loss={loss:.4f}")
        acc_by[m] = (grid, accs)

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "04_mlp_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(), w.writerows(rows)

    print("\nmethod              best acc (tuned)   median acc over LR sweep")
    for m in methods:
        _, accs = acc_by[m]
        print(f"{m:18s}  {max(accs):.3f}              {np.median(accs):.3f}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in methods:
        grid, accs = acc_by[m]
        if m == "Prodigy":
            ax.axhline(accs[0], color="k", ls=":", label="Prodigy (no LR to tune)")
        else:
            ax.semilogx(grid, accs, "o-", label=m)
    ax.set_xlabel("initial / base learning rate"), ax.set_ylabel("test accuracy (mean of 3 seeds)")
    ax.set_title("Spiral MLP: accuracy vs. learning rate (flat & high = robust)")
    ax.legend(fontsize=8), ax.set_ylim(0.3, 1.0)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "04_mlp_benchmark.png", dpi=110)
    print("saved figures/04_mlp_benchmark.png")
