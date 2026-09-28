"""Experiment 3: can the optimizer find its own learning rate?

Task: a noisy, ill-conditioned 50-D quadratic (condition number 100).
We start every method from initial LRs spread over 4 orders of magnitude
and look at the final loss. A method that "tunes itself" should give a
flat curve: similar final loss whatever LR you start from.

Also plots how the LR evolves inside HD-SGD and HD-WSD. HD-SGD finds a
sensible LR level quickly but never anneals it to zero, so it stays at
the gradient-noise floor. The one-step greedy signal can't "see" the
benefit of a final decay (the short-horizon issue, Wu et al. 2018).
HD-WSD adds that decay from theory instead.
Output: figures/03_sensitivity.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lrlab import HDSGD, HDWSD, make_schedule

ROOT = Path(__file__).resolve().parent.parent
DIM, STEPS, NOISE = 50, 2000, 1.0
H = torch.logspace(0, 2, DIM, dtype=torch.float64)       # curvatures 1..100, L = 100


def loss_fn(w):
    return 0.5 * (H * w ** 2).sum()


def run(method, lr0, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.ones(DIM, dtype=torch.float64, requires_grad=True)
    sched = None
    if method == "HD-SGD (Baydin 2018)":
        opt = HDSGD([w], lr=lr0, hyper_lr=1e-7, momentum=0.9)  # 1e-6 diverges: the raw dot product is scale-sensitive
    elif method == "HD-WSD (ours)":
        opt = HDWSD([w], lr=lr0, momentum=0.9, hyper_lr=0.02, total_steps=STEPS)
    else:
        opt = torch.optim.SGD([w], lr=lr0, momentum=0.9)
        sched = make_schedule(method.split()[0], opt, STEPS)
    lrs = []
    for _ in range(STEPS):
        opt.zero_grad()
        loss = loss_fn(w)
        loss.backward()
        w.grad += NOISE * torch.randn(DIM, generator=g, dtype=torch.float64)
        opt.step()
        if sched:
            sched.step()
        lrs.append(opt.param_groups[0]["lr"])
        if not torch.isfinite(w).all():
            return float("inf"), lrs
    return loss_fn(w).item(), lrs


if __name__ == "__main__":
    methods = ["constant SGD", "cosine SGD", "wsd SGD", "HD-SGD (Baydin 2018)", "HD-WSD (ours)"]
    lr_grid = np.logspace(-5, -1, 9)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    print(f"final loss (median of 3 seeds); stability limit 2/L = {2 / 100:.3g} (momentum lowers it)")
    print("init lr    " + "".join(f"{m[:14]:>16s}" for m in methods))
    table = {m: [] for m in methods}
    for lr0 in lr_grid:
        row = []
        for m in methods:
            v = np.median([run(m, lr0, s)[0] for s in range(3)])
            table[m].append(v)
            row.append(v)
        print(f"{lr0:8.1e}   " + "".join(f"{v:16.4g}" for v in row))
    for m in methods:
        a1.loglog(lr_grid, np.minimum(table[m], 1e4), "o-", label=m)
    a1.set_xlabel("initial learning rate"), a1.set_ylabel("final loss (capped at 1e4)")
    a1.set_title("Sensitivity to the initial LR (flatter = less tuning)"), a1.legend(fontsize=8)

    for lr0 in [1e-5, 1e-3, 1e-2]:
        for m, ls in [("HD-SGD (Baydin 2018)", "--"), ("HD-WSD (ours)", "-")]:
            a2.semilogy(run(m, lr0, 0)[1], ls, label=f"{m[:6]} start {lr0:g}")
    a2.set_xlabel("step"), a2.set_ylabel("learning rate"), a2.legend(fontsize=7)
    a2.set_title("How the learned LR moves during training")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "03_sensitivity.png", dpi=110)
    print("saved figures/03_sensitivity.png")
