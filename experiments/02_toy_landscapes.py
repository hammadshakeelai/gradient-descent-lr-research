"""Experiment 2: LR schedules on two classic 2-D landscapes, with noisy gradients.

* Rosenbrock : a curved narrow valley (the "river valley" picture, Wen et al. 2024).
* Rastrigin  : lots of local minima; global minimum 0 at (0, 0).

Question: does "big LR first, then decay" actually get us closer to the
minimum than a fixed LR? We run many random starts per method and report
the median final loss, so one lucky run can't fool us.
Output: figures/02_rosenbrock.png, figures/02_rastrigin.png
"""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lrlab import HDWSD, make_schedule

ROOT = Path(__file__).resolve().parent.parent
N_STARTS = 20  # ~4 min on a laptop CPU; raise it for tighter statistics
(ROOT / "figures").mkdir(exist_ok=True)


def rosenbrock(p):
    x, y = p[..., 0], p[..., 1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rastrigin(p):
    return 20 + (p ** 2 - 10 * torch.cos(2 * math.pi * p)).sum(-1)


def run(f, start, method, lr, steps, noise, seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    if method == "hd_wsd (ours)":
        opt, sched = HDWSD([p], lr=lr, momentum=0.9, total_steps=steps), None
    else:
        opt = torch.optim.SGD([p], lr=lr, momentum=0.9)
        sched = make_schedule(method.split()[0], opt, steps)
    path = [p.detach().clone().numpy()]
    for _ in range(steps):
        opt.zero_grad()
        f(p).backward()
        p.grad += noise * torch.randn(2, generator=g, dtype=torch.float64)  # stochastic gradient
        torch.nn.utils.clip_grad_norm_([p], 1e3)                           # keep toy runs finite
        opt.step()
        if sched:
            sched.step()
        path.append(p.detach().clone().numpy())
    return np.array(path), f(p).item()


def study(name, f, starts_fn, lr_small, lr_big, steps, noise, extent, levels):
    methods = [("constant (small lr)", lr_small), ("constant (big lr)", lr_big),
               ("cosine (big lr)", lr_big), ("wsd (big lr)", lr_big), ("hd_wsd (ours)", lr_small)]
    rng = np.random.default_rng(0)
    starts = [starts_fn(rng) for _ in range(N_STARTS)]
    print(f"\n== {name}: median / best final loss over {N_STARTS} random starts ==")
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4))
    X, Y = np.meshgrid(np.linspace(*extent[:2], 300), np.linspace(*extent[2:], 300))
    Z = f(torch.tensor(np.stack([X, Y], -1))).numpy()
    for ax, (m, lr) in zip(axes, methods):
        finals = []
        for i, s in enumerate(starts):
            path, fin = run(f, s, m, lr, steps, noise, seed=i)
            finals.append(fin if np.isfinite(fin) else np.inf)
            if i == 0:
                show = path
        finals = np.array(finals)
        print(f"{m:22s} median={np.median(finals):10.4f}   best={finals.min():10.4f}")
        ax.contour(X, Y, Z, levels=levels, cmap="viridis", linewidths=0.6)
        ax.plot(show[:, 0], show[:, 1], "r-", lw=0.7, alpha=0.8)
        ax.plot(*show[0], "ko"), ax.plot(*show[-1], "r*", ms=12)
        ax.set_xlim(*extent[:2]), ax.set_ylim(*extent[2:])
        ax.set_title(f"{m}\nmedian final loss {np.median(finals):.3g}", fontsize=9)
    fig.suptitle(f"{name}: SGD+momentum with noisy gradients (one example path shown)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"02_{name.lower()}.png", dpi=110)


if __name__ == "__main__":
    study("Rosenbrock", rosenbrock, lambda r: [r.uniform(-2, 0), r.uniform(1, 3)],
          lr_small=2e-5, lr_big=2e-4, steps=3000, noise=5.0,
          extent=(-2.2, 2.0, -1.0, 3.5), levels=np.logspace(-1, 3.5, 20))
    study("Rastrigin", rastrigin, lambda r: list(r.uniform(-5, 5, 2)),
          lr_small=1e-3, lr_big=5e-3, steps=2000, noise=20.0,
          extent=(-5.5, 5.5, -5.5, 5.5), levels=15)
