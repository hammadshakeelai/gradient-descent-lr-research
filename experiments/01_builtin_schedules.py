"""Experiment 1: what every classic LR schedule looks like.

All of these are PyTorch built-ins (see lrlab/schedules.py), so this is
the "use what already exists" step. Output: figures/01_schedules.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lrlab import make_schedule, ALL_SCHEDULES

STEPS, LR = 1000, 0.1
fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey=True)
for ax, name in zip(axes.flat, ALL_SCHEDULES):
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=LR)
    sched = make_schedule(name, opt, STEPS)
    lrs = []
    for _ in range(STEPS):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    ax.plot(lrs)
    ax.set_title(name)
    ax.set_xlabel("step")
axes[0, 0].set_ylabel("learning rate")
axes[1, 0].set_ylabel("learning rate")
fig.suptitle("Classic learning-rate schedules (all built into torch.optim.lr_scheduler)")
fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "figures" / "01_schedules.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=120)
print("saved", out)
