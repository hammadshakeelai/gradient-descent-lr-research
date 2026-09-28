"""Prototype optimizers that are NOT built into PyTorch.

PyTorch already ships every classic schedule (StepLR, CosineAnnealingLR,
OneCycleLR, SequentialLR for WSD, ...), and Prodigy / Schedule-Free are on
pip. What is missing is:

1. HDSGD  - hypergradient descent (Baydin et al. 2018, arXiv 1703.04782):
            the learning rate is itself trained by gradient descent.
2. HDWSD  - our prototype hybrid: hypergradient picks the LR level during a
            "stable" phase, then a theory-backed linear cooldown to zero takes
            over (WSD shape, Hägele 2024 / Defazio 2023). The greedy
            hypergradient is switched OFF during the cooldown, because greedy
            one-step LR learning is known to decay too early (Wu et al. 2018).

Both keep one scalar LR per param group so the code stays readable.
"""
import math

import torch
from torch.optim import Optimizer


def _flat_dot(xs, ys):
    return sum((x * y).sum() for x, y in zip(xs, ys))


class HDSGD(Optimizer):
    """SGD (+ optional momentum) whose learning rate is learned online.

    For w_t = w_{t-1} - lr * d_{t-1}, the chain rule gives
        dL(w_t)/d lr = -g_t . d_{t-1}
    so one step of gradient descent on the LR is
        lr <- lr + hyper_lr * (g_t . d_{t-1}).
    If consecutive steps agree (positive dot product) the LR grows,
    if we overshot (negative dot product) it shrinks.
    """

    def __init__(self, params, lr=0.01, hyper_lr=1e-3, momentum=0.0):
        super().__init__(params, dict(lr=lr, hyper_lr=hyper_lr, momentum=momentum))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            grads = [p.grad for p in params]
            prev = [self.state[p].get("direction") for p in params]
            if all(d is not None for d in prev) and prev:
                hypergrad = -_flat_dot(grads, prev)          # dL/d lr
                group["lr"] = max(group["lr"] - group["hyper_lr"] * hypergrad.item(), 0.0)
            for p, g in zip(params, grads):
                st = self.state[p]
                if group["momentum"] > 0:
                    buf = st.get("buf")
                    buf = g.clone() if buf is None else buf.mul_(group["momentum"]).add_(g)
                    st["buf"] = buf
                    d = buf
                else:
                    d = g
                st["direction"] = d.clone()
                p.add_(d, alpha=-group["lr"])
        return loss


class HDWSD(Optimizer):
    """Prototype: Hypergradient-controlled Warmup-Stable-Decay SGD.

    Phases
    ------
    stable   : lr <- lr * exp(hyper_lr * cos(g_t, d_{t-1}))
               A scale-free (cosine) hypergradient on log(lr). Using cosine
               instead of the raw dot product means `hyper_lr` does not have
               to be retuned when the loss scale changes. A short warmup
               (warmup_steps) linearly ramps the LR in first.
    cooldown : lr <- lr_peak * (1 - progress)      (linear to zero)
               Hypergradient is frozen. The decay SHAPE comes from theory
               rather than from the greedy signal.

    When does the cooldown start?
      * If total_steps is known: at (1 - cooldown_frac) * total_steps.
      * If total_steps is None: when the smoothed training loss (pass it
        to step(loss=...)) stops improving for `patience` steps. The
        cooldown then lasts cooldown_frac * (steps so far), so no training
        budget has to be fixed in advance.
    """

    def __init__(self, params, lr=0.01, momentum=0.9, hyper_lr=0.02,
                 lr_min=1e-6, lr_max=10.0, warmup_steps=0,
                 total_steps=None, cooldown_frac=0.2,
                 patience=200, rel_tol=1e-3, ema=0.98):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        self.hyper_lr, self.lr_min, self.lr_max = hyper_lr, lr_min, lr_max
        self.warmup_steps, self.total_steps = warmup_steps, total_steps
        self.cooldown_frac = cooldown_frac
        self.patience, self.rel_tol, self.ema = patience, rel_tol, ema
        self.t = 0
        self.phase = "stable"
        self.base_lr = [g["lr"] for g in self.param_groups]   # learned level
        self._cool_start = self._cool_len = None
        self._peak = None
        self._loss_ema = self._best = None
        self._since_best = 0

    # ---- decay trigger ---------------------------------------------------
    def _maybe_start_cooldown(self, loss):
        if self.phase != "stable":
            return
        if self.total_steps is not None:
            if self.t >= int((1 - self.cooldown_frac) * self.total_steps):
                self._start_cooldown(self.total_steps - self.t)
            return
        if loss is None:
            return
        loss = float(loss)
        self._loss_ema = loss if self._loss_ema is None else self.ema * self._loss_ema + (1 - self.ema) * loss
        if self._best is None or self._loss_ema < self._best * (1 - self.rel_tol):
            self._best, self._since_best = self._loss_ema, 0
        else:
            self._since_best += 1
        if self._since_best >= self.patience and self.t > self.warmup_steps:
            self._start_cooldown(max(1, int(self.cooldown_frac * self.t)))

    def _start_cooldown(self, length):
        self.phase = "cooldown"
        self._cool_start, self._cool_len = self.t, max(1, length)
        self._peak = [g["lr"] for g in self.param_groups]

    # ---- main step -------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None, loss=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.t += 1
        self._maybe_start_cooldown(loss)

        for i, group in enumerate(self.param_groups):
            params = [p for p in group["params"] if p.grad is not None]
            grads = [p.grad for p in params]
            prev = [self.state[p].get("buf") for p in params]

            if self.phase == "stable":
                if prev and all(b is not None for b in prev):
                    num = _flat_dot(grads, prev)
                    den = math.sqrt(_flat_dot(grads, grads).item() * _flat_dot(prev, prev).item()) + 1e-12
                    cos = num.item() / den
                    self.base_lr[i] = min(max(self.base_lr[i] * math.exp(self.hyper_lr * cos),
                                              self.lr_min), self.lr_max)
                warm = min(1.0, self.t / self.warmup_steps) if self.warmup_steps else 1.0
                group["lr"] = self.base_lr[i] * warm
            elif self.phase == "cooldown":
                progress = min(1.0, (self.t - self._cool_start) / self._cool_len)
                group["lr"] = self._peak[i] * (1.0 - progress)
                if progress >= 1.0:
                    self.phase = "done"
            else:  # done
                group["lr"] = 0.0

            for p, g in zip(params, grads):
                st = self.state[p]
                buf = st.get("buf")
                buf = g.clone() if buf is None else buf.mul_(group["momentum"]).add_(g)
                st["buf"] = buf
                p.add_(buf, alpha=-group["lr"])
        return loss
