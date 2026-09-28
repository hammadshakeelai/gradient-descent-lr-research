"""Thin helpers that build the classic schedules from PyTorch BUILT-INS.

Nothing here is new; it just shows which torch.optim.lr_scheduler class
gives you each schedule from the literature.
"""
from torch.optim import lr_scheduler as S


def make_schedule(name, opt, total_steps, warmup_frac=0.05, cooldown_frac=0.2):
    warm = max(1, int(warmup_frac * total_steps))
    if name == "constant":
        return S.ConstantLR(opt, factor=1.0, total_iters=0)
    if name == "step":                       # x0.1 at 50% and 75% (ResNet recipe)
        return S.MultiStepLR(opt, milestones=[total_steps // 2, 3 * total_steps // 4], gamma=0.1)
    if name == "exponential":                # ends at ~1% of the start value
        return S.ExponentialLR(opt, gamma=0.01 ** (1 / total_steps))
    if name == "cosine":                     # Loshchilov & Hutter 2017
        return S.CosineAnnealingLR(opt, T_max=total_steps)
    if name == "cosine_restarts":            # SGDR
        return S.CosineAnnealingWarmRestarts(opt, T_0=max(1, total_steps // 4))
    if name == "linear":                     # linear decay to zero (Defazio 2023)
        return S.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=total_steps)
    if name == "onecycle":                   # Smith & Topin 2018
        return S.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=total_steps + 1)
    if name == "wsd":                        # warmup -> stable -> linear cooldown
        cool = max(1, int(cooldown_frac * total_steps))
        stable = total_steps - warm - cool
        return S.SequentialLR(opt, schedulers=[
            S.LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=warm),
            S.ConstantLR(opt, factor=1.0, total_iters=stable),
            S.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=cool),
        ], milestones=[warm, warm + stable])
    raise ValueError(name)


ALL = ["constant", "step", "exponential", "cosine", "cosine_restarts", "linear", "onecycle", "wsd"]
