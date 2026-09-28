# What Is Already Built In, and What We Had to Prototype

Rule used in this repo: **if a well-tested implementation exists, use it**. Only write code for what's missing.

## Built into PyTorch (`torch.optim.lr_scheduler`)

| Schedule from the literature | PyTorch class |
|---|---|
| Constant | `ConstantLR(factor=1.0)` or no scheduler |
| Step decay | `StepLR`, `MultiStepLR` |
| Exponential decay | `ExponentialLR` |
| Polynomial / power decay | `PolynomialLR` |
| Linear decay / linear warmup | `LinearLR` |
| Cosine annealing | `CosineAnnealingLR` |
| SGDR (warm restarts) | `CosineAnnealingWarmRestarts` |
| Cyclical LR | `CyclicLR` |
| 1cycle | `OneCycleLR` |
| Warmup-Stable-Decay | `SequentialLR([LinearLR, ConstantLR, LinearLR])` (no dedicated class, but it composes) |
| Decay when the loss plateaus | `ReduceLROnPlateau` |
| Any formula you want | `LambdaLR(lambda step: ...)` |

See [`lrlab/schedules.py`](../lrlab/schedules.py) and `experiments/01_builtin_schedules.py`.

## Available on pip (official author code)

| Method | Install | Class |
|---|---|---|
| Schedule-Free SGD / AdamW | `pip install schedulefree` | `schedulefree.SGDScheduleFree`, `AdamWScheduleFree` |
| Prodigy | `pip install prodigyopt` | `prodigyopt.Prodigy` |
| D-Adaptation | `pip install dadaptation` | `dadaptation.DAdaptSGD`, `DAdaptAdam` |
| DoG | `pip install dog-optimizer` | `dog.DoG` |
| Mechanic | `pip install mechanic-pytorch` | `mechanic_pytorch.mechanize` |

Only Prodigy and Schedule-Free are used in the experiments here, to keep the dependency list short.

## NOT built in, so we prototyped them

| Method | Where | Notes |
|---|---|---|
| **HD-SGD** (hypergradient descent, Baydin 2018) | `lrlab/optimizers.py::HDSGD` | Not in PyTorch or a maintained pip package. The authors' research code is at github.com/gbaydin/hypergradient-descent. Ours is about 30 lines. |
| **HD-WSD** (our hybrid) | `lrlab/optimizers.py::HDWSD` | New combination: a stabilised hypergradient sets the LR *level*, and a linear cooldown to zero (a WSD shape from theory) handles the *decay*. It can trigger the cooldown on a loss plateau when the training budget is unknown. |

### Using the prototype

```python
from lrlab import HDWSD

opt = HDWSD(model.parameters(), lr=1e-3,       # any rough guess, it adapts
            momentum=0.9, hyper_lr=0.02,
            total_steps=10_000,                # or None -> plateau-triggered cooldown
            warmup_steps=500)
for x, y in loader:
    loss = loss_fn(model(x), y)
    opt.zero_grad(); loss.backward()
    opt.step(loss=loss.item())                 # the loss is only needed when total_steps=None
    print(opt.phase, opt.param_groups[0]["lr"])
```
