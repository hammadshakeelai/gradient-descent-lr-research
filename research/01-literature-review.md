# Adaptive Learning Rate Research: Literature Review and Gap Analysis

*Date: 2026-09-28. Phase 1 of the project: find out what already exists, then pick a gap worth working on.*

---

## 0. The idea, stated precisely

> Start with a large learning rate (LR), shrink it as training goes on so we land close to the minimum. Better still, **learn the LR itself with a second optimization loop** (one loop for the model weights, one for the LR), so we don't have to run many trials to tune it.

That's two separate claims, and the literature treats them differently:

| Claim | Status in the literature |
|---|---|
| A. Decay the LR over time (big → small) | **Very old.** Required for SGD to converge at all (Robbins & Monro, 1951). Every modern training recipe does it. |
| B. Optimize the LR with its own gradient ("two optimizations at once") | **Exists.** It's called *hypergradient descent* (Almeida et al. 1998; Baydin et al. 2018). The first convergence proof came out at ICML 2025. |
| C. Get rid of LR tuning runs completely | **An active research area** ("learning-rate-free" / "parameter-free" optimizers). Partly solved, and nothing works reliably everywhere yet. |

So the basic idea has been done. The research questions still open are about **how to do it well**, and there are real gaps (Section 5).

---

## 1. Hand-designed decay schedules (claim A)

| Schedule | Formula / shape | Key reference |
|---|---|---|
| Robbins–Monro condition | Σηₜ = ∞, Σηₜ² < ∞ (e.g. ηₜ = η₀/t) | Robbins & Monro, 1951 |
| Step decay | η multiplied by 0.1 every k epochs | Standard ResNet recipe (He et al., 2016) |
| Exponential decay | ηₜ = η₀·γᵗ | Classic |
| Cosine annealing (+ restarts) | ηₜ = η_min + ½(η₀−η_min)(1+cos(πt/T)) | Loshchilov & Hutter, *SGDR*, ICLR 2017 |
| Linear warmup | ramp from 0 to η₀ over the first steps | Goyal et al., 2017 |
| Cyclical LR / 1cycle | LR goes up then down ("super-convergence") | Smith, 2017; Smith & Topin, 2018 |
| Linear decay to zero | ηₜ = η₀(1 − t/T) | Defazio et al., *Optimal Linear Decay LR Schedules*, 2023 |
| Warmup–Stable–Decay (WSD) | constant LR, then a short sharp cooldown at the end | Hägele et al., 2024; Wen et al., 2024 |

**What theory says now (2024–2026):**
- **Convex theory predicts real LLM loss curves.** A last-iterate bound from non-smooth convex optimization, with the base LR set to its optimum, reproduces the empirical loss curves of cosine and WSD schedules ([Schaipp et al., 2025](https://arxiv.org/pdf/2501.18965)). This means we can design schedules with math and check them on real models.
- **"River valley" picture.** The loss surface looks like a valley with a river running along the bottom. A high LR moves fast along the river but bounces between the valley walls; the decay phase stops the bouncing and shows the real progress ([Wen et al., 2024](https://arxiv.org/abs/2410.05192)). Follow-up on the cooldown phase: [arXiv 2508.01483](https://arxiv.org/pdf/2508.01483).
- **The best schedule depends on the task.** Under functional scaling laws, easy tasks favour a power decay to zero, ηₜ = η_peak(1 − t/N)^(2β−1), and hard tasks favour a WSD shape ([Li et al., 2026](https://arxiv.org/abs/2602.06797)).
- Recent benchmark comparing scheduling policies in controlled settings: [BrachistoneLR, arXiv 2609.08069](https://arxiv.org/pdf/2609.08069).

---

## 2. Learning the LR with gradients: the "second optimization" (claim B)

### 2.1 Hypergradient descent (HDM), the closest match to this idea
Take the derivative of the loss with respect to the LR. For plain SGD it's a one-liner:

```
∂L(θₜ)/∂η  =  −∇L(θₜ) · ∇L(θₜ₋₁)
η ← η + β · ∇L(θₜ)·∇L(θₜ₋₁)        # β = "hyper learning rate"
```

If two gradients in a row point the same way, the LR goes up. If they disagree (we overshot), the LR goes down.

- Almeida et al., 1998: the original idea.
- Baydin et al., *Online Learning Rate Adaptation with Hypergradient Descent*, ICLR 2018 ([arXiv 1703.04782](https://arxiv.org/abs/1703.04782)). Adds HD versions of SGD, SGD-Nesterov and Adam.
- **Chu, Gao, Ye, Udell, *Provable and Practical Online LR Adaptation with Hypergradient Descent*, ICML 2025** ([arXiv 2502.11229](https://arxiv.org/abs/2502.11229)). First rigorous convergence proof. It explains why HDM is unstable, fixes it, proves local superlinear convergence, and adds heavy-ball and Nesterov variants. **This is the main paper to read and build on.**

### 2.2 Known failure: short-horizon bias
Wu, Ren, Liao & Grosse, *Understanding Short-Horizon Bias in Stochastic Meta-Optimization*, ICLR 2018. Greedy one-step hypergradients **shrink the LR too early**, because a smaller step always looks better in the short term. This is the main reason pure hypergradient methods lose to a simple cosine schedule on real networks.

### 2.3 Longer-horizon or heavier LR learning
- Maclaurin, Duvenaud & Adams, 2015: backprop through the whole training run. Exact, but memory-heavy.
- Donini et al., *MARTHE*, 2020: forward-mode hypergradients with a longer horizon.
- Jin et al., *AutoLRS*, 2021: Bayesian optimization of the LR while training runs ([arXiv 2105.10762](https://arxiv.org/pdf/2105.10762)).
- Learned schedule networks, e.g. Meta-LR-Schedule-Net (Shu et al.).
- Optimal-control view: the LR schedule as a Hamilton–Jacobi–Bellman problem ([arXiv 2601.07830](https://arxiv.org/html/2601.07830v1)).
- Bandit-based LR selection for deep RL ([LRRL, arXiv 2410.12598](https://arxiv.org/html/2410.12598)).

---

## 3. Learning-rate-free optimizers: no tuning runs (claim C)

| Method | Core idea | Reference |
|---|---|---|
| Polyak step / SPS | η = (f − f*) / ‖∇f‖² | Polyak 1987; Loizou et al., 2021 |
| L4 | Polyak-like step for deep learning | Rolinek & Martius, 2018 |
| D-Adaptation | estimate the distance to the solution, D, online | Defazio & Mishchenko, ICML 2023 |
| Prodigy | a faster-adapting D-Adaptation | Mishchenko & Defazio, 2023 ([arXiv 2306.06101](https://arxiv.org/html/2306.06101)) |
| DoG / DoWG | "Distance over Gradients" | Ivgi, Hinder & Carmon, 2023 |
| Mechanic | learns an LR scale on top of any optimizer | Cutkosky, Defazio & Mehta, 2023 |
| **Schedule-Free SGD/AdamW** | no decay schedule at all; averaging replaces it | Defazio et al., *The Road Less Scheduled*, 2024 |
| ScheduleFree+ | Schedule-Free scaled to LLMs | [arXiv 2605.19095](https://arxiv.org/html/2605.19095v1) |

Code: [facebookresearch/schedule_free](https://github.com/facebookresearch/schedule_free), [konstmish/prodigy](https://github.com/konstmish/prodigy), [LoganBooker/prodigy-plus-schedule-free](https://github.com/LoganBooker/prodigy-plus-schedule-free), and Optax `contrib` (JAX versions of several of these, [docs](https://optax.readthedocs.io/en/stable/api/contrib.html)). The standard benchmark for "train with no tuning" is **AlgoPerf** (MLCommons, Dahl et al., 2023).

**Overall assessment:** [Henheik, Eimer & Lindauer, *Revisiting Learning Rate Control*, 2025](https://arxiv.org/pdf/2507.01724) compared schedules, hyperparameter-free methods and multi-fidelity hyperparameter optimization. Each works on some tasks, and **none is reliable across settings**. HPO gets worse as models get bigger. They also point out that nobody has studied how to *choose* an LR-control method for a given setting.

---

## 4. Corrections to the original intuition (these affect the math)

1. **In deep learning, "reaching the global minimum" isn't really the goal.** Over-parameterized networks have many near-zero-loss minima. What matters is *which* minimum you land in, because that determines generalization and overfitting.
2. **The large early LR helps for a reason besides speed.** It acts as regularization: it pushes the model toward flatter minima that generalize better (Li, Wei & Ma, 2019). Training also sits at the "edge of stability", where sharpness settles near 2/η (Cohen et al., 2021). The LR controls *where* you end up, not only how fast you get there.
3. **Decaying the LR doesn't escape local minima. It traps you in the current basin.** Escaping comes from a *high* LR and gradient noise early on. The decay then settles you in the basin you're in. The river-valley view (Section 1) models this better than the "local vs global minimum" picture.
4. **Two-level optimization is greedy by default** (short-horizon bias, Section 2.2). Any design we make has to deal with this, or it will decay the LR too fast.

---

## 5. Open gaps (where we can contribute)

| # | Gap | Why it's open | Evidence |
|---|---|---|---|
| G1 | **Hypergradient methods that fix short-horizon bias cheaply** | Long-horizon fixes (unrolling, MARTHE) cost too much memory and compute; one-step HDM is greedy | Wu et al. 2018; Chu et al. 2025 only analyse local convergence |
| G2 | **Deciding *when* to start the decay, online** | WSD needs the cooldown start fixed in advance. The river-valley theory says what the decay does, not when to trigger it | Wen et al. 2024; Hägele et al. 2024 |
| G3 | **Reliability across tasks and algorithm selection** | No LR controller wins across settings, and choosing one has hardly been studied | Henheik et al. 2025 |
| G4 | **Combining the approaches**: hypergradient for the peak/stable LR plus a theory-derived decay shape | These are studied separately; hybrid methods are rare | Schaipp 2025 + Li 2026 give the decay shape; HDM gives the level |
| G5 | **Controlling for generalization, not just training loss** | Hypergradients usually minimise training loss, which can push toward sharp minima and overfitting | Li, Wei, Ma 2019; Cohen 2021 |

### Candidate research direction (to formalise in Phase 2)
**"Hypergradient-controlled WSD" (working name):**
1. **Stable phase:** a stabilized hypergradient (following Chu et al. 2025) adjusts the LR level online, with no manual tuning.
2. **Decay trigger:** start the cooldown from a measured signal (for example, loss progress along the "river" flattening out, or disagreement between gradients in a row) instead of a fixed step count. This targets **G2**.
3. **Decay shape:** use the theoretically optimal shape (power decay or linear-to-zero, from Schaipp 2025 and Li 2026) rather than learning it greedily. This avoids short-horizon bias (**G1**).
4. **Optional:** a flatness/sharpness term in the hyper-objective, targeting **G5**.

The claim to test: **one run with no tuning matches a tuned cosine/WSD baseline**, with less spread across tasks than Prodigy or Schedule-Free.

> **Status:** steps 1–3 are prototyped as `HDWSD` in [`lrlab/optimizers.py`](../lrlab/optimizers.py). Math: [02-math-notes.md](02-math-notes.md). First results: [05-results.md](05-results.md).

---

## 6. Proposed experiment plan (Phase 3)

1. **Toy functions:** quadratic, Rosenbrock (river valley), Rastrigin (many local minima). Plot the trajectories.
2. **Convex:** logistic regression, to check against the theory.
3. **Small deep nets:** MNIST MLP, CIFAR-10 ResNet-18.
4. **Small language model:** nanoGPT on TinyShakespeare or OpenWebText subsets.
5. **Baselines:** constant, step, cosine, linear-to-zero, WSD (all tuned with a grid), HD-SGD/HD-Adam (Baydin), Prodigy, Schedule-Free AdamW, D-Adaptation.
6. **Metrics:** final train/val loss, test accuracy (generalization), **number of tuning runs needed**, and sensitivity to the initial LR (sweep η₀ over 3 orders of magnitude). Report at least 3 seeds.

---

## 7. Reading list (priority order)
1. Chu et al., 2025, [arXiv 2502.11229](https://arxiv.org/abs/2502.11229): hypergradient theory
2. Baydin et al., 2018, [arXiv 1703.04782](https://arxiv.org/abs/1703.04782): HD-SGD/HD-Adam
3. Wu et al., 2018: short-horizon bias
4. Defazio et al., 2024: Schedule-Free ([code](https://github.com/facebookresearch/schedule_free))
5. Wen et al., 2024, [arXiv 2410.05192](https://arxiv.org/abs/2410.05192): river valley and WSD
6. Schaipp et al., 2025, [arXiv 2501.18965](https://arxiv.org/pdf/2501.18965): convex theory ↔ schedules
7. Li et al., 2026, [arXiv 2602.06797](https://arxiv.org/abs/2602.06797): optimal schedules under scaling laws
8. Henheik et al., 2025, [arXiv 2507.01724](https://arxiv.org/pdf/2507.01724): nothing is reliable yet
9. Mishchenko & Defazio, Prodigy, [arXiv 2306.06101](https://arxiv.org/html/2306.06101)
10. Li, Wei & Ma, 2019; Cohen et al., 2021: why a large LR helps generalization
