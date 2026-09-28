# Roadmap: Future Plans

What's done, what's still open, and the planned order of work. Each item says **what to try**, **why**, and **how we'll know it worked**. Contributions welcome: open an issue that names the item number.

## Where things stand (Sept 2026)

| Area | Status |
|---|---|
| Literature review (28 papers), math notes, paper summaries | ✅ done |
| Built-in schedules + Prodigy / Schedule-Free baselines | ✅ done |
| HD-SGD and HD-WSD prototypes | ✅ done |
| HD-WSD robust to the initial LR (quadratic, MLP) | ✅ shown at small scale |
| **Exploration on multi-modal landscapes (Rastrigin)** | ❌ **open.** 4 fixes tried, all trade-offs ([results](research/05-results.md#attempts-to-fix-the-rastrigin-exploration-problem-negative-results)) |
| Plateau-triggered cooldown (no fixed budget) | ⚠️ implemented, not benchmarked |
| Real-scale benchmarks (CIFAR, GPT) | ❌ not started |
| Theory (convergence proof for HD-WSD) | ❌ not started |

---

## Phase 1: The exploration problem (highest priority)

**The core issue:** from one-step gradients alone, a greedy LR controller can't tell "stuck in a local minimum, so explore" apart from "at the noise floor, so settle". Each candidate below adds the missing information in a different way.

| # | Idea | Extra information it uses | Cost |
|---|---|---|---|
| 1.1 | **Noise-floor test.** Estimate gradient variance σ² (per-sample or EMA) and compare the plateau loss with the predicted noise ball η σ²/(2(2−ηh)) (Math Notes §2). Explore only if the loss is well *above* that prediction. | gradient variance | ~1 extra buffer |
| 1.2 | **Sharpness test.** Estimate top Hessian eigenvalue λ with a few power-iteration steps (Hessian-vector products). A sharp, narrow basin means a local minimum worth leaving. Link to edge of stability (Cohen 2021). | curvature | 1–3 HVPs every N steps |
| 1.3 | **Population / branching.** Run k short copies at LR ×{1, 3, 10} from the same checkpoint, keep the best (like PBT, or WSD-S branching). | parallel runs | k× compute for short windows |
| 1.4 | **Validation-driven kicks.** Kick the LR only if *validation* loss plateaus while train loss is also stuck. | held-out data | one eval per window |
| 1.5 | **Longer-horizon hypergradient.** Differentiate through k unrolled steps (MARTHE-style) instead of 1, which directly targets short-horizon bias. | memory for k steps | k× memory |

**How we'll know it worked:** on Rastrigin, median final loss ≤ 6 (cosine with a tuned big LR gets 5.0), **while** the quadratic stays ≤ 0.02 and the MLP stays ≥ 0.99 at every initial LR. A fix that breaks the smooth problems does not count; that's exactly what failed before.

## Phase 2: Finish and test the prototype

- 2.1 **Benchmark the plateau-triggered cooldown** (`total_steps=None`). Does it match a budget-aware WSD without knowing the budget? Measure the gap vs. step count.
- 2.2 **HD-AdamW variant.** Apply the same cosine hypergradient to Adam's global LR. Most modern models train with AdamW, not SGD.
- 2.3 **Per-layer LRs.** One learned LR per parameter group (layer). Check whether it helps or just adds noise.
- 2.4 **Ablations.** Remove each part in turn (warmup, cosine normalisation, cooldown shape: linear vs. power decay from Li 2026) to see what actually matters.
- 2.5 **Sensitivity of HD-WSD's own knobs** (`hyper_lr`, `cooldown_frac`). The claim is "no tuning needed", so show that the defaults are robust.

## Phase 3: Scale up

| # | Benchmark | Baselines | Budget |
|---|---|---|---|
| 3.1 | CIFAR-10, ResNet-18 | SGD + cosine (tuned), Prodigy, Schedule-Free, D-Adaptation, DoG | free Colab / Kaggle GPU |
| 3.2 | nanoGPT on TinyShakespeare → OpenWebText subset | AdamW + cosine / WSD (tuned), Schedule-Free AdamW, Prodigy | 1 GPU, a few hours |
| 3.3 | AlgoPerf self-tuning track (subset of workloads) | official baselines | later, once 3.1–3.2 look promising |

Report the final metric, **number of tuning runs needed**, and spread over 3+ seeds and a 3-order-of-magnitude initial-LR sweep.

## Phase 4: Theory

- 4.1 **Convergence bound for HD-WSD**: combine the stable-phase hypergradient analysis (Chu et al. 2025) with the linear-decay last-iterate bound (Defazio et al. 2023).
- 4.2 **Optimal switch point**: the cooldown start that minimises the Schaipp et al. (2025) bound, given the loss curve observed so far. A principled replacement for the plateau heuristic.
- 4.3 **Formalise the trade-off** from Phase 1. Can one prove that no one-step-signal controller is optimal on both convex and multi-modal problems? A clean negative result would be publishable in its own right.

## Phase 5: Repo and community

- 5.1 Unit tests for `lrlab` (the schedule shapes, HD-SGD reduces to SGD when hyper_lr = 0, HDWSD phase transitions) plus GitHub Actions CI.
- 5.2 A Colab notebook version of the four experiments for learners.
- 5.3 An interactive explainer page: drag the LR and watch the optimizer path on each landscape.
- 5.4 If Phases 1–3 give a positive result: write it up as a workshop paper (e.g. the NeurIPS OPT workshop).

---

**Suggested order:** 1.1 → 1.2 (cheapest candidates first) → 2.1, 2.2 → 3.1 → 3.2 → 4.x. Phase 5 can run in parallel whenever.
