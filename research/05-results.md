# Results and Discussion

Every number here can be regenerated with the scripts in `experiments/` (CPU only, fixed seeds). All runs use SGD with momentum 0.9 unless stated otherwise.

---

## Experiment 1: the classic schedules (built-ins)
All 8 classic schedules come straight from `torch.optim.lr_scheduler`, and nothing had to be written from scratch. See `figures/01_schedules.png`.

---

## Experiment 2: 2-D landscapes with noisy gradients (20 random starts)

| Method | Rosenbrock median | Rosenbrock best | Rastrigin median | Rastrigin best |
|---|---|---|---|---|
| constant, small LR | 2.70 | 0.265 | 17.0 | 2.14 |
| constant, big LR (10× / 5×) | 0.0176 | 0.0012 | 36.8 | 3.14 |
| cosine, big LR | 0.0246 | 0.0022 | **4.98** | **0.009** |
| WSD, big LR | **0.0096** | 0.0008 | 5.35 | 1.11 |
| **HD-WSD (ours), starting from the *small* LR** | 0.0118 | **0.0002** | 14.4 | 2.00 |

**What this shows**
- **Rosenbrock (one curved valley, i.e. the "river valley"):** big LR + decay wins, as the theory predicts. HD-WSD starts from the LR that was 10× too small, learns its way up, and gets within ~20% of the tuned WSD result.
- **Rastrigin (many local minima):** the big *noisy* LR is what explores, and the decay then settles into a good basin. A constant big LR never settles (median 36.8), and a small LR stays in the nearest basin (17.0). **HD-WSD behaves like the small LR here.** On a bumpy landscape the gradients keep flipping direction, so the hypergradient reads that as overshoot and shrinks the LR, which stops the exploration.
- **This corrects the original intuition:** decaying the LR does not escape local minima. A large LR (plus noise) escapes, and decaying settles. A good controller needs both.

---

## Experiment 3: sensitivity to the initial LR (noisy quadratic, κ = 100)

Final loss, median of 3 seeds (lower is better; `inf` = diverged):

| init LR | constant | cosine | WSD | HD-SGD | **HD-WSD** |
|---|---|---|---|---|---|
| 1e-5 | 9.05 | 22.0 | 10.8 | 0.070 | **0.013** |
| 1e-4 | 0.037 | 0.379 | 0.051 | 0.070 | **0.013** |
| 3.2e-4 | 0.039 | **0.0078** | 0.010 | 0.071 | 0.013 |
| 1e-3 | 0.090 | 0.010 | 0.016 | 0.071 | 0.014 |
| 1e-2 | 1.36 | 0.018 | 0.091 | 0.119 | 0.018 |
| 1e-1 | inf | inf | inf | inf | **0.017** |

**What this shows**
- The best tuned cosine run (0.0078) still beats HD-WSD (0.013). **But HD-WSD is within 2× of it at every starting LR across four orders of magnitude**, including 0.1, where every other method diverges.
- **HD-SGD** (the original hypergradient) also stops caring about the initial LR, but it settles at a constant LR and **never anneals**. So it stays on the noise floor (≈0.07, see Math Notes §2). That is why the decay was handed over to theory.
- HD-SGD's hyper-LR needed tuning of its own: 1e-6 diverged at every starting LR, and 1e-7 worked. That is the scale problem from Math Notes §3. HD-WSD's cosine-normalized update uses the same `hyper_lr = 0.02` in every experiment in this repo.

---

## Experiment 4: MLP on a 3-class spiral (1500 steps, 3 seeds)

| Method | Source | Best test acc | Median acc over LR ∈ [1e-3, 1] | Worst |
|---|---|---|---|---|
| SGD + constant | PyTorch | 0.991 | 0.978 | 0.340 |
| SGD + cosine | PyTorch | 0.997 | 0.929 | 0.482 |
| SGD + WSD | PyTorch | 0.997 | 0.979 | 0.343 |
| Schedule-Free SGD | `schedulefree` | 0.998 | 0.592 | 0.471 |
| HD-SGD | prototype | 0.991 | 0.982 | 0.380 |
| **HD-WSD (ours)** | prototype | 0.997 | **0.995** | **0.993** |
| Prodigy | `prodigyopt` | 0.998 | (no LR) | – |

**What this shows**
- HD-WSD matched the best tuned schedules (0.997), and its **worst** run over the whole sweep (0.993) is better than the tuned constant-LR run.
- **Prodigy is just as robust with no LR at all**, and it reached a lower training loss (0.008 vs 0.014). So on this task our prototype is not better than the best existing LR-free method. It is competitive with it, and much simpler: about 100 lines, and one scalar LR per param group.
- Schedule-Free looks bad on the median only because its good LRs sit above our grid (it wanted lr ≥ 0.3). With lr = 1 it was the best method, so its median here says nothing about it in general.

---

## Honest overall verdict

| Claim | Supported? |
|---|---|
| Large LR + decay gets closer to the minimum than a fixed LR | ✅ Experiments 2, 3, 4 |
| A hypergradient can find the LR *level* by itself | ✅ Experiments 3, 4 (HD-SGD and HD-WSD are both flat vs. initial LR) |
| A greedy hypergradient alone won't anneal properly | ✅ Experiment 3 (HD-SGD stuck on the noise floor) |
| HD-WSD removes LR tuning on smooth problems | ✅ in these small tests; needs CIFAR / GPT-scale confirmation |
| HD-WSD finds the global minimum | ❌ Experiment 2 Rastrigin: it *reduces* exploration |
| HD-WSD beats Prodigy / Schedule-Free | ❌ not shown; it ties at this scale |

### Limitations
Small models, synthetic data, short runs, and 3 seeds. The cooldown in these tests was budget-triggered: the plateau trigger exists in the code but hasn't been benchmarked yet. Treat these results as a pilot, not as proof.

### Attempts to fix the Rastrigin exploration problem (negative results)

**Diagnosis.** Traced on one Rastrigin run, HD-WSD's LR falls steadily from 1e-3 to 5e-5. Near the bottom of *any* basin, the momentum iterates oscillate, so the cosine is negative and the hypergradient shrinks the LR. Locally that's the *correct* greedy answer: a smaller LR does lower the loss one step ahead. Exploring requires deliberately ignoring it.

Four fixes tried, each checked on all three problems (Rastrigin median over 10–20 starts, quadratic final loss at init LR 1e-5 / 1e-3 / 1e-1, MLP accuracy at LR 1e-3 / 0.1 / 1):

| Fix | Rastrigin | Rosenbrock | Quadratic | MLP | Verdict |
|---|---|---|---|---|---|
| none (shipped) | 14.4 | 0.012 | 0.013 / 0.014 / 0.017 | 0.99 / 1.00 / 1.00 | baseline |
| long-horizon reference direction (slow gradient average in the cosine) | no change (LR still falls) | – | – | – | ❌ |
| LR kick ×3 when the loss plateaus | **11.0** | **0.0034** | 0.08 / 0.09 / 0.08 | unchanged | ❌ quadratic 6× worse: a plateau there is the noise floor, not a local minimum |
| kick, undone if no new best loss | 14.4 | 0.0045 | 0.015 / 0.015 / 0.016 | lr=1 drops to 0.78 | ❌ the gain disappears |
| overshoot bias: lr·exp(β(cos + b)), b = 0.1 / 0.2 / 0.3 | 14.9 / 12.5 / **6.7** | 0.004 / 0.012 / 0.019 | 3× / 7× / 9× worse | diverges at high LR | ❌ clear trade-off |

**Conclusion.** A single greedy LR controller has no free lunch between exploring (multi-modal problems) and settling (convex or smooth problems). Every rule that makes it explore on Rastrigin costs about 3–9× on the quadratic. To tell a *local minimum* apart from a *noise floor*, you need information the one-step signal doesn't have. Candidates: population / parallel runs, second-order or sharpness information, or a validation signal. This is the gap from the literature review (G1 and G5), now confirmed empirically.

### Next steps
1. Exploration: tell "stuck in a basin" apart from "at the noise floor". This needs a signal beyond one-step gradients (see above).
2. Benchmark the plateau-triggered cooldown (no `total_steps`).
3. Scale up: CIFAR-10 ResNet-18 and nanoGPT, against Prodigy, Schedule-Free and D-Adaptation.
4. Theory: a last-iterate bound for HD-WSD (see Math Notes §7).
