# Math Notes: Learning Rates from First Principles

This is the math behind every experiment in the repo, derived step by step. Only calculus and a little linear algebra are assumed.

---

## 1. Gradient descent and the stability limit

Update: **w ← w − η ∇L(w)**.

Take a 1-D quadratic L(w) = ½ h w² with curvature h > 0. Then ∇L = h w and

```
w_{t+1} = (1 − η h) w_t   ⇒   w_t = (1 − η h)^t w_0
```

- It converges when |1 − ηh| < 1, i.e. **0 < η < 2/h**.
- It is fastest at η = 1/h (one step lands exactly on the minimum).
- With many directions (curvatures h₁ ≤ … ≤ h_d, where L = h_max is the "smoothness"), the stiffest direction sets the limit, **η < 2/L**, and the flattest direction sets the speed, rate ≈ (1 − η h_min). The *condition number* κ = L / h_min tells you how bad this tension is.

➡ This is why one fixed LR is a compromise. It has to be small enough for the steep directions, and that makes it slow in the flat ones.

Momentum (heavy ball, β = 0.9): w ← w − η d, with d ← β d + ∇L. For a quadratic this is stable when **η h < 2(1+β)**, so momentum *raises* the LR limit to 2(1+β)/L ≈ 3.8/L for β = 0.9. Near the limit, though, the iterates oscillate more, because the effective step is about η/(1−β) = 10η. (Experiment 3: constant SGD survives η = 0.032 > 2/L = 0.02, but diverges at η = 0.1 > 0.038.)

*Experiment 3 uses h from 1 to 100 (κ = 100) to show this.*

---

## 2. Why noise forces the LR to decay (Robbins–Monro)

With stochastic gradients g = ∇L + ξ, where the noise ξ has variance σ², a constant η does **not** converge. It settles in a "noise ball" around the minimum. For the quadratic:

```
E[L(w_∞)]  ≈  η σ² / (2(2 − η h))  ∝  η
```

Halving the LR roughly halves the leftover loss. To reach the exact minimum you need η_t → 0, but not too quickly:

```
Σ η_t = ∞     (you can still travel any distance)
Σ η_t² < ∞    (the accumulated noise stays finite)
```

η_t = η₀ / t satisfies both. This is the mathematical reason behind "big steps first, then small steps", and it dates to 1951.

*Experiment 3: HD-SGD finds a good LR level but never decays it, so it stays on the noise floor (~0.07). Every method that decays to zero gets about 5–10× lower.*

---

## 3. Hypergradient descent: optimizing η with gradient descent

The idea is to treat η as a parameter too. One SGD step is w_t = w_{t−1} − η d_{t−1}, where d is the update direction (the gradient itself, or the momentum buffer). By the chain rule:

```
∂L(w_t)/∂η  =  ∇L(w_t) · ∂w_t/∂η  =  −g_t · d_{t−1}
```

So the gradient-descent step on η is

```
η ← η − β · ∂L/∂η  =  η + β (g_t · d_{t−1})         (Baydin et al. 2018)
```

What this does:
- **g_t · d_{t−1} > 0**: the new gradient still points the way we just moved, so we could have gone further. Increase η.
- **g_t · d_{t−1} < 0**: we overshot. Decrease η.

**Problem 1: scale.** The dot product scales with ‖g‖². If the loss is 100× bigger, β has to be 10⁴× smaller. We saw this directly: β = 10⁻⁶ diverged and β = 10⁻⁷ worked (Experiment 3). This is the instability that Chu et al. (2025) analyse.

**Fix used in our prototype.** Descend on log η and normalize by the norms, which gives the cosine:

```
log η ← log η + β · cos(g_t, d_{t−1})     ⇔    η ← η · exp(β cos θ)
```

With cos θ ∈ [−1, 1], one step changes η by at most a factor e^β. With β = 0.02 that is ±2% per step, regardless of the loss scale. η also stays positive automatically.

**Problem 2: short horizon.** The hypergradient only asks whether the *next* step would have been better with a different η. The benefit of a final decay (Section 2) only shows up over many steps, so the greedy signal can't learn it. Wu et al. (2018) prove that greedy meta-optimization is biased toward schedules that are too short-sighted.

---

## 4. The decay shape: why linear-to-zero / WSD

For convex Lipschitz problems, the last-iterate bound for SGD with schedule η_t over T steps can be minimised in closed form. The optimum is close to **linear decay to zero**, η_t = η₀(1 − t/T) (Defazio et al. 2023). Schaipp et al. (2025) show that the same bound reproduces real LLM loss curves for cosine and WSD, including the sudden loss drop during the cooldown.

WSD (warmup → constant → short linear cooldown) is a practical version of this. It keeps the LR high to make progress along the "river", then removes the noise ball (Section 2) at the end.

Li et al. (2026) give the optimal shape under scaling laws: η(t) = η_peak (1 − t/N)^(2β−1), where β is a task-difficulty exponent. Easy tasks favour a smooth power decay, and hard tasks favour a long stable phase plus a late decay (WSD).

---

## 5. The prototype: HD-WSD

Combine the two parts that each work well:

| Phase | LR rule | Where the rule comes from |
|---|---|---|
| warmup (optional) | η_t = η̂_t · t / T_w | Goyal et al. 2017 |
| stable | η̂ ← η̂ · exp(β cos(g_t, d_{t−1})), clipped to [η_min, η_max] | hypergradient (Sec. 3) |
| cooldown | η_t = η_peak (1 − progress) | convex theory (Sec. 4) |

- The hypergradient decides **how big** the LR should be. That removes the "which η₀?" tuning, and a local, greedy signal can do this job reliably.
- Theory decides **how to decay**. That is the part the greedy signal gets wrong.
- **When to decay:** at (1 − c)·T if the budget T is known. If it isn't, decay when the smoothed loss has stopped improving for `patience` steps, and make the cooldown c × (steps so far) long.

Code: [`lrlab/optimizers.py`](../lrlab/optimizers.py), class `HDWSD`.

---

## 6. What the math does *not* promise

- **A global minimum.** On non-convex landscapes (Rastrigin, Experiment 2) a large, noisy LR is what explores. Decay *settles* you in whichever basin you're in. The hypergradient signal treats oscillation as overshoot and **shrinks the LR**, which reduces exploration. In our results HD-WSD is great on smooth valleys and poor on Rastrigin. This is a real trade-off, not a bug.
- **Generalization.** Everything above minimises *training* loss. A large early LR also acts as a regularizer toward flat minima (Li, Wei & Ma 2019; Cohen et al. 2021). A hypergradient that shrinks the LR early could hurt test accuracy. Experiment 4 measures test accuracy for this reason.

## 7. Open math problems (possible research)

1. A convergence rate for HD-WSD: the stable phase plus the linear-decay last-iterate bound. Chu et al. 2025 + Defazio et al. 2023 is the natural starting point.
2. A principled decay trigger: the switch point that minimises the Schaipp et al. bound, given the loss observed so far.
3. An exploration-aware hypergradient: keep η large while the landscape is multi-modal, e.g. by adding a noise or sharpness term to the hyper-objective.
