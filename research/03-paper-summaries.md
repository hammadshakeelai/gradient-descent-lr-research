# Paper Summaries (28 papers)

A short summary of each paper, in my own words. Run `python scripts/download_papers.py` to get the PDFs into `papers/`. Each entry gives the file name, what the paper does, and why it matters for this project.

## A. Hypergradient descent: learning the learning rate

| File | Paper | Summary | Why it matters |
|---|---|---|---|
| `baydin2018_hypergradient_descent` | Baydin et al., ICLR 2018, [1703.04782](https://arxiv.org/abs/1703.04782) | Derives ∂L/∂η = −g_t·g_{t−1} and adds a one-line LR update to SGD, Nesterov and Adam. Shows it recovers from bad initial LRs at almost no extra cost. | The core idea behind this project. Our `HDSGD` implements it. |
| `chu2025_provable_hypergradient` | Chu, Gao, Ye & Udell, ICML 2025, [2502.11229](https://arxiv.org/abs/2502.11229) | The first convergence analysis of hypergradient descent, using online learning. Explains why it is unstable, proves local superlinear convergence, and adds momentum variants. | The theory to build on. It motivated the stabilised (cosine / log-space) update in `HDWSD`. |
| `wu2018_short_horizon_bias` | Wu, Ren, Liao & Grosse, ICLR 2018, [1803.02021](https://arxiv.org/abs/1803.02021) | Shows on a noisy quadratic that greedy, short-horizon meta-optimization of the LR is biased and prefers decaying too early. | The main known weakness of learning η greedily. It's why `HDWSD` hands the decay over to theory. |
| `donini2020_marthe` | Donini et al., 2020, [1910.08525](https://arxiv.org/abs/1910.08525) | MARTHE: online hypergradients with a tunable, longer horizon, interpolating between greedy HD and full unrolling. | One way to fix short-horizon bias, at a cost. |
| `maclaurin2015_reversible_learning` | Maclaurin, Duvenaud & Adams, ICML 2015, [1502.03492](https://arxiv.org/abs/1502.03492) | Backpropagates through an entire training run (made memory-feasible with reversible dynamics) to get exact hypergradients for LR schedules. | The "gold standard" hypergradient. Too expensive for big models. |
| `jin2021_autolrs` | Jin et al., ICLR 2021, [2105.10762](https://arxiv.org/abs/2105.10762) | AutoLRS: during training, briefly tries candidate LRs, predicts the loss with a small model, and uses Bayesian optimisation to pick the next LR. | A non-gradient alternative for learning the schedule online. |

## B. Hand-designed schedules

| File | Paper | Summary | Why it matters |
|---|---|---|---|
| `loshchilov2017_sgdr_cosine` | Loshchilov & Hutter, ICLR 2017, [1608.03983](https://arxiv.org/abs/1608.03983) | Cosine annealing with warm restarts (SGDR). | Cosine is the default baseline in most papers. |
| `goyal2017_warmup_large_batch` | Goyal et al., 2017, [1706.02677](https://arxiv.org/abs/1706.02677) | Linear LR scaling with batch size plus a gradual warmup. Trains ImageNet in 1 hour. | Where "warmup" comes from. |
| `smith2017_cyclical_lr` | Smith, WACV 2017, [1506.01186](https://arxiv.org/abs/1506.01186) | Cyclical LRs, plus the "LR range test" for finding a good LR quickly. | A cheap practical way to find the LR scale. |
| `smith2018_super_convergence` | Smith & Topin, 2018, [1708.07120](https://arxiv.org/abs/1708.07120) | The 1cycle policy: LR goes up then down, allowing very large LRs and fast training. | Shows a big LR can be a feature, not something to avoid. |
| `defazio2023_optimal_linear_decay` | Defazio et al., 2023, [2310.07831](https://arxiv.org/abs/2310.07831) | Derives the optimal last-iterate schedule for SGD, which is essentially linear decay to zero. Also refines schedules using observed gradient norms. | Theory behind the cooldown shape in `HDWSD`. |
| `hagele2024_wsd_scaling` | Hägele et al., NeurIPS 2024, [2405.18392](https://arxiv.org/abs/2405.18392) | Constant LR plus a short cooldown matches cosine, and lets you branch off checkpoints for any training length. | Why WSD is now used for LLMs. |
| `wen2024_river_valley_wsd` | Wen et al., 2024, [2410.05192](https://arxiv.org/abs/2410.05192) | The "river valley" landscape explains WSD: a high LR moves fast along the river, and the cooldown removes the side-to-side oscillation. Proposes WSD-S. | The best intuition for *why* decaying works. |
| `schaipp2025_convex_theory_schedules` | Schaipp et al., 2025, [2501.18965](https://arxiv.org/abs/2501.18965) | A convex last-iterate bound predicts real LLM loss curves under cosine and WSD, and can be used to design schedules. | Lets us design a schedule on paper before running it. |
| `li2026_optimal_schedules_scaling_laws` | Li et al., 2026, [2602.06797](https://arxiv.org/abs/2602.06797) | Optimal schedules under functional scaling laws: power decay for easy tasks, WSD shape for hard tasks. | Says the best shape depends on the task, which motivates adaptivity. |

## C. Learning-rate-free optimizers

| File | Paper | Summary | Why it matters |
|---|---|---|---|
| `loizou2021_stochastic_polyak` | Loizou et al., AISTATS 2021, [2002.10542](https://arxiv.org/abs/2002.10542) | The stochastic Polyak step η = (f_i − f_i*)/(c‖∇f_i‖²). Strong when models can interpolate the data. | An LR from the loss value itself. |
| `rolinek2018_l4` | Rolinek & Martius, NeurIPS 2018, [1802.05074](https://arxiv.org/abs/1802.05074) | L4: a Polyak-style step that targets a fraction of the remaining loss, wrapped around Adam or momentum. | An early practical LR-free method for deep learning. |
| `defazio2023_d_adaptation` | Defazio & Mishchenko, ICML 2023, [2301.07733](https://arxiv.org/abs/2301.07733) | Estimates D = ‖w₀ − w*‖ online and sets η from it. Optimal convex rate with no tuning. | A basis for Prodigy. |
| `mishchenko2023_prodigy` | Mishchenko & Defazio, ICML 2024, [2306.06101](https://arxiv.org/abs/2306.06101) | Prodigy: estimates D faster than D-Adaptation, with an Adam variant. Widely used for fine-tuning. | A baseline in Experiment 4 (`pip install prodigyopt`). |
| `ivgi2023_dog` | Ivgi, Hinder & Carmon, ICML 2023, [2302.12022](https://arxiv.org/abs/2302.12022) | DoG: η_t = max distance travelled / √(Σ‖g‖²). Parameter-free with guarantees. | A simple formula worth knowing. |
| `cutkosky2023_mechanic` | Cutkosky, Defazio & Mehta, NeurIPS 2023, [2306.00144](https://arxiv.org/abs/2306.00144) | Mechanic: learns a global LR scale on top of any base optimizer using online convex optimization. | Closest in spirit to "a second optimizer for the LR". |
| `defazio2024_schedule_free` | Defazio et al., NeurIPS 2024, [2405.15682](https://arxiv.org/abs/2405.15682) | Schedule-Free: removes the decay schedule by combining interpolation and averaging. Won the AlgoPerf self-tuning track. | A strong baseline in Experiment 4 (`pip install schedulefree`). |

## D. Benchmarks, surveys, and why a large LR helps

| File | Paper | Summary | Why it matters |
|---|---|---|---|
| `henheik2025_revisiting_lr_control` | Henheik, Eimer & Lindauer, 2025, [2507.01724](https://arxiv.org/abs/2507.01724) | Compares HPO, fixed schedules and hyperparameter-free methods. None is reliable everywhere, and how to choose among them is unstudied. | The evidence that the problem is still open. |
| `dahl2023_algoperf` | Dahl et al., 2023, [2306.07179](https://arxiv.org/abs/2306.07179) | AlgoPerf: a rigorous time-to-result benchmark for training algorithms, with a self-tuning track. | Where a method like HD-WSD would eventually have to compete. |
| `li2019_large_initial_lr_regularization` | Li, Wei & Ma, NeurIPS 2019, [1907.04595](https://arxiv.org/abs/1907.04595) | A large initial LR followed by annealing generalizes better than a small LR, because of what it learns first. | Warns that minimising training loss greedily can hurt test accuracy. |
| `cohen2021_edge_of_stability` | Cohen et al., ICLR 2021, [2103.00065](https://arxiv.org/abs/2103.00065) | In full-batch GD the sharpness rises until it hovers around 2/η: the "edge of stability". | The LR shapes the landscape the optimizer ends up in. |
| `kingma2015_adam` | Kingma & Ba, ICLR 2015, [1412.6980](https://arxiv.org/abs/1412.6980) | Adam: per-parameter adaptive step sizes from moment estimates. | Adam still needs a global LR and a schedule. |
| `loshchilov2019_adamw` | Loshchilov & Hutter, ICLR 2019, [1711.05101](https://arxiv.org/abs/1711.05101) | Decoupled weight decay (AdamW). | Today's default optimizer. |

*Classic reference with no arXiv version:* Robbins & Monro (1951), "A Stochastic Approximation Method", *Annals of Mathematical Statistics*. The origin of the Σηₜ = ∞, Σηₜ² < ∞ conditions.
