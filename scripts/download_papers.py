"""Download the papers referenced in this repo from arXiv into papers/.

The PDFs are NOT committed to git (authors/arXiv hold the copyright);
run this script to get your own local copy:

    python scripts/download_papers.py
"""
import time
import urllib.request
from pathlib import Path

# (arXiv id, short file name, one-line reason it matters here)
PAPERS = [
    # --- Hypergradient descent: "optimize the learning rate with its own gradient"
    ("1703.04782", "baydin2018_hypergradient_descent", "HD-SGD / HD-Adam, the core idea of this repo"),
    ("2502.11229", "chu2025_provable_hypergradient", "first convergence proof of hypergradient descent (ICML 2025)"),
    ("1803.02021", "wu2018_short_horizon_bias", "why greedy LR learning decays too fast"),
    ("1910.08525", "donini2020_marthe", "longer-horizon online hypergradients"),
    ("1502.03492", "maclaurin2015_reversible_learning", "backprop through the whole training run"),
    ("2105.10762", "jin2021_autolrs", "Bayesian optimisation of the LR on the fly"),
    # --- Hand-designed schedules
    ("1608.03983", "loshchilov2017_sgdr_cosine", "cosine annealing + warm restarts"),
    ("1706.02677", "goyal2017_warmup_large_batch", "linear warmup"),
    ("1506.01186", "smith2017_cyclical_lr", "cyclical learning rates, LR range test"),
    ("1708.07120", "smith2018_super_convergence", "1cycle policy"),
    ("2310.07831", "defazio2023_optimal_linear_decay", "linear decay to zero is (near) optimal"),
    ("2405.18392", "hagele2024_wsd_scaling", "warmup-stable-decay (WSD) schedule"),
    ("2410.05192", "wen2024_river_valley_wsd", "river-valley explanation of WSD"),
    ("2501.18965", "schaipp2025_convex_theory_schedules", "convex bounds predict real LLM loss curves"),
    ("2602.06797", "li2026_optimal_schedules_scaling_laws", "optimal schedule shape: power decay vs WSD"),
    # --- Learning-rate-free / parameter-free
    ("2002.10542", "loizou2021_stochastic_polyak", "Polyak step size for SGD"),
    ("1802.05074", "rolinek2018_l4", "loss-based step size for deep learning"),
    ("2301.07733", "defazio2023_d_adaptation", "D-Adaptation"),
    ("2306.06101", "mishchenko2023_prodigy", "Prodigy"),
    ("2302.12022", "ivgi2023_dog", "DoG: distance over gradients"),
    ("2306.00144", "cutkosky2023_mechanic", "Mechanic LR tuner"),
    ("2405.15682", "defazio2024_schedule_free", "Schedule-Free SGD/AdamW"),
    # --- Benchmarks, surveys, and why large LRs help
    ("2507.01724", "henheik2025_revisiting_lr_control", "no LR controller is reliable everywhere"),
    ("2306.07179", "dahl2023_algoperf", "AlgoPerf training-algorithm benchmark"),
    ("1907.04595", "li2019_large_initial_lr_regularization", "large early LR = implicit regularisation"),
    ("2103.00065", "cohen2021_edge_of_stability", "sharpness settles at 2/lr"),
    ("1412.6980", "kingma2015_adam", "Adam"),
    ("1711.05101", "loshchilov2019_adamw", "AdamW"),
]

OUT = Path(__file__).resolve().parent.parent / "papers"


def main():
    OUT.mkdir(exist_ok=True)
    failed = []
    for arxiv_id, name, _ in PAPERS:
        path = OUT / f"{name}.pdf"
        if path.exists():
            print(f"skip   {path.name}")
            continue
        for url in (f"https://arxiv.org/pdf/{arxiv_id}", f"https://export.arxiv.org/pdf/{arxiv_id}"):
            print(f"fetch  {url} -> {path.name}")
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (lr-research-downloader)", "Accept": "application/pdf,*/*"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                if data[:4] == b"%PDF":
                    path.write_bytes(data)
                    break
                print("       not a PDF, trying mirror")
            except Exception as e:  # network hiccup / rate limit: try the mirror
                print(f"       failed ({e})")
            time.sleep(5)
        else:
            failed.append(arxiv_id)
        time.sleep(3)  # be polite to arXiv
    if failed:
        print("could not download:", ", ".join(failed), "- rerun the script later")


if __name__ == "__main__":
    main()
