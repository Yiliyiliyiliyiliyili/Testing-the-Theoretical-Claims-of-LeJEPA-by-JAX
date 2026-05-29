"""
show_results.py
---------------
Load results.pkl and display all experimental results as text tables and figures.

Usage:
    python show_results.py
    python show_results.py --path /path/to/results.pkl

Requirements: numpy, pandas, matplotlib
"""

import argparse
import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Load ──────────────────────────────────────────────────────────────────────

def load(path="results.pkl"):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: '{path}' not found.")
        print("Make sure results.pkl is in the same directory as this script.")
        sys.exit(1)
    return (
        data["baseline_results"],
        data["main_results"],
        data["probe_results"],
    )


# ── Text output ───────────────────────────────────────────────────────────────

def print_separator(title=""):
    width = 60
    print(f"\n{'='*width}")
    if title:
        print(f"  {title}")
        print(f"{'='*width}")


def print_training_snapshots(name, hist, every=2000):
    print(f"\n--- {name} ---")
    print(f"  {'Step':>6}  {'Loss':>8}  {'SIGReg':>8}  {'Emb Var':>10}")
    print(f"  {'-'*40}")
    for i, step in enumerate(hist["steps"]):
        if step == 0 or step % every == 0 or step == hist["steps"][-1]:
            print(
                f"  {step:6d}  "
                f"{hist['train_loss'][i]:8.4f}  "
                f"{hist['sigreg'][i]:8.4f}  "
                f"{hist['emb_var'][i]:10.6f}"
            )


def print_final_summary(main_results, probe_results):
    rows = []
    for name, hist in main_results.items():
        pr = probe_results.get(name, {})
        rows.append({
            "Condition":  name,
            "SIGReg":     f"{hist['sigreg'][-1]:.4f}",
            "Inv Loss":   f"{hist['inv'][-1]:.4f}",
            "Emb Var":    f"{hist['emb_var'][-1]:.4f}",
            "Grad Ratio": (
                f"{hist['grad_ratio'][-1]:.4f}"
                if hist["grad_ratio"] else "N/A"
            ),
            "Test Acc":   (
                f"{pr['test_acc']*100:.2f}%"
                if "test_acc" in pr else "—"
            ),
            "Train Acc":  (
                f"{pr['train_acc']*100:.2f}%"
                if "train_acc" in pr else "—"
            ),
        })
    print(pd.DataFrame(rows).set_index("Condition").to_string())


def print_lambda_zones():
    zones = [
        ("Collapse",  "λ ≤ 0.01", "Permanent collapse, emb_var -> 0, test acc = random"),
        ("Critical",  "λ = 0.05", "Delayed escape (~4500 steps), stochastic"),
        ("Stable",    "λ = 0.1",  "No collapse, best view alignment"),
        ("High-reg",  "λ = 0.5",  "Fastest SIGReg convergence, elevated inv loss"),
    ]
    print()
    print(f"  {'Zone':<12}  {'Lambda':<12}  Behavior")
    print(f"  {'-'*60}")
    for zone, lam, behavior in zones:
        print(f"  {zone:<12}  {lam:<12}  {behavior}")


# ── Plots ─────────────────────────────────────────────────────────────────────

COLORS = {
    "VICReg":          "#1f77b4",
    "LeJEPA_lam0.01":  "#ffb347",
    "LeJEPA_lam0.05":  "#d62728",
    "LeJEPA_lam0.1":   "#9467bd",
    "LeJEPA_lam0.5":   "#8c564b",
}
BASELINE_COLORS = {
    "PureInv": "#7f7f7f",
    "PureSIG": "#2ca02c",
}
CORE = {"VICReg", "LeJEPA_lam0.1"}


def lw(name):
    return 2.5 if name in CORE else 1.5


def plot_baseline(baseline_results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for name, hist in baseline_results.items():
        ax.plot(hist["steps"], hist["sigreg"], label=name,
                color=BASELINE_COLORS.get(name, "#555"), linewidth=2.0)
    ax.set_xlabel("Step"); ax.set_ylabel("SIGReg Loss")
    ax.set_title("Baseline Conditions: SIGReg Loss\n"
                 "PureInv=no regularization  PureSIG=no invariance")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    for name, hist in baseline_results.items():
        ax.plot(hist["steps"], hist["emb_var"], label=name,
                color=BASELINE_COLORS.get(name, "#555"), linewidth=2.0)
    ax.set_xlabel("Step"); ax.set_ylabel("Embedding Variance")
    ax.set_title("Baseline Conditions: Embedding Variance\n"
                 "collapse->0  /  over-regularization>>1")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_main(main_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for name, hist in main_results.items():
        ax.plot(hist["steps"], hist["sigreg"], label=name,
                color=COLORS.get(name, "#555"), linewidth=lw(name), alpha=0.9)
    ax.set_xlabel("Step"); ax.set_ylabel("SIGReg Loss")
    ax.set_title("Distribution Quality  (SIGReg Loss, lower is better)\n"
                 "All conditions evaluated on the same scale")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for name, hist in main_results.items():
        ax.plot(hist["steps"], hist["emb_var"], label=name,
                color=COLORS.get(name, "#555"), linewidth=lw(name), alpha=0.9)
    ax.set_xlabel("Step"); ax.set_ylabel("Mean Embedding Variance")
    ax.set_title("Embedding Variance  (collapse -> 0)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    for name, hist in main_results.items():
        if not hist["grad_ratio"]:
            continue
        ax.semilogy(hist["grad_ratio_steps"], hist["grad_ratio"],
                    label=name, color=COLORS.get(name, "#555"),
                    linewidth=lw(name), alpha=0.9)
    ax.set_xlabel("Step")
    ax.set_ylabel("||d(SIGReg)/dtheta|| / ||d(Inv)/dtheta||")
    ax.set_title("Gradient Ratio  (log scale, no lambda scaling)\n"
                 "Guard dynamic: strong early, recedes as dist -> Gaussian")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_probe(probe_results):
    names      = list(probe_results.keys())
    test_accs  = [probe_results[n]["test_acc"]  * 100 for n in names]
    train_accs = [probe_results[n]["train_acc"] * 100 for n in names]

    x   = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x - 0.2, test_accs,  width=0.4, label="Test Acc",
                  color=[COLORS.get(n, "#555") for n in names], alpha=0.9)
    ax.bar(      x + 0.2, train_accs, width=0.4, label="Train Acc",
                  color=[COLORS.get(n, "#555") for n in names], alpha=0.4)

    for bar, val in zip(bars, test_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Linear Probe Accuracy  (frozen 512-dim backbone, CIFAR-10)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 75)

    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Display experimental results from results.pkl"
    )
    parser.add_argument(
        "--path", default="results.pkl",
        help="Path to results.pkl (default: results.pkl)"
    )
    args = parser.parse_args()

    baseline_results, main_results, probe_results = load(args.path)

    print_separator("BASELINE GROUP  (1000 steps)")
    for name, hist in baseline_results.items():
        print_training_snapshots(name, hist, every=500)

    print_separator("MAIN EXPERIMENT  (20000 steps)")
    for name, hist in main_results.items():
        print_training_snapshots(name, hist, every=2000)

    print_separator("FINAL VALUES + LINEAR PROBE")
    print_final_summary(main_results, probe_results)

    print_separator("LAMBDA BEHAVIOR ZONES")
    print_lambda_zones()

    plot_baseline(baseline_results)
    plot_main(main_results)
    plot_probe(probe_results)


if __name__ == "__main__":
    main()
