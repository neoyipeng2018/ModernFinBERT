"""
Generate publication-quality figures for the ModernFinBERT paper.

Figures:
1. Protocol Gap: Bar chart showing same model under different eval protocols
2. Architecture Comparison: Grouped bar chart of BERT vs ModernBERT vs DeBERTa
3. DataBoost Impact: Before/after per-class F1 comparison
4. Confusion Matrix: Best model (LoRA+DataBoost) on FPB 50agree

Usage:
    python scripts/generate_figures.py

NOTE: Some figures require results from Kaggle experiments (DeBERTa, confusion matrix).
      Run with --available-only to generate figures that don't need pending results.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse
import os

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def fig1_protocol_gap():
    """Figure 1: Same model, different evaluation protocol = different numbers."""
    protocols = ["Held-out\n(FPB excluded)", "10-fold CV\n(in-domain FPB)"]
    accuracy = [80.44, 86.88]
    macro_f1 = [77.05, 85.40]

    x = np.arange(len(protocols))
    width = 0.3

    fig, ax = plt.subplots(figsize=(6, 4))
    bars1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="#2196F3", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, macro_f1, width, label="Macro F1", color="#4CAF50", edgecolor="black", linewidth=0.5)

    # Add gap annotation
    ax.annotate(
        "", xy=(0.85, 86.88), xytext=(0.85, 80.44),
        arrowprops=dict(arrowstyle="<->", color="red", lw=1.5),
    )
    ax.text(1.05, 83.5, "6.4pp\ngap", color="red", fontsize=10, ha="left", va="center")

    ax.bar_label(bars1, fmt="%.1f%%", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=9)

    ax.set_ylabel("Score (%)")
    ax.set_title("The Protocol Gap: Same ModernBERT Model,\nDifferent Evaluation Protocol")
    ax.set_xticks(x)
    ax.set_xticklabels(protocols)
    ax.set_ylim(70, 95)
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1_protocol_gap.pdf")
    plt.savefig(path)
    plt.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    plt.close()


def fig2_databoost_impact():
    """Figure 3: DataBoost before/after per-class F1 comparison."""
    classes = ["NEGATIVE", "NEUTRAL", "POSITIVE", "Macro Avg"]

    # From NB02 DataBoost results on aggregated test set
    baseline_f1 = [46.4, 82.3, 71.1, 66.6]  # approximate per-class from paper
    boosted_f1 = [59.2, 85.4, 78.8, 74.4]  # approximate per-class from paper

    x = np.arange(len(classes))
    width = 0.3

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width / 2, baseline_f1, width, label="Baseline", color="#BBDEFB", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, boosted_f1, width, label="+ DataBoost", color="#1565C0", edgecolor="black", linewidth=0.5)

    # Add delta annotations
    for i in range(len(classes)):
        delta = boosted_f1[i] - baseline_f1[i]
        ax.text(x[i] + width / 2, boosted_f1[i] + 1.5, f"+{delta:.1f}", ha="center", fontsize=8, color="#1565C0", fontweight="bold")

    ax.set_ylabel("F1 Score (%)")
    ax.set_title("DataBoost Impact: Targeted Augmentation\nof Misclassified Examples")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(30, 95)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_databoost_impact.pdf")
    plt.savefig(path)
    plt.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    plt.close()


def fig3_architecture_comparison():
    """Figure 4: Architecture comparison (BERT vs ModernBERT vs DeBERTa)."""
    models = ["BERT-base", "DeBERTa-v3", "ModernBERT"]

    # Held-out FPB 50agree results
    accuracy_50 = [73.09, None, 80.93]  # DeBERTa TBD
    # allAgree results
    accuracy_all = [83.66, None, 93.29]  # DeBERTa TBD

    # Check if DeBERTa results exist
    has_deberta = accuracy_50[1] is not None

    if not has_deberta:
        print("WARNING: DeBERTa results not yet available. Using placeholder.")
        # Use placeholder values - replace after running NB13
        accuracy_50[1] = 0  # placeholder
        accuracy_all[1] = 0  # placeholder

    x = np.arange(len(models))
    width = 0.3

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width / 2, accuracy_50, width, label="FPB 50agree", color="#2196F3", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, accuracy_all, width, label="FPB allAgree", color="#FF9800", edgecolor="black", linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
                break

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Architecture Comparison on Held-Out FPB\n(All models: LoRA r=16, identical training data)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(60, 100)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if not has_deberta:
        ax.text(1, 5, "DeBERTa: run NB13", ha="center", fontsize=9, color="gray", style="italic",
                transform=ax.get_xaxis_transform())

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_architecture_comparison.pdf")
    plt.savefig(path)
    plt.savefig(path.replace(".pdf", ".png"))
    print(f"Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--available-only", action="store_true",
                        help="Only generate figures that don't need pending experiment results")
    args = parser.parse_args()

    print("Generating figures for ModernFinBERT paper...")
    print(f"Output directory: {FIGURES_DIR}\n")

    fig1_protocol_gap()
    fig2_databoost_impact()

    if not args.available_only:
        fig3_architecture_comparison()
    else:
        print("Skipping architecture comparison (--available-only, needs DeBERTa results)")

    print(f"\nDone! {len(os.listdir(FIGURES_DIR))} files in {FIGURES_DIR}")
    print("\nNOTE: Confusion matrix figure will be generated by NB15 error analysis notebook.")


if __name__ == "__main__":
    main()
