"""Draw the two figures used in docs/PROJECT_REPORT.md from the saved reports.

    python scripts/make_figures.py

Reads reports/oof_comparison.csv and reports/test_analysis.json and writes docs/figures/*.png. It needs
matplotlib (in requirements-dev.txt). The figures are plain pictures of numbers already in those files.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def titles(fig, title, subtitle):
    fig.text(0.04, 0.95, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.885, subtitle, fontsize=9.5, color=INK_2, va="top")


def comparison_figure():
    table = pd.read_csv(ROOT / "reports" / "oof_comparison.csv", index_col=0).sort_values("PR-AUC")
    base_rate = float(table["Base Rate"].iloc[0])
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160, facecolor=SURFACE)
    fig.subplots_adjust(left=0.27, right=0.96, top=0.78, bottom=0.12)
    style(ax)
    ax.grid(axis="y", visible=False)
    bars = ax.barh(table.index, table["PR-AUC"], height=0.55, color=BLUE)
    for bar, value in zip(bars, table["PR-AUC"], strict=True):
        ax.text(value + 0.003, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9.5,
                color=INK)
    ax.axvline(base_rate, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(base_rate + 0.002, -0.62, f"random guessing: {base_rate:.3f}", fontsize=8.5, color=INK_2, va="bottom")
    ax.set_xlim(0, 0.27)
    ax.set_ylim(-0.7, len(table) - 0.4)
    ax.tick_params(axis="y", labelsize=10, labelcolor=INK)
    ax.tick_params(axis="x", labelsize=8.5)
    titles(fig, "XGBoost ranks readmissions best, but only moderately",
           "PR-AUC of five models on out-of-fold training predictions (higher is better)")
    fig.savefig(OUT / "model_comparison.png", facecolor=SURFACE)
    plt.close(fig)


def calibration_figure():
    result = json.loads((ROOT / "reports" / "test_analysis.json").read_text(encoding="utf-8"))
    slices = result["calibration"]["slices"]
    predicted = [row["mean_score"] for row in slices]
    observed = [row["observed_rate"] for row in slices]
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=160, facecolor=SURFACE)
    fig.subplots_adjust(left=0.13, right=0.95, top=0.84, bottom=0.12)
    style(ax)
    limit = 0.32
    ax.plot([0, limit], [0, limit], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.plot(predicted, observed, color=BLUE, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.text(0.205, 0.225, "perfect match", fontsize=9, color=INK_2, rotation=33, ha="center")
    ax.annotate(f"highest tenth: predicted {predicted[-1]:.3f},\nobserved {observed[-1]:.3f}",
                xy=(predicted[-1], observed[-1]), xytext=(0.115, 0.285), fontsize=9, color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    ax.annotate(f"lowest tenth: predicted {predicted[0]:.3f},\nobserved {observed[0]:.3f}",
                xy=(predicted[0], observed[0]), xytext=(0.075, 0.03), fontsize=9, color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("Average predicted score in the tenth", fontsize=9.5, color=INK_2)
    ax.set_ylabel("Share actually readmitted within 30 days", fontsize=9.5, color=INK_2)
    titles(fig, "The scores are close to the observed readmission rates",
           f"Held-out test set, {result['n_test_encounters']:,} encounters in ten equal slices by score")
    fig.savefig(OUT / "calibration.png", facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    comparison_figure()
    calibration_figure()
    print(f"Saved the figures to {OUT}")


if __name__ == "__main__":
    main()
