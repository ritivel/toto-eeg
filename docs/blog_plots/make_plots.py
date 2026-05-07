"""Generate clean publication-style plots for the EEG-FM blog post.

Matches a Thinking-Machines / minimal-academic aesthetic: muted palette,
single accent color, generous whitespace, no chartjunk.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent

INK   = "#222222"
GRAY  = "#9aa1a8"
LGRAY = "#dcdfe3"
ACCENT = "#d97757"
NAVY   = "#2d4a5f"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "regular",
    "axes.labelsize": 10,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "savefig.pad_inches": 0.2,
})


def plot_pretrain():
    """Pretraining val_loss: 4-bar build-up baseline -> +JEPA -> +AAMP -> +PARS."""
    labels = [
        "exp26\nbaseline\n(no aux)",
        "+ JEPA\n(exp33)",
        "+ JEPA + AAMP\n(exp34)",
        "+ JEPA + AAMP + PARS\nexp36 TRIPLE",
    ]
    vals = [0.1838, 0.175, 0.1675, 0.1384]
    deltas_pct = [0.0, -4.8, -8.9, -24.7]
    colors = [GRAY, GRAY, GRAY, ACCENT]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, width=0.55, color=colors, edgecolor="none", zorder=3)

    ax.axhline(vals[0], color=LGRAY, lw=0.9, ls="--", zorder=1)

    for i, (b, v, d) in enumerate(zip(bars, vals, deltas_pct)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0035, f"{v:.4f}",
                ha="center", va="bottom", fontsize=10.5, color=INK,
                fontweight="bold" if i == 3 else "regular")
        if i > 0:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.013,
                    f"{d:+.1f}%",
                    ha="center", va="bottom", fontsize=8.5,
                    color=ACCENT if i == 3 else GRAY,
                    fontweight="bold" if i == 3 else "regular")

    ax.text(-0.45, vals[0], "baseline", va="center",
            ha="right", color=GRAY, fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Pretraining val_loss  (pinball, asinh-z)")
    ax.set_ylim(0.10, 0.215)
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.set_title("Pretraining val_loss build-up: each auxiliary adds value",
                 loc="left", pad=14, color=INK)

    ax.yaxis.grid(True, color=LGRAY, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    plt.subplots_adjust(bottom=0.22, top=0.88)
    fig.text(0.5, 0.02,
             "24M params  •  d=384, 12L  •  30k steps  •  HBN-EEG  •  4×H100",
             ha="center", va="bottom", color=GRAY, fontsize=8.5)

    fig.savefig(OUT / "01_pretrain_buildup.png")
    plt.close(fig)
    print(f"  wrote {OUT / '01_pretrain_buildup.png'}")


def plot_downstream():
    """Downstream BAC: exp22 vs exp36 paired bars across 8 datasets, with SOTA markers."""
    datasets = ["arith", "bcic2a", "bcic2020-3", "chbmit",
                "faced", "isruc-sleep", "mdd", "physionet"]
    chance   = [50.0, 25.0, 20.0, 50.0, 11.1, 20.0, 50.0, 25.0]
    exp22    = [50.7, 31.1, 22.0, 54.5, 14.5, 53.2, 84.3, 33.3]
    exp36    = [55.9, 30.0, 22.1, 50.0, 15.8, 55.9, 81.2, 31.5]
    sota_lp  = [74.0, 60.3, 32.0, np.nan, 55.1, 75.8, 98.0, 61.7]

    delta = [a - b for a, b in zip(exp36, exp22)]
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    x = np.arange(len(datasets))
    w = 0.34

    bars22 = ax.bar(x - w/2, exp22, w, label="exp22 baseline (best of 4 strategies)",
                    color=GRAY, edgecolor="none", zorder=3)
    bars36 = ax.bar(x + w/2, exp36, w, label="exp36 TRIPLE (best of ridge/frozen/full-FT)",
                    color=ACCENT, edgecolor="none", zorder=3)

    for i, c in enumerate(chance):
        ax.hlines(c, x[i] - w - 0.05, x[i] + w + 0.05,
                  color=NAVY, lw=1.2, ls=":", zorder=2)
    ax.plot([], [], ls=":", color=NAVY, lw=1.2, label="chance level")

    sota_x, sota_y = [], []
    for i, s in enumerate(sota_lp):
        if not np.isnan(s):
            sota_x.append(x[i])
            sota_y.append(s)
    ax.scatter(sota_x, sota_y, marker="v", s=46, color=INK,
               zorder=5, label="SOTA linear-probe", clip_on=False)

    for i, (b, v22, v36, d) in enumerate(zip(bars36, exp22, exp36, delta)):
        col = "#2e7d32" if d > 0.5 else ("#b71c1c" if d < -0.5 else GRAY)
        top = max(v22, v36) + 1.8
        ax.text(x[i], top, f"{d:+.1f}",
                ha="center", va="bottom", fontsize=8.5, color=col,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylabel("Test balanced accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Downstream eval — exp36 vs exp22 across 8 EEG benchmarks",
                 loc="left", pad=14, color=INK)
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.32),
              frameon=False, fontsize=9)
    ax.yaxis.grid(True, color=LGRAY, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    fig.text(0.5, -0.01,
             "3 seeds, mean BAC.   Δ above each pair (green = exp36 win, red = exp22 win).",
             ha="center", va="top", color=GRAY, fontsize=8.5)

    fig.savefig(OUT / "02_downstream_bars.png")
    plt.close(fig)
    print(f"  wrote {OUT / '02_downstream_bars.png'}")


def plot_sota_gap():
    """Per-dataset horizontal lollipop: chance | exp36 | SOTA-LP."""
    datasets = ["arith", "bcic2a", "bcic2020-3", "chbmit",
                "faced", "isruc-sleep", "mdd", "physionet"]
    chance  = [50.0, 25.0, 20.0, 50.0, 11.1, 20.0, 50.0, 25.0]
    exp36   = [55.9, 30.0, 22.1, 50.0, 15.8, 55.9, 81.2, 31.5]
    sota_lp = [74.0, 60.3, 32.0, np.nan, 55.1, 75.8, 98.0, 61.7]

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    y = np.arange(len(datasets))[::-1]

    for yi, c, e, s in zip(y, chance, exp36, sota_lp):
        if not np.isnan(s):
            ax.hlines(yi, c, s, color=LGRAY, lw=2.2, zorder=1)
        else:
            ax.hlines(yi, c, e, color=LGRAY, lw=2.2, zorder=1)
        ax.scatter(c, yi, s=40, color=NAVY, zorder=4, marker="o", edgecolor="none")
        ax.scatter(e, yi, s=110, color=ACCENT, zorder=5, marker="o", edgecolor="white", linewidth=1.5)
        if not np.isnan(s):
            ax.scatter(s, yi, s=40, color=INK, marker="v", zorder=4)

    for yi, c, e, s in zip(y, chance, exp36, sota_lp):
        ax.text(e, yi + 0.32, f"{e:.1f}", ha="center", va="bottom",
                fontsize=8.5, color=ACCENT, fontweight="bold")
        if not np.isnan(s):
            ax.text(s, yi + 0.32, f"{s:.1f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK)
        ax.text(c, yi - 0.45, f"{c:.0f}", ha="center", va="top",
                fontsize=8.5, color=NAVY)

    ax.set_yticks(y)
    ax.set_yticklabels(datasets, fontsize=10)
    ax.set_xlabel("Test balanced accuracy (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Where exp36 sits — chance to SOTA linear-probe", loc="left", pad=14, color=INK)
    ax.grid(True, axis="x", color=LGRAY, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=NAVY,
                   markersize=8, label="chance"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ACCENT,
                   markersize=11, markeredgecolor="white", label="exp36 TRIPLE (best strategy)"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor=INK,
                   markersize=8, label="SOTA linear-probe"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", ncol=3, bbox_to_anchor=(1.0, -0.15))

    fig.savefig(OUT / "03_sota_gap.png")
    plt.close(fig)
    print(f"  wrote {OUT / '03_sota_gap.png'}")


def plot_strategy_comparison():
    """Per-dataset: ridge vs frozen-SGD vs full-FT for exp36 only."""
    datasets    = ["arith", "bcic2a", "bcic2020-3", "mdd", "physionet"]
    ridge       = [55.9, 30.0, 22.1, 76.9, 31.5]
    frozen_sgd  = [49.8, 28.1, 21.8, 76.3, 29.4]
    full_ft     = [49.7, 28.1, 21.7, 81.2, 29.3]
    chance      = [50.0, 25.0, 20.0, 50.0, 25.0]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(datasets))
    w = 0.26

    ax.bar(x - w, ridge,      w, label="ridge probe (closed-form)",      color=ACCENT, edgecolor="none", zorder=3)
    ax.bar(x    , frozen_sgd, w, label="frozen-SGD linear (30 epochs)", color=GRAY,   edgecolor="none", zorder=3)
    ax.bar(x + w, full_ft,    w, label="full fine-tune (30 epochs)",    color=NAVY,   edgecolor="none", zorder=3)

    for xi, c in zip(x, chance):
        ax.hlines(c, xi - w - 0.05, xi + w + 0.05,
                  color=INK, lw=1.0, ls=":", zorder=2)
    ax.plot([], [], ls=":", color=INK, lw=1.0, label="chance")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylabel("Test balanced accuracy (%)")
    ax.set_title("exp36: ridge probe wins on 4 of 5 fully-evaluated datasets",
                 loc="left", pad=14, color=INK)
    ax.set_ylim(0, 95)
    ax.legend(loc="upper left", ncol=2)
    ax.yaxis.grid(True, color=LGRAY, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    fig.text(0.99, 0.01,
             "chbmit / faced / isruc_sleep killed for slow SGD — ridge-only re-runs.",
             ha="right", va="bottom", color=GRAY, fontsize=8.5)

    fig.savefig(OUT / "04_strategy_comparison.png")
    plt.close(fig)
    print(f"  wrote {OUT / '04_strategy_comparison.png'}")


def main():
    print("Generating blog plots...")
    plot_pretrain()
    plot_downstream()
    plot_sota_gap()
    plot_strategy_comparison()
    print("Done.")


if __name__ == "__main__":
    main()
