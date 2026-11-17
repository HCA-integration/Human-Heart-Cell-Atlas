"""Fig 3 k: DVP cardiomyocyte transmural proteome, one volcano per cohort (x = slope, epi <- 0 -> endo;
y = -log10 raw p; grey n.s., orange raw p < 0.05, red BH q < 0.05).

Inputs: the protein trend table of scripts/cardiomyocyte/dvp_trend.py. It is computed by hand from the DVP
protein-group matrices, so workflow/hpc/figures_render.sbatch does not draw this panel.
Outputs: results/figures/Fig3/panel_k.{png,pdf}

Usage: python figures/Fig3/panel_k.py [--table results/cardiomyocyte/fig3k/trend_table.csv]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from hhca import results_dir
from hhca.pl import save_panel

COHORTS = [(1, 6), (2, 8)]  # cohort id, donors (UCLA6: 6 donors; UCLA7: 8 donors)
LABELLED_GENES = ["TP53I3", "ANK1", "RTN4", "CRYM"]
COLOR_NS, COLOR_P, COLOR_Q = "lightgray", "sandybrown", "tab:red"


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--table",
        default=results_dir("cardiomyocyte", "fig3k", "trend_table.csv"),
        help="protein trend table of dvp_trend.py",
    )
    return parser.parse_args()


def draw_volcano(ax, trends, cohort, n_donors):
    """Volcano of the cardiomyocyte protein slopes of one cohort."""
    d = trends[trends.cohort == cohort].drop_duplicates("gene").set_index("gene")
    slope, p, q = d.slope, d.p_value, d.q_value
    neg_log_p = -np.log10(p.clip(lower=1e-300))
    is_q, is_p = q < 0.05, (p < 0.05) & ~(q < 0.05)
    ns = ~(is_q | is_p)
    q_threshold = float(p[is_q].max()) if is_q.any() else np.nan
    ax.scatter(slope[ns], neg_log_p[ns], s=4, c=COLOR_NS, alpha=0.75, linewidths=0, rasterized=True, zorder=2)
    ax.scatter(slope[is_p], neg_log_p[is_p], s=5, c=COLOR_P, alpha=0.9, linewidths=0, zorder=3)
    ax.scatter(slope[is_q], neg_log_p[is_q], s=5, c=COLOR_Q, alpha=0.9, linewidths=0, zorder=4)
    ax.axhline(-np.log10(0.05), color=COLOR_P, ls="--", lw=0.9, zorder=1)
    if np.isfinite(q_threshold):
        ax.axhline(-np.log10(q_threshold), color=COLOR_Q, ls="--", lw=0.9, zorder=1)
    ax.axvline(0, color="0.4", ls=":", lw=0.7, zorder=1)
    for gene in LABELLED_GENES:
        if gene in slope.index:
            ax.scatter(
                slope[gene], neg_log_p[gene], s=12, facecolors="none", edgecolors="black", linewidths=0.6, zorder=6
            )
            dx = -22 if slope[gene] < 0 else 8
            ax.annotate(
                gene,
                (slope[gene], neg_log_p[gene]),
                xytext=(dx, 3),
                textcoords="offset points",
                fontsize=6.5,
                ha="right" if slope[gene] < 0 else "left",
                va="bottom",
                arrowprops={"arrowstyle": "-", "color": "0.4", "lw": 0.5, "shrinkA": 0, "shrinkB": 2},
            )
    ax.set_xlabel("transmural slope (epi ← 0 → endo)", fontsize=7)
    ax.set_ylabel("-log10(p)", fontsize=7)
    ax.set_title(f"Cohort {cohort} (n = {n_donors})", fontsize=8.5, pad=10)
    ax.text(
        0.5,
        1.015,
        f"q<0.05: {int(is_q.sum())}, raw p<0.05: {int((p < 0.05).sum())}, tested: {len(slope)}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.5,
        color="0.35",
    )
    ax.tick_params(labelsize=6.5)
    ax.set_ylim(bottom=-0.15)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", ms=3.5, color=COLOR_NS, label=f"n.s. (n={int(ns.sum())})"),
            Line2D([], [], marker="o", ls="", ms=3.5, color=COLOR_P, label=f"p<0.05 (raw) (n={int(is_p.sum())})"),
            Line2D([], [], marker="o", ls="", ms=3.5, color=COLOR_Q, label=f"q<0.05 (FDR) (n={int(is_q.sum())})"),
            Line2D([], [], ls="--", color=COLOR_P, lw=0.9, label="raw p = 0.05"),
            Line2D([], [], ls="--", color=COLOR_Q, lw=0.9, label="q = 0.05 (FDR)"),
        ],
        loc="upper right",
        fontsize=5.2,
        frameon=False,
        handlelength=1.4,
        labelspacing=0.35,
    )
    print(f"cohort {cohort}: tested {len(slope)}  q<0.05 {int(is_q.sum())}  raw p<0.05 {int((p < 0.05).sum())}")


def main():
    """Draw the two cohorts one above the other."""
    args = parse_args()
    trends = pd.read_csv(args.table)
    trends = trends[trends.cell_type == "CM"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 6.9))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.93, bottom=0.07, hspace=0.42)
    for ax, (cohort, n_donors) in zip(axes, COHORTS):
        draw_volcano(ax, trends, cohort, n_donors)
    print(save_panel(fig, "Fig3", "k"))


if __name__ == "__main__":
    main()
