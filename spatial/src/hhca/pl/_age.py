"""Panels of the age effects on the cytokine-response programmes (Fig 5 m, n), in the colours of the age direction."""

from __future__ import annotations

import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from hhca.pl import style
from hhca.pl._panel import panel

INK = "#222222"
MUTED = "#6E6E6E"
AGE_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10.5,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}
FOOTER = (
    "Donor-level pseudobulk (n=270 adults 20-89), OLS score ~ age + sex; "
    "BH-FDR q; region+assay sensitivity model reported. "
    "Healthy HHCA reference, 280 donors (11 duplicate donor ids pooled). Donors are the unit of replication."
)


def direction_colour(direction, robust=True) -> str:
    """Colour of an age effect: purple for an increase, green for a decrease, light when it is not robust."""
    key = "up_with_age" if str(direction).startswith("up") else "down_with_age"
    return (style.AGE_DIRECTION if robust else style.AGE_DIRECTION_LIGHT)[key]


def _add_footer(fig, y):
    fig.text(0.5, y, FOOTER, ha="center", va="bottom", fontsize=6.6, color=MUTED, wrap=True)


# ---------------------------------------------------------------------------------------------------------------
# Fig 5 m: programmes x immune cell types

CELL = 0.40  # side of one heat-map cell (inch)
# layout (inch): label margin, gap and width of the colour bar; title, tick-label, legend and footer bands
LEFT_LABELS, COLORBAR_GAP, COLORBAR_WIDTH = 1.30, 0.28, 0.17
TOP, X_LABELS, LEGEND, FOOTER_BAND = 0.78, 0.78, 0.58, 0.40
RHO_LIMIT = 0.42


def age_programmes(matrix):
    """Fig 5 m: Spearman rho of the per-donor programme score against age, programmes x immune cell types.

    A star marks the effects that hold in the sensitivity model, a circle those significant after FDR only.

    Parameters
    ----------
    matrix
        ``{"rho", "robust", "fdr"}`` tables of :func:`hhca.tl.programme_matrix`.
    """
    rho, robust, fdr = matrix["rho"], matrix["robust"], matrix["fdr"]
    n_rows, n_cols = rho.shape
    grid_w = n_cols * CELL
    grid_h = n_rows * CELL
    fig_w = LEFT_LABELS + grid_w + COLORBAR_GAP + COLORBAR_WIDTH + 1.35
    fig_h = TOP + grid_h + X_LABELS + LEGEND + FOOTER_BAND
    with panel((fig_w, fig_h), AGE_RC) as fig:
        ax = fig.add_axes(
            [LEFT_LABELS / fig_w, (FOOTER_BAND + LEGEND + X_LABELS) / fig_h, grid_w / fig_w, grid_h / fig_h]
        )
        norm = TwoSlopeNorm(vmin=-RHO_LIMIT, vcenter=0, vmax=RHO_LIMIT)
        im = ax.imshow(rho.values, cmap=style.AGE_DIVERGING, norm=norm, aspect="equal")
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(rho.columns, rotation=40, ha="right", fontsize=8.6)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(rho.index, fontsize=8.8)
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.4)
        ax.tick_params(which="minor", length=0)
        for i in range(n_rows):
            for j in range(n_cols):
                if bool(robust.values[i, j]):
                    ax.text(j, i, "$\\bigstar$", ha="center", va="center", fontsize=8.5, color="white")
                elif bool(fdr.values[i, j]):
                    ax.plot(j, i, marker="o", ms=3.0, mfc="none", mec="#444444", mew=0.9)
        ax.set_title("Cytokine-response programs (huCIRA)", fontsize=12.5, loc="left", pad=18, weight="bold")
        ax.annotate(
            "immune cell types  x  cytokine response programs (FDR-significant programs only)",
            xy=(0, 1.006),
            xycoords="axes fraction",
            fontsize=8.6,
            color=MUTED,
            va="bottom",
        )

        cb_h = grid_h * 0.72
        cb_y0 = FOOTER_BAND + LEGEND + X_LABELS + (grid_h - cb_h) / 2
        cax = fig.add_axes(
            [(LEFT_LABELS + grid_w + COLORBAR_GAP) / fig_w, cb_y0 / fig_h, COLORBAR_WIDTH / fig_w, cb_h / fig_h]
        )
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Spearman ρ  (donor score vs age)", fontsize=8.8)
        cb.ax.tick_params(labelsize=7.5)
        cb.ax.text(
            0.5,
            1.035,
            "up with age",
            transform=cb.ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=style.AGE_DIRECTION["up_with_age"],
        )
        cb.ax.text(
            0.5,
            -0.035,
            "down with age",
            transform=cb.ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            color=style.AGE_DIRECTION["down_with_age"],
        )
        legend_lines = [
            ("$\\bigstar$", "region+assay-robust (q<0.05 and sensitivity p<0.05)"),
            ("$\\odot$", "FDR-significant only (q<0.05)"),
            ("", "blank = not significant"),
        ]
        for k, (glyph, txt) in enumerate(legend_lines):
            y = (FOOTER_BAND + LEGEND - 0.14 - k * 0.16) / fig_h
            fig.text(
                LEFT_LABELS / fig_w,
                y,
                f"{glyph}  {txt}" if glyph else txt,
                fontsize=8.0,
                color=INK,
                ha="left",
                va="center",
            )
        _add_footer(fig, y=0.010)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 5 n: age trajectories of the macrophage programmes

MIN_LABEL_GAP = 0.145  # smallest vertical distance between two curve labels
X_LEADER, X_LABEL = 84.6, 85.2  # end of the leader line and start of the label text


def age_trajectories(significant, trajectories):
    """Fig 5 n: LOESS course over age of the programmes of one cell type that change with age.

    Robust programmes are drawn as solid lines with their bootstrap band, the others dashed; the labels at the
    right edge are pushed apart where curves end close together.

    Parameters
    ----------
    significant
        Programmes to draw with columns ``feature``, ``direction`` and ``robust``
        (:func:`hhca.tl.changing_programmes`).
    trajectories
        Mapping of programme -> fit of :func:`hhca.tl.age_trajectories`.
    """
    with panel((9.4, 6.4), AGE_RC) as fig:
        ax = fig.add_axes([0.085, 0.135, 0.60, 0.645])
        ax.axhline(0, color="#BBBBBB", lw=0.9, zorder=0)
        ends = []
        for row in significant.itertuples():
            fit = trajectories[row.feature]
            grid, center = fit.index.to_numpy(), fit["fit"].to_numpy()
            col = direction_colour(row.direction, robust=bool(row.robust))
            if row.robust:
                ax.fill_between(
                    grid, fit["low"].to_numpy(), fit["high"].to_numpy(), color=col, alpha=0.10, lw=0, zorder=1
                )
                ax.plot(grid, center, color=col, lw=2.4, ls="-", zorder=3, solid_capstyle="round")
            else:
                ax.plot(
                    grid,
                    center,
                    color=col,
                    lw=1.9,
                    ls=(0, (4.5, 2.2)),
                    zorder=4,
                    path_effects=[pe.Stroke(linewidth=3.4, foreground="white", alpha=0.9), pe.Normal()],
                )
            ends.append([center[-1], center[-1], f"{row.feature} response", col, bool(row.robust)])
        ends.sort(key=lambda e: -e[0])
        for i in range(1, len(ends)):
            if ends[i - 1][1] - ends[i][1] < MIN_LABEL_GAP:
                ends[i][1] = ends[i - 1][1] - MIN_LABEL_GAP
        for true_y, label_y, label, col, robust in ends:
            ax.plot(
                [grid[-1] + 0.3, X_LEADER],
                [true_y, label_y],
                color=col,
                lw=0.7,
                ls="-",
                alpha=0.8,
                clip_on=False,
                zorder=2,
            )
            ax.annotate(
                label,
                xy=(X_LABEL, label_y),
                xycoords="data",
                va="center",
                ha="left",
                fontsize=8.6,
                color=col,
                weight="semibold" if robust else "normal",
                annotation_clip=False,
            )
        ax.set_xlim(22, 84)
        ax.set_ylim(-2.1, 2.1)
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Standardized program score (z)\nlow  <—>  high")
        down, up = style.AGE_DIRECTION["down_with_age"], style.AGE_DIRECTION["up_with_age"]
        handles = [
            Line2D([0], [0], color=INK, lw=2.4, ls="-", label="region+assay-robust (q<0.05 and sensitivity p<0.05)"),
            Line2D([0], [0], color=INK, lw=1.9, ls=(0, (4.5, 2.2)), label="FDR-significant only (q<0.05), no CI band"),
            Line2D([0], [0], color=down, lw=2.4, ls="-", label="decreases with age"),
            Line2D([0], [0], color=up, lw=2.4, ls="-", label="increases with age"),
        ]
        ax.legend(
            handles=handles, loc="lower left", fontsize=8.3, handlelength=2.6, labelspacing=0.4, borderpad=0.5, ncol=1
        )
        fig.suptitle("Cytokine (huCIRA)", x=0.085, y=0.965, ha="left", fontsize=15, weight="bold")
        fig.text(
            0.085,
            0.912,
            "macrophage cytokine-response programs significant with age",
            ha="left",
            fontsize=9.5,
            color=MUTED,
        )
        fig.text(
            0.085,
            0.862,
            "Per-donor program scores z-standardized within each program; LOESS fit on a shared age axis. "
            "95% CI band shown for region+assay-robust programs.",
            ha="left",
            fontsize=8.8,
            color=INK,
        )
        _add_footer(fig, y=0.008)
    return fig
