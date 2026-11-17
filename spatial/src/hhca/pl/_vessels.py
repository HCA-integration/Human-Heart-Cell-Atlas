"""Panels of the vessel analysis (Fig 4 f-k)."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator

from hhca.pl import style
from hhca.pl._panel import panel
from hhca.stats import stars
from hhca.tl._markers import COMPARTMENTS

VESSEL_TYPES = ["Arterial(muscular)", "Arteriole", "Capillary/microvessel", "Venous", "Venous(muscular)"]
VESSEL_LABELS = ["Arterial\n(muscular)", "Arteriole", "Capillary /\nmicrovessel", "Venous", "Venous\n(muscular)"]

# ---------------------------------------------------------------------------------------------------------------
# Fig 4 f: arteriole and venule fields on H&E

# the vascular types of the fields, in the Level-3 palette of the spatial panels
FIELD_COLOURS = {
    cell_type: style.SPATIAL[cell_type]
    for cell_type in (
        "Arterial Endothelial cells",
        "Capillary Endothelial cells",
        "Venous Endothelial cells",
        "Pericytes",
        "Vascular Smooth Muscle cells",
    )
}
FIELD_LABELS = {
    "Arterial Endothelial cells": "Arterial EC",
    "Capillary Endothelial cells": "Capillary EC",
    "Venous Endothelial cells": "Venous EC",
    "Pericytes": "Pericytes",
    "Vascular Smooth Muscle cells": "VSMC",
    "Lymphatic Endothelial Cells": "Lymphatic EC",
    "Endocardial Endothelial Cells": "Endocardial EC",
}
FIELD_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "hatch.linewidth": 0.6,
    "hatch.color": "#111111",
}


def _draw_field(ax, field, show_types):
    """One field: H&E, filled vascular bins, hatched unassigned bins and a scale bar."""
    x0, x1, y1, y0 = field["extent"]
    ax.imshow(field["he"], extent=[x0, x1, y1, y0], interpolation="lanczos", zorder=0)
    filled, colours = [], []
    for x, y, level3_type in zip(field["ax"], field["ay"], field["atype"]):
        if level3_type in show_types:
            filled.append(Rectangle((x - 1, y - 1), 2, 2))
            colours.append(FIELD_COLOURS[level3_type])
    ax.add_collection(PatchCollection(filled, facecolor=colours, edgecolor="none", alpha=0.85, zorder=2))
    hatched = [Rectangle((x - 1, y - 1), 2, 2) for x, y in zip(field["ux"], field["uy"])]
    ax.add_collection(
        PatchCollection(hatched, facecolor="none", edgecolor="#111111", linewidth=0, hatch="////", zorder=3)
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar_um = 20.0
    bar_x, bar_y = x0 + 0.06 * (x1 - x0), y1 - 0.06 * (y1 - y0)
    ax.plot([bar_x, bar_x + bar_um], [bar_y, bar_y], color="black", lw=3, solid_capstyle="butt", zorder=5)
    ax.text(
        bar_x + bar_um / 2,
        bar_y - 0.025 * (y1 - y0),
        "20 µm",
        ha="center",
        va="bottom",
        fontsize=8,
        zorder=5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1},
    )


def _field_legend(fig, show_types, y=0.02):
    """Cell-type and hatch key below the fields."""
    handles = [
        Patch(facecolor=FIELD_COLOURS[level3_type], edgecolor="none", alpha=0.85, label=FIELD_LABELS[level3_type])
        for level3_type in show_types
    ]
    handles.append(
        Patch(facecolor="white", edgecolor="#111111", hatch="////", linewidth=0.5, label="no cell-type assignment")
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1.2,
        columnspacing=1.2,
    )


def vessel_fields(fields, titles, show_types, n_cols=None):
    """Fig 4 f: fields of the H&E with the vascular 2 um bins filled and the unassigned bins hatched.

    Parameters
    ----------
    fields
        Field extracts of :func:`hhca.tl.extract_field`.
    titles
        One title per field.
    show_types
        Level-3 types whose bins are filled.
    n_cols
        Draw a contact sheet with this many columns instead of the fields side by side.
    """
    n_fields = len(fields)
    if n_cols is None:
        with panel((3.3 * n_fields, 3.75), FIELD_RC) as fig:
            axs = fig.subplots(1, n_fields, squeeze=False)
            for ax, field, title in zip(axs[0], fields, titles):
                _draw_field(ax, field, show_types)
                ax.set_title(title, fontsize=10, pad=4)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.17, wspace=0.04)
            _field_legend(fig, show_types)
        return fig
    n_rows = int(np.ceil(n_fields / n_cols))
    height = 3.25 * n_rows + 0.6
    with panel((3.0 * n_cols, height), FIELD_RC) as fig:
        axs = fig.subplots(n_rows, n_cols, squeeze=False)
        for ax, field, title in zip(axs.ravel(), fields, titles):
            _draw_field(ax, field, show_types)
            ax.set_title(title, fontsize=7.5)
        for ax in axs.ravel()[n_fields:]:
            ax.axis("off")
        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.6 / height + 0.02, hspace=0.25, wspace=0.05)
        _field_legend(fig, show_types, y=0.0)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 4 g: composition along calibre

# (label in the composition table, colour) per subtype; Okabe-Ito colours
SUBTYPE_COLOURS = list(style.VASCULAR_SUBTYPE.items())
CALIBRE_TICKS = [10, 15, 20, 30, 50, 80, 120, 200]
COMPOSITION_RC = {"font.size": 11, "font.family": "DejaVu Sans", "pdf.fonttype": 42}


def composition_along_calibre(bins, edges):
    """Fig 4 g: vascular composition of vessel structures along calibre.

    Mean share of each subtype per calibre bin with its 95 % confidence band, on a reversed log calibre axis
    (larger vessels left).

    Parameters
    ----------
    bins
        Composition per diameter bin (:func:`hhca.tl.composition_by_calibre` with ``name="diam_um"``); only
        the bins with ``enough`` structures are drawn.
    edges
        Bin edges, which set the limits of the calibre axis.
    """
    bins = bins[bins.enough].sort_values("median_diam_um")
    with panel((8.84, 5.3), COMPOSITION_RC) as fig:
        ax = fig.add_axes([0.0783, 0.20, 0.9104, 0.78])
        x = bins.median_diam_um.values
        for label, colour in SUBTYPE_COLOURS:
            y, ci = bins[f"{label}__pct"].values, bins[f"{label}__ci95_pct"].values
            ax.fill_between(x, y - ci, y + ci, color=colour, alpha=0.18, linewidth=0)
            ax.plot(x, y, color=colour, lw=2.3, marker="o", ms=3.6)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlim(edges[-1] * 1.02, edges[0] * 0.98)
        ax.xaxis.set_major_locator(FixedLocator(CALIBRE_TICKS))
        ax.xaxis.set_major_formatter(FixedFormatter([str(tick) for tick in CALIBRE_TICKS]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Vessel caliber: equivalent diameter (µm), log scale")
        ax.set_ylabel("Member bins assigned to subtype (mean %)")
        ax.set_ylim(-2, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.35)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        # arrow below the axis from larger to smaller vessels
        ax.annotate(
            "",
            xy=(0.0, -0.135),
            xytext=(1.0, -0.135),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "color": "0.35", "lw": 1.4},
        )
        ax.text(0.0, -0.175, "larger vessels", transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color="0.35")
        ax.text(
            1.0, -0.175, "smaller vessels", transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color="0.35"
        )

        handles = [Line2D([0], [0], color=colour, lw=2.6, marker="o", ms=4.5) for _, colour in SUBTYPE_COLOURS]
        handles.append(Patch(facecolor="#888888", alpha=0.25))
        ax.legend(
            handles,
            [label for label, _ in SUBTYPE_COLOURS] + ["95% CI of the mean"],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=False,
            fontsize=11.5,
            columnspacing=2.4,
            handletextpad=0.6,
            handlelength=2.2,
        )
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 4 h: wall markers per compartment

INK, MUTED = "#1a1a1a", "#6b6b6b"
GROUP_COLOURS = {
    "VSMC": "#393b79",
    "Arterial EC": "#7f9fce",
    "Pericyte": "#b5b53d",
    "Capillary EC": "#2ca02c",
    "Venous EC": "#5254a3",
    "Lymphatic EC": "#8c564b",
    "Endocardial": "#d62728",
    "Connexins": "#e69f00",
    "Tight junction": "#cc79a7",
    "Caveolins": "#148f77",
}
N_CALIBRE = 7  # columns along the arterio-venous axis; a grey line separates the specialised endothelia
# (gene group, genes, key of GROUP_COLOURS)
MARKER_GROUPS = [
    ("Vascular smooth muscle", ["MYH11", "ACTA2"], "VSMC"),
    ("Arterial EC", ["SEMA3G"], "Arterial EC"),
    ("Pericyte", ["RGS5", "ABCC9"], "Pericyte"),
    ("Capillary EC", ["CA4", "RGCC", "GPIHBP1"], "Capillary EC"),
    ("Venous EC", ["ACKR1", "PLVAP"], "Venous EC"),
    ("Lymphatic EC", ["PROX1", "LYVE1"], "Lymphatic EC"),
    ("Endocardial EC", ["NPR3", "SMOC1"], "Endocardial"),
    ("Connexins", ["GJA5", "GJA4", "GJA1", "GJC1"], "Connexins"),
    (
        "Endothelial junction",
        ["CLDN5", "CDH5", "TJP1", "OCLN", "F11R", "ESAM", "JAM2", "CGNL1", "CLDN11"],
        "Tight junction",
    ),
    ("Caveolins", ["CAV1", "CAV2"], "Caveolins"),
]
MARKER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
    "text.color": INK,
}


def _row_zscores(pooled):
    """Genes of the panel that are in the pooled table, and their row z-scores across the compartments."""
    logcp = pooled.pivot(index="gene", columns="type", values="logcp").reindex(
        columns=[compartment for compartment, _ in COMPARTMENTS]
    )
    genes = [gene for _, group_genes, _ in MARKER_GROUPS for gene in group_genes if gene in logcp.index]
    logcp = logcp.reindex(index=genes)
    z = (
        logcp.sub(logcp.mean(axis=1), axis=0)
        .div(logcp.std(axis=1, ddof=0).replace(0, np.nan), axis=0)
        .fillna(0.0)
        .values
    )
    return genes, z


def wall_markers(pooled):
    """Fig 4 h: vascular marker genes, connexins, junction genes and caveolins across vessel-wall compartments.

    Heatmap of the row z-scores of log1p CP10k, with gene-group colour bars and their key.

    Parameters
    ----------
    pooled
        Compartment x gene table of :func:`hhca.tl.pool_markers`.
    """
    genes, z = _row_zscores(pooled)
    vmax = round(min(2.5, float(np.nanmax(np.abs(z)))) + 0.05, 1)
    norm, cmap = mcolors.Normalize(-vmax, vmax), plt.get_cmap("RdBu_r")
    n_rows, n_cols = len(genes), len(COMPARTMENTS)
    with panel((8.0, 11.1), MARKER_RC) as fig:
        ax = fig.add_axes([0.098, 0.108, 0.606, 0.889])
        ax.imshow(
            z,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=[-0.5, n_cols - 0.5, n_rows - 0.5, -0.5],
            interpolation="nearest",
        )
        for x in np.arange(-0.5, n_cols, 1):
            ax.axvline(x, color="white", lw=0.8)
        for y in np.arange(-0.5, n_rows, 1):
            ax.axhline(y, color="white", lw=0.8)
        ax.axvline(N_CALIBRE - 0.5, color="#9a9a9a", lw=1.6)
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(
            [label for _, label in COMPARTMENTS],
            fontsize=7.5,
            rotation=40,
            ha="right",
            rotation_mode="anchor",
            style="italic",
        )
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(genes, fontsize=7.5, style="italic")
        ax.tick_params(length=0, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        first_row = 0
        for _, group_genes, colour_key in MARKER_GROUPS:  # gene-group colour bars left of the heatmap
            present = [gene for gene in group_genes if gene in genes]
            ax.add_patch(
                Rectangle(
                    (-0.5 - 1.42, first_row - 0.5 + 0.08),
                    0.2,
                    len(present) - 0.16,
                    color=GROUP_COLOURS[colour_key],
                    clip_on=False,
                )
            )
            first_row += len(present)

        # colourbar, top-aligned with the heatmap
        colourbar_ax = fig.add_axes([0.77, 0.581, 0.033, 0.399])
        colourbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colourbar_ax, extend="both")
        colourbar.set_label("row z-score\n(mean log$_1$$_p$ CP10k)", fontsize=10.5, color=INK, labelpad=8)
        colourbar.ax.tick_params(labelsize=10.5, color="#bbbbbb", labelcolor=MUTED)
        colourbar.outline.set_edgecolor("#cccccc")
        colourbar.outline.set_linewidth(0.6)

        # key of the gene-group colours below the colourbar
        key_ax = fig.add_axes([0, 0, 1, 1])
        key_ax.axis("off")
        for j, (label, _, colour_key) in enumerate(MARKER_GROUPS):
            y = 0.536 - j * 0.0381
            key_ax.add_patch(
                Rectangle((0.737, y - 0.010), 0.028, 0.020, color=GROUP_COLOURS[colour_key], transform=key_ax.transAxes)
            )
            key_ax.text(0.80, y, label, transform=key_ax.transAxes, fontsize=9, color=MUTED, va="center", ha="left")
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 4 i: RAAS row z-scores

RAAS_HEATMAP_GENES = ["ACE", "AGTR1", "ACE2", "ENPEP"]
RAAS_HEATMAP_RC = {"font.family": "sans-serif", "pdf.fonttype": 42}


def raas_heatmap(emm):
    """Fig 4 i: row z-scores of the richness-adjusted marginal means of RAAS genes across vessel types.

    The vessel type with the highest mean of each gene is outlined; an artery-vein colour bar runs below.

    Parameters
    ----------
    emm
        Adjusted means per gene and vessel type (``emm`` of :func:`hhca.tl.fit_raas`).
    """
    genes = RAAS_HEATMAP_GENES
    means = emm.pivot(index="gene", columns="vessel_type", values="emm_log1p_cp10k").loc[genes, VESSEL_TYPES]
    means = means.to_numpy(float)
    z = (means - means.mean(1, keepdims=True)) / means.std(1, ddof=0, keepdims=True)
    with panel((1934 / 300, 1687 / 300), RAAS_HEATMAP_RC) as fig:
        ax = fig.add_axes([0.110, 0.144, 0.708, 0.791])
        image = ax.imshow(z, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
        for i in range(len(genes)):
            ax.add_patch(
                plt.Rectangle(
                    (int(np.argmax(z[i])) - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#5A1A1A",
                    linewidth=2.2,
                    zorder=5,
                )
            )
        ax.set_xticks(range(len(VESSEL_TYPES)), VESSEL_LABELS, fontsize=10)
        ax.set_yticks(range(len(genes)), genes, fontsize=12)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title("Richness-adjusted RAAS gene expression across vessel type", fontsize=12, pad=8)

        colourbar_ax = fig.add_axes([0.838, 0.144, 0.034, 0.791])
        colourbar = fig.colorbar(image, cax=colourbar_ax)
        colourbar.set_label("row z-score", fontsize=11)
        colourbar.outline.set_visible(False)
        colourbar.ax.tick_params(labelsize=12)

        gradient_ax = fig.add_axes([0.110, 0.030, 0.708, 0.045])
        artery_vein = LinearSegmentedColormap.from_list("av", ["#8C2D24", "#c0564a", "#9b9aa3", "#5a74a8", "#2D4A8C"])
        gradient_ax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=artery_vein)
        gradient_ax.set_axis_off()
        gradient_ax.text(
            -0.02,
            0.5,
            "artery",
            transform=gradient_ax.transAxes,
            ha="right",
            va="center",
            fontsize=8.5,
            color="#8C2D24",
        )
        gradient_ax.text(
            1.02, 0.5, "vein", transform=gradient_ax.transAxes, ha="left", va="center", fontsize=8.5, color="#2D4A8C"
        )
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 4 j: RAAS marginal means

# gene -> (colour, line style, marker)
RAAS_STYLE = {"ACE": ("#C05050", "-", "o"), "ACE2": ("#4C9A55", "-", "o"), "AGTR1": ("#6858A0", "--", "D")}
GREY = "#7F7F7F"
MEAN_AND_INTERVAL = ("adj_cp10k", "adj_cp10k_lo", "adj_cp10k_hi")
RAAS_MEANS_RC = {"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42}

# artery -> capillary bed -> vein sketch across the top of the axes
RED, BLUE = np.array(to_rgb("#8B1E1E")), np.array(to_rgb("#3F5FB0"))
SKETCH_Y = 0.49
SKETCH_LINE = {"zorder": 5, "clip_on": False, "solid_capstyle": "round"}


def _draw_means(ax, ax_right, emm):
    """AGT band on the right axis and the ACE / ACE2 / AGTR1 means with intervals on the left axis."""
    tables = {
        column: emm.pivot(index="gene", columns="vessel_type", values=column)[VESSEL_TYPES]
        for column in MEAN_AND_INTERVAL
    }
    x = np.arange(len(VESSEL_TYPES))
    mean, lo, hi = (tables[column].loc["AGT"].values for column in MEAN_AND_INTERVAL)
    ax_right.fill_between(x, lo, hi, color=GREY, alpha=0.28, linewidth=0, zorder=1)
    ax_right.plot(x, mean, color=GREY, lw=1.3, zorder=2)
    for gene, (colour, line_style, marker) in RAAS_STYLE.items():
        mean, lo, hi = (tables[column].loc[gene].values for column in MEAN_AND_INTERVAL)
        ax.errorbar(
            x,
            mean,
            yerr=[mean - lo, hi - mean],
            color=colour,
            ls=line_style,
            lw=1.2,
            marker=marker,
            ms=5,
            capsize=2.5,
            elinewidth=1.0,
            zorder=4,
        )

    ax.set_xlim(-0.35, len(VESSEL_TYPES) - 0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(VESSEL_LABELS, fontsize=7.5)
    ax.set_ylim(0, 0.52)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.tick_params(axis="both", labelsize=7.5, length=3, width=0.7)
    ax.set_ylabel("adjusted CP10k (reference richness)", fontsize=7.5)
    ax_right.set_ylim(0, 1.27)
    ax_right.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_right.tick_params(axis="y", colors=GREY, labelsize=7.5, length=3, width=0.7)
    ax_right.set_ylabel("AGT CP10k", color=GREY, fontsize=7.5)
    for axes in (ax, ax_right):
        axes.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["right"].set_color("#222222")


def _draw_branch(ax, x, y, dx, dy, colour, lw, depth):
    """One tapering side branch: a curved segment that forks twice."""
    ax.plot([x, x + 0.55 * dx, x + dx], [y, y + 0.35 * dy, y + dy], color=colour, lw=lw, **SKETCH_LINE)
    if depth > 0:
        for fork in (0.65, -0.35):
            _draw_branch(ax, x + dx, y + dy, 0.55 * dx, 0.55 * dy + fork * abs(dy), colour, lw * 0.6, depth - 1)


def _draw_vessel_sketch(ax):
    """Tapering artery and vein with side branches, joined by a capillary bed, coloured red to blue."""
    for x0, x1, t0, t1, direction in ((-0.3, 1.72, 0.0, 0.3, 1), (2.28, 4.3, 0.7, 1.0, -1)):
        xs = np.linspace(x0, x1, 40)
        for i in range(len(xs) - 1):
            t = t0 + (t1 - t0) * i / 38
            taper = i / 38 if direction > 0 else 1 - i / 38
            ax.plot(
                xs[i : i + 2],
                [SKETCH_Y, SKETCH_Y],
                color=RED * (1 - t) + BLUE * t,
                lw=3.2 - 1.2 * taper,
                **SKETCH_LINE,
            )
        for j, branch_x in enumerate(np.linspace(x0 + 0.25, x1 - 0.2, 4)):
            t = t0 + (t1 - t0) * (branch_x - x0) / (x1 - x0)
            colour = RED * (1 - t) + BLUE * t
            for sign in (1, -1):
                _draw_branch(ax, branch_x, SKETCH_Y, direction * 0.13, sign * (0.028 - 0.004 * j), colour, 1.3, 1)
    theta = np.linspace(0, 2 * np.pi, 120)
    for k in range(14):  # capillary bed: a bulb of thin interlaced loops
        angle, radius = 2 * np.pi * k / 14, 0.045 + 0.02 * (k % 3)
        centre_x, centre_y = 2.0 + 0.10 * np.cos(angle), SKETCH_Y + 0.014 * np.sin(angle)
        t = float(np.clip((centre_x - 1.8) / 0.4, 0, 1))
        ax.plot(
            centre_x + (radius + 0.03) * np.cos(theta),
            centre_y + 0.30 * radius * np.sin(theta),
            color=RED * (1 - t) + BLUE * t,
            lw=0.55,
            alpha=0.9,
            zorder=5,
            clip_on=False,
        )


def raas_means(emm):
    """Fig 4 j: richness-adjusted means of RAAS genes across vessel types on the CP10k scale.

    ACE, ACE2 and AGTR1 with their 95 % interval on the left axis, AGT as a grey band on the right axis, and
    an artery-capillary-vein sketch along the top.

    Parameters
    ----------
    emm
        Adjusted means per gene and vessel type (``emm`` of :func:`hhca.tl.fit_raas`).
    """
    with panel((6.0, 4.53), RAAS_MEANS_RC) as fig:
        ax = fig.add_axes([0.098, 0.156, 0.745, 0.727])
        ax_right = ax.twinx()
        _draw_means(ax, ax_right, emm)
        _draw_vessel_sketch(ax)
        handles = [
            Line2D([0], [0], color=colour, ls=line_style, marker=marker, ms=5.5, lw=1.3)
            for colour, line_style, marker in RAAS_STYLE.values()
        ]
        handles.append(Patch(facecolor=GREY, alpha=0.5))
        ax.legend(
            handles,
            list(RAAS_STYLE) + ["AGT (right axis)"],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            frameon=False,
            fontsize=7.5,
            handlelength=2.2,
            columnspacing=1.6,
            handletextpad=0.6,
        )
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 4 k: drug-target classes by calibre

DRUG_INK = "#2B2B2B"
MURAL_COLOUR, ENDOTHELIAL_COLOUR = "#8161bc", "#4292a6"  # VSMC and arterial EC
DRUG_SHORT = {
    "Endothelin receptor antagonists": "Endothelin RAs",
    "Angiotensin receptor blockers": "ARBs",
    "Calcium-channel blockers": "Ca-channel blockers",
    "Direct vasodilators (K-ATP)": "Direct vasodil. (K-ATP)",
    "Prostacyclin pathway": "Prostacyclin",
}
DRUG_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.6,
    "xtick.color": DRUG_INK,
    "ytick.color": DRUG_INK,
    "text.color": DRUG_INK,
}


def drug_calibre(summary):
    """Fig 4 k: calibre selectivity of vascular drug-target classes (large minus small vessels).

    Mural targets in the VSMC colour, endothelial targets in the arterial-EC colour; stars from the
    BH-adjusted p value.

    Parameters
    ----------
    summary
        One row per drug class (:func:`hhca.tl.calibre_selectivity`).
    """
    summary = summary.sort_values("coef_large_minus_small_adj", ascending=False)
    y = np.arange(len(summary))[::-1]
    colours = [MURAL_COLOUR if compartment == "mural" else ENDOTHELIAL_COLOUR for compartment in summary.compartment]
    coefficients = summary.coef_large_minus_small_adj.values
    with panel((6.16, 4.14), DRUG_RC) as fig:
        ax = fig.add_axes([0.327, 0.16, 0.66, 0.80])
        ax.hlines(y, 0, coefficients, color=colours, lw=2.4, zorder=2)
        ax.scatter(coefficients, y, c=colours, s=70, zorder=4)
        ax.axvline(0, color="#444444", lw=0.8, zorder=1)
        for y_class, coefficient, p_adj in zip(y, coefficients, summary.p_adj_LvsS.values):
            ax.text(
                coefficient + (0.018 if coefficient >= 0 else -0.018),
                y_class,
                stars(p_adj),
                va="center",
                ha="left" if coefficient >= 0 else "right",
                fontsize=10,
            )
        ax.set_yticks(y)
        ax.set_yticklabels([DRUG_SHORT.get(drug_class, drug_class) for drug_class in summary.drug_class], fontsize=11.5)
        for tick_label, colour in zip(ax.get_yticklabels(), colours):
            tick_label.set_color(colour)
        ax.set_ylim(-0.6, len(summary) - 0.4)
        ax.set_xlim(-0.45, 0.52)
        ax.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])
        ax.tick_params(axis="x", labelsize=10.5, length=2.5, width=0.6)
        ax.tick_params(axis="y", length=2.5, width=0.6)
        ax.set_xlabel("large - small (richness adjusted drug target score)", fontsize=13, labelpad=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", ls="", mfc=MURAL_COLOUR, mec="none", ms=8, label="mural target"),
                Line2D(
                    [0], [0], marker="o", ls="", mfc=ENDOTHELIAL_COLOUR, mec="none", ms=8, label="endothelial target"
                ),
            ],
            loc="lower right",
            bbox_to_anchor=(1.0, 0.0),
            fontsize=10.5,
            frameon=False,
            handletextpad=0.4,
            labelspacing=0.3,
        )
    return fig
