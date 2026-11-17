"""Panels of the cardiomyocyte analysis (Fig 1 j, Fig 3 d-l, ED6 d-f)."""

from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgb, to_rgba
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle
from scipy import stats

from hhca.pl import style
from hhca.pl._panel import panel
from hhca.stats import stars
from hhca.tl._bins import CARDIOMYOCYTES
from hhca.tl._transmural import consensus_profile

ARIAL_RC = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "pdf.fonttype": 42}
INK = "#111111"

# ---------------------------------------------------------------------------------------------------------------
# Fig 1 j: H&E fields with the 2 um bins of one cell type


def _draw_celltype_field(ax, field, tile, palette):
    """H&E crop with the non-cardiomyocyte bins on top and the target type labelled."""
    x0, x1, y0, y1 = field["ext"]
    ax.imshow(field["he"], extent=[x0, x1, y1, y0], interpolation="lanczos")
    labels = field["cats"][field["code"]]
    for target in (False, True):
        m = (~np.isin(labels, CARDIOMYOCYTES)) & ((labels == tile["target"]) == target)
        if m.any():
            ax.scatter(
                field["x"][m],
                field["y"][m],
                marker="s",
                s=9 if target else 6,
                linewidths=0,
                c=[to_rgba(palette.get(label, "#cccccc"), 0.9 if target else 0.45) for label in labels[m]],
            )
    is_target = labels == tile["target"]
    if is_target.any():
        tx, ty = np.median(field["x"][is_target]), np.median(field["y"][is_target])
    else:
        tx, ty = (x0 + x1) / 2, (y0 + y1) / 2
    ax.annotate(
        tile["label"].replace("\\n", "\n"),
        xy=(tx, ty),
        xytext=(x0 + 0.06 * (x1 - x0), y0 + 0.82 * (y1 - y0)),
        color="black",
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
        linespacing=1.25,
        arrowprops={"arrowstyle": "-", "color": "black", "lw": 1.1},
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def celltype_fields(fields, tiles, palette=None):
    """Fig 1 j: fields of the H&E with the 2 um bins on top, one field per cell type.

    Cardiomyocyte bins are not drawn, the bins of other Level-3 types are semi-transparent and those of the
    target type opaque; a line points from the label to the median position of the target bins.

    Parameters
    ----------
    fields
        Mapping of field name -> field of :func:`hhca.tl.he_window`.
    tiles
        Mapping of field name -> ``{"label": ..., "target": Level-3 type}``.
    palette
        Level-3 type -> colour (default: the palette of the spatial panels).
    """
    palette = palette or style.SPATIAL
    with panel((9.0, 2.55), {}) as fig:
        axes = fig.subplots(1, len(fields))
        fig.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.002, wspace=0.012)
        for ax, (name, field) in zip(axes, fields.items()):
            _draw_celltype_field(ax, field, tiles[name], palette)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 3 d: stress score across the wall

STRESS_TITLES = {"epi": "Epicardium\nNear", "myo": "Mid\nMyocardium", "endo": "Endocardium\nNear"}
STRESS_SCALE_BAR_UM = 200.0


def stress_fields(grids, fields, strip, limits):
    """Fig 3 d: the stress score of ventricular cardiomyocytes in one field per section of the wall.

    Parameters
    ----------
    grids
        Mapping of layer (``epi``, ``myo``, ``endo``) -> score on the display grid (:func:`hhca.tl.field_grid`).
    fields
        Mapping of layer -> [xlo, xhi, ylo, yhi] of the field in um.
    strip
        Part of each field that is shown: ``{"w": width, "y0": ..., "y1": ...}`` in um from its lower corner.
    limits
        (vmin, vmax) of the colour scale.
    """
    norm = Normalize(*limits)
    cmap = plt.get_cmap(style.SEQUENTIAL).copy()
    cmap.set_bad("white")
    with panel((7.8, 5.9), ARIAL_RC) as fig:
        axes = []
        for layer, left in zip(grids, [0.023, 0.319, 0.614]):
            ax = fig.add_axes([left, 0.029, 0.264, 0.780])
            xlo, xhi, ylo, yhi = fields[layer]
            ax.imshow(
                grids[layer],
                cmap=cmap,
                norm=norm,
                origin="upper",
                extent=[xlo, xhi, yhi, ylo],
                interpolation="nearest",
                rasterized=True,
            )
            ax.set_xlim(xlo, xlo + strip["w"])
            ax.set_ylim(ylo + strip["y1"], ylo + strip["y0"])
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(STRESS_TITLES[layer], fontsize=17, pad=14, linespacing=1.15)
            axes.append(ax)
        bar = STRESS_SCALE_BAR_UM / strip["w"]
        axes[0].plot(
            [0.01, 0.01 + bar],
            [0.104, 0.104],
            transform=axes[0].transAxes,
            color="k",
            lw=5.5,
            solid_capstyle="butt",
            clip_on=False,
        )
        axes[0].text(0.01 + bar / 2, 0.15, "200 µm", transform=axes[0].transAxes, ha="center", va="bottom", fontsize=13)
        cax = fig.add_axes([0.891, 0.134, 0.019, 0.546])
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        colorbar = fig.colorbar(mappable, cax=cax)
        colorbar.ax.tick_params(labelsize=8, length=3, width=0.6)
        colorbar.outline.set_linewidth(0.6)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 3 e: bins expressing NPPA and NPPB

EXPRESSING_COLOUR = "#B8102E"


def expressing_bins(field, genes, palette=None):
    """Fig 3 e: the 2 um bins that express a gene, over the Level-3 map of a field (one map per gene).

    Parameters
    ----------
    field
        Bins of the field with the counts of `genes` (:func:`hhca.tl.extract_window`).
    genes
        Genes to show.
    palette
        Level-3 type -> colour (default: the palette of the spatial panels).
    """
    palette = palette or style.SPATIAL
    x, y = field["x"], field["y"]
    labels = field["cats"][field["code"]]
    x0, x1, y0, y1 = field["ext"]
    colors = np.array([to_rgb(palette.get(label, "#cccccc")) for label in labels])
    with panel((4.9, 2.95), ARIAL_RC) as fig:
        for i, gene in enumerate(genes):
            ax = fig.add_axes([0.03 + i * 0.49, 0.02, 0.455, 0.78])
            ax.scatter(x, y, c=colors, s=0.45, marker="s", linewidths=0, rasterized=True)
            counts = field[gene]
            expressed = counts > 0
            if expressed.any():
                alpha = np.clip(0.35 + 0.65 * counts[expressed] / max(np.percentile(counts[expressed], 99), 1), 0, 1)
            else:
                alpha = []
            ax.scatter(
                x[expressed],
                y[expressed],
                c=[to_rgb(EXPRESSING_COLOUR)],
                alpha=alpha,
                s=2.2,
                marker="s",
                linewidths=0,
                rasterized=True,
            )
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(gene, fontsize=15, fontstyle="italic", pad=6)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 3 f-h: distance of gene-high cardiomyocyte bins to their neighbours

NICHE_LABELS = {
    "dist_fibroblast_um": "fibroblast-rich",
    "dist_endocardial_um": "endocardium",
    "dist_vessel_um": "vessel-rich",
}
NICHE_RC = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def niche_distance(pool, metric, sections, gene="NPPA"):
    """Fig 3 f-h: median distance of gene-high and the other cardiomyocyte bins to a neighbour, per section.

    Stars give the adjusted one-sided Mann-Whitney q value.

    Parameters
    ----------
    pool
        Test table of :func:`hhca.tl.niche_tests`.
    metric
        ``dist_fibroblast_um`` (f), ``dist_vessel_um`` (g) or ``dist_endocardial_um`` (h).
    sections
        Mapping of section -> label on the axis, in the order of the bars.
    """
    rows = pool[(pool.metric == metric) & (pool.gene == gene)].set_index("slide").reindex(list(sections))
    with panel((0.55 * len(sections) + 1.2, 4.2), NICHE_RC) as fig:
        ax = fig.subplots()
        x = np.arange(len(sections))
        w = 0.38
        ax.bar(x - w / 2, rows.high_median.values, w, color=style.HIGH_LOW["high"], label=f"{gene}-high CM")
        ax.bar(x + w / 2, rows.low_median.values, w, color=style.HIGH_LOW["low"], label=f"{gene}-low CM")
        for xi, (_, r) in zip(x, rows.iterrows()):
            if np.isfinite(r.mw_q):
                ax.text(xi, max(r.high_median, r.low_median) * 1.02, stars(r.mw_q), ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(list(sections.values()), rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("median distance (µm)")
        ax.set_title(f"Distance to nearest {NICHE_LABELS[metric]}", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 3 i: transmural gene modules

MODULE_COLOURS = [
    "#3b6ea5",
    "#5aa897",
    "#c0762f",
    "#8c5a96",
    "#7a8b3b",
    "#b0506a",
    "#4f7f6f",
    "#a98b3a",
    "#6b6b8a",
    "#9c6b4a",
]
MODULE_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _epi_endo_arrow(ax, y=-0.135):
    """Arrow below the axis pointing from epi to endo."""
    arrow = FancyArrowPatch(
        (0.0, y),
        (1.0, y),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=20,
        lw=2.0,
        color="#333333",
        clip_on=False,
    )
    ax.add_patch(arrow)
    for position, text, align in ((0.0, "epi", "left"), (1.0, "endo", "right")):
        ax.text(
            position,
            y - 0.05,
            text,
            transform=ax.transAxes,
            ha=align,
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#333333",
        )


def _draw_module(ax, genes, title, profiles, sigma, n_per):
    """Consensus curve and donor spread of every gene of a module."""
    n_bins = 3 * n_per
    wall_bin = np.arange(n_bins)
    handles = []
    for i, gene in enumerate(genes):
        color = MODULE_COLOURS[i % len(MODULE_COLOURS)]
        curve, band = consensus_profile(profiles, gene, sigma)
        m = np.isfinite(curve)
        (line,) = ax.plot(wall_bin[m], curve[m], color=color, lw=2.2, label=gene)
        handles.append(line)
        mb = np.isfinite(band) & m
        ax.fill_between(wall_bin[mb], (curve - band)[mb], (curve + band)[mb], color=color, alpha=0.15, lw=0)
    for edge in (n_per, 2 * n_per):
        ax.axvline(edge, color="#cccccc", lw=0.9, ls="--")
    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("z-scored smoothed expression")
    ax.set_xticks([n_per * 0.5, n_per * 1.5, n_per * 2.5])
    ax.set_xticklabels(["epi", "myo", "endo"], fontsize=10)
    ax.set_xlim(0, n_bins - 1)
    ax.margins(x=0.01)
    _epi_endo_arrow(ax)
    ax.legend(handles=handles, frameon=False, fontsize=8.5, ncol=2, loc="best")


def transmural_modules(profiles, modules, sigma=1.3):
    """Fig 3 i: expression across the wall of the genes that rise towards the endocardium and the epicardium.

    Per gene, the wall profile of each donor is z-scored; the line is the smoothed mean over the donors and the
    band their spread.

    Parameters
    ----------
    profiles
        Per-donor wall profiles of the module genes (:func:`hhca.tl.module_profiles`).
    modules
        ``{"endo": genes, "epi": genes}``.
    sigma
        Width of the Gaussian smoothing, in wall bins.
    """
    n_per = next(iter(profiles.values())).shape[1] // 3
    with panel((13.0, 6.4), MODULE_RC) as fig:
        ax = fig.subplots(1, 2, sharex=True)
        _draw_module(ax[0], modules["endo"], "endo-rising module", profiles, sigma, n_per)
        _draw_module(ax[1], modules["epi"], "epi-rising module", profiles, sigma, n_per)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.12, wspace=0.16)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 3 l: protein against transcript slopes

COLOUR_CONCORDANT, COLOUR_DISCORDANT, COLOUR_QUADRANT = "#4C72B0", "#DD8452", "#DBE7F3"
SLOPE_RC = {"font.family": "DejaVu Sans", "font.size": 12, "pdf.fonttype": 42}


def _padded(low, high):
    """Axis limits with 8 % of the range added on both sides."""
    return (low - 0.08 * (high - low), high + 0.08 * (high - low))


def slope_concordance(slopes):
    """Fig 3 l: transmural slope of the protein against the transcript of each gene.

    Sign-concordant quadrants are shaded, the dashed line is the least-squares fit, and the box gives Spearman
    rho, its p value and the share of genes with the same sign. Genes with a zero or missing slope are left out.

    Parameters
    ----------
    slopes
        One row per gene with ``gene``, ``transcript_slope`` and ``protein_slope``.
    """
    slopes = slopes.dropna(subset=["transcript_slope", "protein_slope"])
    slopes = slopes[(slopes.transcript_slope != 0) & (slopes.protein_slope != 0)]
    x, y = slopes.transcript_slope.to_numpy(float), slopes.protein_slope.to_numpy(float)
    genes = slopes.gene.tolist()
    rho, p = stats.spearmanr(x, y)
    concordant = np.sign(x) == np.sign(y)
    fit_slope, fit_intercept = np.polyfit(x, y, 1)
    xlim = _padded(min(x.min(), 0), max(x.max(), 0))
    ylim = _padded(min(y.min(), 0), max(y.max(), 0))
    with panel((7.0552, 6.4397), SLOPE_RC) as fig:
        ax = fig.add_axes([0.1161, 0.0894, 0.8697, 0.8581])
        ax.add_patch(plt.Rectangle((0, 0), xlim[1], ylim[1], facecolor=COLOUR_QUADRANT, edgecolor="none", zorder=0))
        ax.add_patch(
            plt.Rectangle((xlim[0], ylim[0]), -xlim[0], -ylim[0], facecolor=COLOUR_QUADRANT, edgecolor="none", zorder=0)
        )
        ax.axhline(0, color="0.53", lw=0.8, zorder=1)
        ax.axvline(0, color="0.53", lw=0.8, zorder=1)
        line_x = np.array(xlim)
        ax.plot(line_x, fit_intercept + fit_slope * line_x, color="0.35", lw=1.2, ls=(0, (4.44, 1.92)), zorder=2)
        ax.scatter(
            x[concordant],
            y[concordant],
            s=55,
            c=COLOUR_CONCORDANT,
            edgecolors="none",
            zorder=3,
            label="sign-concordant",
        )
        ax.scatter(
            x[~concordant],
            y[~concordant],
            s=55,
            c=COLOUR_DISCORDANT,
            edgecolors="none",
            zorder=3,
            label="sign-discordant",
        )
        if len(genes) <= 50:
            for gene, xi, yi in zip(genes, x, y):
                ax.annotate(
                    gene,
                    (xi, yi),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=7 if len(genes) > 20 else 8,
                    ha="left",
                    va="center",
                )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("transcript transmural slope  (epi ← 0 → endo)", fontsize=12)
        ax.set_ylabel("protein transmural direction  (epi ← 0 → endo)", fontsize=12)
        ax.set_title(f"DVP protein vs transcript transmural slope ({len(genes)} genes)", fontsize=11.5)
        ax.text(
            0.02,
            0.97,
            f"Spearman ρ = {rho:.2f}\np = {p:.2g}\nsign agreement {concordant.mean():.0%}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "0.7", "lw": 0.8},
        )
        ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=1, edgecolor="0.7")
    return fig


# ---------------------------------------------------------------------------------------------------------------
# ED6 d: marker genes of the Level-3 labels

MARKER_XLABEL = "Visium HD FICTURE Level-3 label (CMFLAG-clean 2 µm bins, 11 slides)"
# layout in points: page size, heat-map axes and colour-bar axes (left, bottom, width, height)
MARKER_PAGE = (667.8949978221, 913.1558336047)
MARKER_AXES = (57.371875, 122.3113026381, 543.217122931, 754.9726559666)
MARKER_COLOURBAR = (612.0, 328.5158336047, 17.28, 342.96)
MARKER_RC = {
    "font.sans-serif": ["DejaVu Sans"],
    "savefig.pad_inches": 0.1,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "pdf.fonttype": 42,
}


def label_markers(z, markers):
    """ED6 d: z-scored expression of marker genes across the Level-3 labels; the expected label is boxed.

    Parameters
    ----------
    z
        Marker genes x labels z-scores (:func:`hhca.tl.marker_zscores`).
    markers
        Mapping of label -> its marker genes, in the order of the rows.
    """
    width, height = MARKER_PAGE
    genes, types = list(z.index), list(z.columns)
    with panel((width / 72, height / 72), MARKER_RC) as fig:
        ax = fig.add_axes(
            [MARKER_AXES[0] / width, MARKER_AXES[1] / height, MARKER_AXES[2] / width, MARKER_AXES[3] / height]
        )
        im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
        ax.set_yticks(range(len(genes)))
        ax.set_yticklabels(genes, fontsize=9.5)
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels(types, rotation=45, ha="right", fontsize=12, fontweight="bold")
        ax.set_xlabel(MARKER_XLABEL, fontsize=11)
        row = 0
        for cell_type, type_genes in markers.items():
            ax.add_patch(
                Rectangle(
                    (types.index(cell_type) - 0.5, row - 0.5),
                    1,
                    len(type_genes),
                    fill=False,
                    edgecolor="#111111",
                    linewidth=1.2,
                )
            )
            row += len(type_genes)
        cax = fig.add_axes(
            [
                MARKER_COLOURBAR[0] / width,
                MARKER_COLOURBAR[1] / height,
                MARKER_COLOURBAR[2] / width,
                MARKER_COLOURBAR[3] / height,
            ]
        )
        colorbar = fig.colorbar(im, cax=cax)
        colorbar.set_label("z (mean expr across types)", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# ED6 e: FICTURE factors against atlas types

CONCORDANCE_RC = {"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42}


def factor_concordance(r, short_names=None):
    """ED6 e: correlation of the FICTURE factors (rows) with the atlas types (columns) over the atlas markers.

    The diagonal is boxed; a dot marks the best-matching atlas type of a factor where it is off the diagonal.

    Parameters
    ----------
    r
        Concordance matrix of :func:`hhca.tl.factor_concordance`.
    short_names
        Level-3 type -> label on the axes.
    """
    labels = [(short_names or {}).get(cell_type, cell_type) for cell_type in r.index]
    self_match = sum(r.loc[cell_type].idxmax() == cell_type for cell_type in r.index)
    with panel((6.2, 5.6), CONCORDANCE_RC) as fig:
        ax = fig.subplots()
        im = ax.imshow(r.values, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(len(r)):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#111111", lw=0.8))
            j = int(np.argmax(r.values[i]))
            if j != i:
                ax.plot(j, i, marker="o", ms=3, color="#111111")
        ax.set_xticks(range(len(r)))
        ax.set_xticklabels(labels, rotation=60, ha="right")
        ax.set_yticks(range(len(r)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("snRNA atlas Level-3 type (marker genes)")
        ax.set_ylabel("Visium HD FICTURE factor")
        ax.set_title(
            f"self-match {self_match}/{len(r)}, diagonal mean r = {float(np.diag(r.values).mean()):.2f}", fontsize=9
        )
        colorbar = fig.colorbar(im, ax=ax, shrink=0.7)
        colorbar.set_label("Pearson r")
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------------------
# ED6 f: whole section with a detail field

CM_GREY = "#D5D5D5"
SECTION_RC = {**ARIAL_RC, "savefig.pad_inches": 0.0125}


def _draw_section(ax, raster, palette, zoom_box):
    """Whole-section label raster with the outline of the detail field and a 1 mm scale bar."""
    cats = raster["cats"]
    lut = np.array(
        [to_rgb(CM_GREY if c in CARDIOMYOCYTES else palette.get(c, "#cccccc")) for c in cats] + [(1.0, 1.0, 1.0)]
    )
    img = raster["img"].astype(np.int64)
    rgb = lut[np.where(img < 0, len(cats), img)]
    x0, x1, y0, y1 = raster["ext"]
    zx0, zy0, zw = zoom_box
    ax.imshow(rgb, extent=[x0, x1, y1, y0], interpolation="nearest", rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color("#9A9A9A")
        spine.set_linewidth(0.68)
    ax.add_patch(Rectangle((zx0, zy0), zw, zw, fill=False, edgecolor=INK, lw=0.9, zorder=5))
    ax.text(
        0.016,
        0.0165,
        "2 µm bins, whole-transcriptome\natlas-guided FICTURE",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=INK,
        linespacing=1.15,
    )
    bar = 1000.0
    bx = x1 - bar - 0.022 * (x1 - x0)
    by = y1 - 0.026 * (y1 - y0)
    ax.plot([bx, bx + bar], [by, by], color=INK, lw=2.2, solid_capstyle="butt")
    ax.text(bx + bar / 2, by - 0.035 * (y1 - y0), "1 mm", ha="center", va="bottom", fontsize=7.0, color=INK)


def _draw_detail(ax, detail, palette, zoom_limits, zw):
    """Bins of the detail field (cardiomyocyte bins small, the others on top) with a 100 um scale bar."""
    zx0, zx1, zy0, zy1 = zoom_limits
    labels = detail["cats"][detail["code"]]
    is_cm = np.isin(labels, CARDIOMYOCYTES)
    colors = np.array([to_rgb(palette.get(label, "#cccccc")) for label in labels])
    ax.scatter(
        detail["x"][is_cm], detail["y"][is_cm], s=0.17, c=colors[is_cm], linewidths=0, marker="s", rasterized=True
    )
    ax.scatter(
        detail["x"][~is_cm],
        detail["y"][~is_cm],
        s=1.6,
        c=colors[~is_cm],
        linewidths=0,
        marker="s",
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.set_xlim(zx0, zx1)
    ax.set_ylim(zy1, zy0)
    ax.set_xticks([])
    ax.set_yticks([])
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(INK)
        spine.set_linewidth(0.9)
    bar = 100.0
    bx = zx1 - bar - 0.064 * zw
    by = zy1 - 0.055 * zw
    ax.plot([bx, bx + bar], [by, by], color=INK, lw=2.2, solid_capstyle="butt")
    ax.text(
        bx + bar / 2,
        by - 0.043 * zw,
        "100 µm",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=INK,
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
    )


def section_map(raster, detail, palette=None):
    """ED6 f: Level-3 labels of a whole section (cardiomyocytes light grey) and of a detail field.

    Parameters
    ----------
    raster
        Label raster of the section (:func:`hhca.tl.label_raster`).
    detail
        Bins of the detail field (:func:`hhca.tl.extract_window`).
    palette
        Level-3 type -> colour (default: the palette of the spatial panels).
    """
    palette = palette or style.SPATIAL
    zx0, zx1, zy0, zy1 = (float(v) for v in detail["ext"])
    zw = zx1 - zx0
    with panel((4.276, 2.100), SECTION_RC, save_dpi=400) as fig:
        ax_section = fig.add_axes([0.004, 0.007, 0.524, 0.936])
        ax_detail = fig.add_axes([0.552, 0.023, 0.444, 0.905])
        _draw_section(ax_section, raster, palette, (zx0, zy0, zw))
        _draw_detail(ax_detail, detail, palette, (zx0, zx1, zy0, zy1), zw)
        for (dx, dy), (rx, ry) in [((zx1, zy0), (0, 1)), ((zx1, zy1), (0, 0))]:
            fig.add_artist(
                ConnectionPatch(
                    xyA=(dx, dy),
                    coordsA=ax_section.transData,
                    xyB=(rx, ry),
                    coordsB=ax_detail.transAxes,
                    color=INK,
                    lw=1.0,
                    ls=(0, (2.8, 1.4)),
                )
            )
    return fig
