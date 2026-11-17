"""Panels of the standard-Visium deconvolution (Fig 3 c, ED6 c, g, h, ED8 b-c)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, ConnectionPatch, Patch, Rectangle, Wedge
from scipy.spatial import cKDTree

from hhca.pl import style
from hhca.pl._panel import panel
from hhca.tl._deconvolution import compare_scores


def _style_rc() -> dict:
    """The settings of :func:`hhca.pl.style.apply`, for panels that are drawn with them."""
    with plt.style.context("default"):
        default = dict(mpl.rcParams)
        style.apply()
        return {key: value for key, value in mpl.rcParams.items() if value != default[key]}


# ---------------------------------------------------------------------------------------------------------------
# Fig 3 c: chamber markers in cardiomyocyte-dominant spots

CHAMBER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
}


def chamber_markers(zscores, direction, n_spots, highlight=()):
    """Fig 3 c: atrial and ventricular cardiomyocyte genes across the chamber regions.

    Heatmap of the z-scored mean expression in cardiomyocyte-dominant spots; atrial genes above the line and
    labelled in orange, ventricular genes below and in blue; cells without a value are hatched.

    Parameters
    ----------
    zscores
        Genes x regions (:func:`hhca.tl.region_zscores`), atrial genes first.
    direction
        Gene -> ``atria`` / ``ventricle``.
    n_spots
        Region -> number of cardiomyocyte-dominant spots, written under the region.
    highlight
        Genes labelled in bold italics.
    """
    regions, genes = list(zscores.columns), list(zscores.index)
    n_atria = int((direction.reindex(genes) == "atria").sum())
    with panel((4.3, 6.6), CHAMBER_RC) as fig:
        ax = fig.subplots()
        vmax = float(np.nanmax(np.abs(zscores.values)))
        im = ax.imshow(zscores.values, cmap=style.DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(
            [f"{r}\n{n_spots[r] // 1000}k" if n_spots[r] >= 1000 else f"{r}\n{n_spots[r]}" for r in regions]
        )
        ax.set_yticks(range(len(genes)))
        ax.set_yticklabels(genes)
        for tick_label, gene in zip(ax.get_yticklabels(), genes):
            tick_label.set_color(style.CHAMBER_GENES[direction[gene]])
            if gene in highlight:
                tick_label.set_fontstyle("italic")
                tick_label.set_fontweight("bold")
        ax.axhline(n_atria - 0.5, color="black", lw=1.2)
        ax.axvline(1.5, color="black", lw=1.2)
        ax.tick_params(length=2)
        for gene_row in range(len(genes)):
            for region_col in range(len(regions)):
                if not np.isfinite(zscores.values[gene_row, region_col]):
                    ax.add_patch(
                        Rectangle(
                            (region_col - 0.5, gene_row - 0.5),
                            1,
                            1,
                            facecolor="0.85",
                            edgecolor="white",
                            hatch="///",
                            lw=0,
                        )
                    )
        colorbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        colorbar.set_label("z-scored mean log1p expression\n(across the four regions)", fontsize=8)
        colorbar.ax.tick_params(labelsize=7.5)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# ED6 c, g: per-spot proportions on one section

SHORT = {
    "Ventricular Cardiomyocytes": "Ventricular CM",
    "Atrial Cardiomyocytes": "Atrial CM",
    "Arterial Endothelial cells": "Arterial EC",
    "Capillary Endothelial cells": "Capillary EC",
    "Venous Endothelial cells": "Venous EC",
    "Endocardial Endothelial Cells": "Endocardial EC",
    "Lymphatic Endothelial Cells": "Lymphatic EC",
    "Vascular Smooth Muscle cells": "VSMC",
    "Lymphoid Cells (Proliferating)": "Lymphoid pr.",
    "Neural Cells": "Neural",
    "Mast cells": "Mast",
    "Plasma Cells": "Plasma",
    "Dendritic cells": "Dendritic",
}
LEGEND = [
    "Ventricular Cardiomyocytes",
    "Atrial Cardiomyocytes",
    "Capillary Endothelial cells",
    "Arterial Endothelial cells",
    "Venous Endothelial cells",
    "Endocardial Endothelial Cells",
    "Lymphatic Endothelial Cells",
    "Pericytes",
    "Vascular Smooth Muscle cells",
    "Fibroblasts",
    "Adipocytes",
    "Epicardium",
    "Neural Cells",
    "Macrophages",
    "Monocytes",
    "Dendritic cells",
    "Mast cells",
    "T cells",
    "NK cells",
    "B cells",
    "Plasma Cells",
    "ILC",
    "Lymphoid Cells (Proliferating)",
]
MAP_METHODS = ["spotiphy", "RCTD", "cell2location", "ucaspatial", "stereoscope", "SPOTlight", "NNLS"]
MAP_DPI = 330
# ED6 g: page of 10.80 x 3.83 in with 8 columns and 2 rows
MAP_PAGE = (10.80155125, 3.8349666666666664)
MAP_COLUMN_LEFT = [0.0162, 0.1396, 0.263, 0.3865, 0.5099, 0.6333, 0.7568, 0.8802]
MAP_AXES_WIDTH = 0.117
MAP_TOP_ROW, MAP_BOTTOM_ROW = (0.5952, 0.3139), (0.1895, 0.3295)  # bottom edge and height of the axes
# ED6 c: axes sized as the first of the three rows of the panel
PIE_PAGE = (3.35, 9.30)
PIE_AXES_WIDTH, PIE_AXES_LEFT = 0.62, 0.19
PIE_ROW_HEIGHT = 0.255


def _section(adata):
    """Spot positions (x, y in full-resolution pixels), scale to the H&E image, spot radius and pitch."""
    library = next(iter(adata.uns["spatial"]))
    scale = adata.uns["spatial"][library]["scalefactors"]["tissue_hires_scalef"]
    coords = pd.DataFrame(np.asarray(adata.obsm["spatial"]), index=adata.obs_names, columns=["x", "y"])
    xy = coords[["x", "y"]].values
    pitch = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    return coords, scale, 0.5 * pitch * scale * 0.98, pitch


def _he_image(adata):
    """H&E image of the section as floats between 0 and 1."""
    library = next(iter(adata.uns["spatial"]))
    image = adata.uns["spatial"][library]["images"]["hires"]
    return np.divide(image, 255, dtype=np.float32) if image.dtype == np.uint8 else image


def _method_spots(adata, coords, method):
    """Proportions of one method and the positions of the spots it has values for."""
    proportions = adata.obsm[method]
    complete = proportions.notna().all(axis=1).values
    return (coords, proportions) if complete.all() else (coords[complete], proportions[complete])


def _draw_dominant(ax, spots, proportions, colours, scale, radius):
    """One disc per spot in the colour of its dominant cell type."""
    dominant = proportions.idxmax(axis=1).loc[spots.index]
    circles = [Circle((x * scale, y * scale), radius) for x, y in spots[["x", "y"]].values]
    ax.add_collection(PatchCollection(circles, facecolors=[colours[cell_type] for cell_type in dominant], linewidths=0))


def _draw_pies(ax, spots, proportions, colours, scale, radius, min_frac=0.02):
    """One pie per spot: cell types with a proportion of at least min_frac, largest first, clockwise from 12."""
    wedges, face_colours = [], []
    for barcode, (x, y) in spots[["x", "y"]].iterrows():
        fractions = proportions.loc[barcode]
        fractions = fractions[fractions >= min_frac].sort_values(ascending=False)
        if fractions.empty:
            continue
        fractions = fractions / fractions.sum()
        theta = 90.0
        for cell_type, fraction in fractions.items():
            span = 360.0 * float(fraction)
            wedges.append(Wedge((x * scale, y * scale), radius, theta - span, theta))
            face_colours.append(colours[cell_type])
            theta -= span
    ax.add_collection(PatchCollection(wedges, facecolors=face_colours, linewidths=0))


def _window(fov):
    """Edges x0, x1, y0, y1 of a square window given by its centre and size."""
    (centre_x, centre_y), size = fov["centre_px"], fov["size_px"]
    return centre_x - size / 2, centre_x + size / 2, centre_y - size / 2, centre_y + size / 2


def spot_pies(adata, method, fov):
    """ED6 c, Visium row: the spots of a window drawn as pies of their deconvolved proportions.

    Parameters
    ----------
    adata
        Spots of one section: positions in ``obsm["spatial"]`` (full-resolution pixels), the proportions in
        ``obsm[method]`` and the scale factors of the section in ``uns["spatial"]``.
    method
        Method whose proportions are drawn.
    fov
        Window: ``{"centre_px": [x, y], "size_px": side}`` in full-resolution pixels.
    """
    coords, scale, radius, pitch = _section(adata)
    spots, proportions = _method_spots(adata, coords, method)
    colours = {cell_type: style.ct(cell_type) for cell_type in proportions.columns}
    x0, x1, y0, y1 = _window(fov)
    window = spots[(spots.x.between(x0, x1)) & (spots.y.between(y0, y1))]
    page_width, page_height = PIE_PAGE
    with panel((page_width, page_height * (PIE_ROW_HEIGHT + 0.06)), _style_rc(), save_dpi=MAP_DPI) as fig:
        ax = fig.add_axes(
            [PIE_AXES_LEFT, 0.03 / (PIE_ROW_HEIGHT + 0.06), PIE_AXES_WIDTH, PIE_ROW_HEIGHT / (PIE_ROW_HEIGHT + 0.06)]
        )
        _draw_pies(ax, window, proportions, colours, scale, radius)
        ax.set_xlim(x0 * scale, x1 * scale)
        ax.set_ylim(y1 * scale, y0 * scale)
        ax.set_facecolor("white")
        bar_um, px_per_um = 100.0, (pitch * scale) / 100.0
        bar_x, bar_y = x1 * scale - bar_um * px_per_um - 8, y1 * scale - 8
        ax.plot([bar_x, bar_x + bar_um * px_per_um], [bar_y, bar_y], color="black", lw=1.8, solid_capstyle="butt")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)
        ax.set_title("Visium, 55 µm spots\ndeconvolution (proportions)", fontsize=8.4, linespacing=1.25, pad=6)
    return fig


def method_maps(adata, fov, methods=MAP_METHODS, best="spotiphy"):
    """ED6 g: per-spot deconvolution of one section by several methods.

    Top row: the H&E image and, per method, the dominant cell type of every spot; bottom row: the spots of the
    zoom window as pies of their proportions.

    Parameters
    ----------
    adata
        Spots of one section: positions in ``obsm["spatial"]`` (full-resolution pixels), the proportions of each
        method in ``obsm[<method>]``, H&E image and scale factors in ``uns["spatial"]``.
    fov
        Zoom window: ``{"centre_px": [x, y], "size_px": side}`` in full-resolution pixels.
    methods
        Methods, in the order of the columns.
    best
        Method whose name is highlighted.
    """
    coords, scale, radius, pitch = _section(adata)
    he_image = _he_image(adata)
    colours = {cell_type: style.ct(cell_type) for cell_type in LEGEND}
    x0, x1, y0, y1 = _window(fov)
    zoom_x0, zoom_x1, zoom_y0, zoom_y1 = x0 * scale, x1 * scale, y0 * scale, y1 * scale
    # extent of the whole section with a margin
    pad = 0.03 * max(coords.x.max() - coords.x.min(), coords.y.max() - coords.y.min())
    full_x0, full_x1 = (coords.x.min() - pad) * scale, (coords.x.max() + pad) * scale
    full_y0, full_y1 = (coords.y.min() - pad) * scale, (coords.y.max() + pad) * scale

    rc = {**_style_rc(), "font.sans-serif": ["DejaVu Sans"], "savefig.pad_inches": 0.02}
    with panel(MAP_PAGE, rc, save_dpi=MAP_DPI) as fig:
        top = [fig.add_axes([left, MAP_TOP_ROW[0], MAP_AXES_WIDTH, MAP_TOP_ROW[1]]) for left in MAP_COLUMN_LEFT]
        bottom = [
            fig.add_axes([left, MAP_BOTTOM_ROW[0], MAP_AXES_WIDTH, MAP_BOTTOM_ROW[1]]) for left in MAP_COLUMN_LEFT
        ]
        top[0].imshow(he_image)
        top[0].set_xlim(0, he_image.shape[1])
        top[0].set_ylim(he_image.shape[0], 0)
        top[0].set_title("H&E, full slide", fontsize=7.2)
        top[0].set_ylabel("reference", fontsize=7.0)
        bottom[0].imshow(he_image)
        bottom[0].set_ylabel("H&E, zoom", fontsize=7.0)
        for column, method in enumerate(methods, start=1):
            spots, proportions = _method_spots(adata, coords, method)
            _draw_dominant(top[column], spots, proportions, colours, scale, radius)
            top[column].set_xlim(full_x0, full_x1)
            top[column].set_ylim(full_y1, full_y0)
            top[column].set_title(
                method,
                fontsize=7.6,
                color=style.METHOD[best] if method == best else "black",
                fontweight="bold" if method == best else "normal",
            )
            zoom = spots[(spots.x.between(x0, x1)) & (spots.y.between(y0, y1))]
            _draw_pies(bottom[column], zoom, proportions, colours, scale, radius)
        for ax in bottom:
            ax.set_xlim(zoom_x0, zoom_x1)
            ax.set_ylim(zoom_y1, zoom_y0)
        for ax in top + bottom:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_visible(False)
        for column in range(len(top)):
            top[column].add_patch(
                Rectangle(
                    (zoom_x0, zoom_y0), zoom_x1 - zoom_x0, zoom_y1 - zoom_y0, fill=False, ec="black", lw=0.7, zorder=5
                )
            )
            for xy_top, xy_bottom in (((zoom_x0, zoom_y1), (0, 1)), ((zoom_x1, zoom_y1), (1, 1))):
                fig.add_artist(
                    ConnectionPatch(
                        xyA=xy_top,
                        coordsA=top[column].transData,
                        xyB=xy_bottom,
                        coordsB=bottom[column].transAxes,
                        linestyle=(0, (4, 3)),
                        lw=0.5,
                        color="black",
                    )
                )
        bar_um, px_per_um = 100.0, (pitch * scale) / 100.0
        ax = bottom[-1]
        ax.plot(
            [zoom_x1 - bar_um * px_per_um - 10, zoom_x1 - 10],
            [zoom_y1 - 10, zoom_y1 - 10],
            lw=1.8,
            color="black",
            solid_capstyle="butt",
        )
        ax.text(
            zoom_x1 - bar_um * px_per_um / 2 - 10,
            zoom_y1 - 15,
            f"{bar_um:.0f} µm",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )
        fig.text(
            0.5,
            0.955,
            f"Standard-Visium per-spot deconvolution across {len(methods)} methods, Level-3 atlas colours "
            "(top: dominant type; bottom: per-spot proportion pies; H&E reference at left)",
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
        )
        handles = [Patch(facecolor=colours[cell_type], label=SHORT.get(cell_type, cell_type)) for cell_type in LEGEND]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=12,
            frameon=False,
            fontsize=6.6,
            handlelength=1.0,
            handleheight=1.0,
            columnspacing=1.2,
            labelspacing=0.3,
            title="Level-3 cell types — HHCA atlas palette",
            title_fontsize=7.0,
        )
    return fig


# ---------------------------------------------------------------------------------------------------------------
# ED6 h: composition per method

COMPOSITION_PAGE_PT = (756.094452, 453.15664)  # page size, points
COMPOSITION_AXES_PT = (50.840625, 54.1400399049, 575.6752, 355.6322248317)  # axes box (left, bottom, width, height)
COMPOSITION_METHODS = ["spotiphy", "cell2location", "RCTD", "ucaspatial", "SPOTlight", "NNLS", "stereoscope"]
COMPOSITION_RC = {
    "font.sans-serif": ["DejaVu Sans"],
    "savefig.pad_inches": 0.1,
    "legend.labelspacing": 0.5,
    "legend.handlelength": 2.0,
    "legend.handletextpad": 0.8,
    "legend.borderpad": 0.4,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
}


def method_composition(composition, methods=COMPOSITION_METHODS, best="spotiphy"):
    """ED6 h: mean cell-type composition of the sections per deconvolution method, as stacked bars.

    Parameters
    ----------
    composition
        Methods x compartments: mean proportion over the sections (the section means of
        :func:`hhca.tl.compartment_composition` averaged per method).
    methods
        Methods, in the order of the bars.
    best
        Method whose name is drawn in bold.
    """
    compartments = list(style.COMPARTMENT)
    composition = composition.reindex(index=methods, columns=compartments)
    page_width, page_height = COMPOSITION_PAGE_PT
    with panel((page_width / 72, page_height / 72), {**_style_rc(), **COMPOSITION_RC}) as fig:
        left, bottom_edge, width, height = COMPOSITION_AXES_PT
        ax = fig.add_axes([left / page_width, bottom_edge / page_height, width / page_width, height / page_height])
        x = np.arange(len(composition))
        bottom = np.zeros(len(composition))
        for compartment in compartments:
            ax.bar(
                x,
                composition[compartment].values,
                bottom=bottom,
                width=0.8,
                color=style.COMPARTMENT[compartment],
                edgecolor="white",
                linewidth=0.5,
                label=compartment,
            )
            bottom += composition[compartment].values
        ax.set_xticks(x)
        ax.set_xticklabels(composition.index, rotation=20, ha="right", fontsize=13)
        for tick_label, method in zip(ax.get_xticklabels(), composition.index):
            if method == best:
                tick_label.set_fontweight("bold")
        ax.set_ylabel("Cohort-mean proportion", fontsize=14)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(labelsize=12)
        ax.set_title(
            f"Cell-type composition per deconvolution method\n({len(methods)} methods, cohort mean of 11 slides)",
            fontsize=15,
            pad=6,
        )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=12,
            frameon=False,
            title="Compartment",
            title_fontsize=13,
        )
    return fig


# ---------------------------------------------------------------------------------------------------------------
# ED8 b-c: atrial-pericyte signature

SIGNATURE_INK, SIGNATURE_MUTED = "#1a1a1a", "#6E6E6E"
SIGNATURE_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelcolor": SIGNATURE_INK,
    "text.color": SIGNATURE_INK,
    "xtick.color": SIGNATURE_INK,
    "ytick.color": SIGNATURE_INK,
    "axes.edgecolor": SIGNATURE_INK,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
}
SIGNATURE_MAX_POINTS = 1200  # spots drawn per group in the box plots
SIGNATURE_PARTS = ("genes", "atria", "chambers")
SIGNATURE_CAPTION = (
    "Kanemaru Visium (LA, RA, LV, RV); cell2location deconvolution shipped with the objects; "
    "pericyte-high/low = top/bottom within-region tertile of total pericyte abundance "
    "(PC1_vent + PC2_atria + PC3_str + PC4_CMC-like)."
)


def _draw_signature_genes(fig, ax, chamber_means):
    """Signature genes x chambers heatmap of the mean z-scored expression."""
    regions = style.CHAMBER_REGION
    vmax = float(np.nanmax(np.abs(chamber_means.values)))
    im = ax.imshow(chamber_means.values, cmap=style.DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(chamber_means.shape[1]))
    ax.set_xticklabels(chamber_means.columns, fontsize=11)
    ax.set_yticks(range(len(chamber_means)))
    ax.set_yticklabels(chamber_means.index, fontsize=8.5, style="italic")
    ax.axvline(1.5, color=SIGNATURE_INK, lw=1.6)
    ax.text(0.5, -1.6, "atria", ha="center", fontsize=11.5, color=regions["atria"], fontweight="bold")
    ax.text(2.5, -1.6, "ventricles", ha="center", fontsize=11.5, color=regions["ventricle"], fontweight="bold")
    colorbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label("mean z-scored expression", fontsize=9.5)
    colorbar.ax.tick_params(labelsize=8.5)
    ax.set_title("Atrial-pericyte signature genes", fontsize=12, pad=22)


def _draw_score_boxes(ax, groups, colours, names, title, rng):
    """Box plot of the signature score of two groups with jittered spots (second group against the first).

    Without an axis only the random draws of the jitter are made, so that later boxes get the same points
    whether or not this one is drawn.
    """
    shown, jitter = [], []
    for i, values in enumerate(groups):
        subsample = len(values) > SIGNATURE_MAX_POINTS
        shown.append(rng.choice(values, SIGNATURE_MAX_POINTS, replace=False) if subsample else values)
        jitter.append(rng.normal(i, 0.07, len(shown[i])))
    if ax is None:
        return
    test = compare_scores(groups[0], groups[1])
    boxes = ax.boxplot(
        groups,
        positions=[0, 1],
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": SIGNATURE_INK, "lw": 1.6},
    )
    for box, colour in zip(boxes["boxes"], colours):
        box.set_facecolor(colour)
        box.set_alpha(0.6)
        box.set_edgecolor(SIGNATURE_MUTED)
    for i in range(len(groups)):
        ax.scatter(jitter[i], shown[i], s=4, alpha=0.25, lw=0, color=colours[i], zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{names[0]}\n(n={len(groups[0]):,})", f"{names[1]}\n(n={len(groups[1]):,})"], fontsize=10)
    ax.set_ylabel("atrial-pericyte signature score", fontsize=10.5)
    ax.set_title(
        f"{title}\nMWU p={test['mwu_p']:.2g}, Δmedian={test['delta_median']:+.3f}",
        fontsize=11,
        pad=10,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def pericyte_signature(chamber_means=None, spots=None, show=SIGNATURE_PARTS, caption=False, seed=0):
    """ED8 b-c: the atrial-pericyte signature on the Visium sections of the four chambers.

    Up to three parts on a fixed layout: ``"genes"``, the mean z-scored expression of the signature genes per
    chamber (ED8 b); ``"atria"``, the signature score of pericyte-high against pericyte-low spots of the atria;
    ``"chambers"``, the score of pericyte-high spots in atria against ventricles (ED8 c). Each box plot gives the
    two-sided Mann-Whitney U test and the difference of the medians.

    Parameters
    ----------
    chamber_means
        Genes x chambers (:func:`hhca.tl.signature_by_chamber`), for ``"genes"``.
    spots
        One row per spot with ``region`` (``atria`` / ``ventricle``), ``peri_grp`` (``pericyte-high`` /
        ``pericyte-low``) and ``score``, for the box plots.
    show
        Parts to draw.
    caption
        Add a line on data and spot groups under the parts.
    seed
        Seed of the jitter of the points.
    """
    regions = style.CHAMBER_REGION
    with panel((15.4, 5.2), SIGNATURE_RC) as fig:
        grid = fig.add_gridspec(
            1, 3, width_ratios=[1.35, 1, 1], wspace=0.42, left=0.075, right=0.975, top=0.86, bottom=0.16
        )
        if "genes" in show:
            _draw_signature_genes(fig, fig.add_subplot(grid[0]), chamber_means)
        if spots is not None:
            rng = np.random.default_rng(seed)
            atria = spots[spots.region == "atria"]
            _draw_score_boxes(
                fig.add_subplot(grid[1]) if "atria" in show else None,
                [atria[atria.peri_grp == group].score.values for group in ["pericyte-low", "pericyte-high"]],
                [style.PERICYTE_LOW, regions["atria"]],
                ["pericyte-low", "pericyte-high"],
                "Atria: pericyte-high vs -low spots",
                rng,
            )
            high = spots[spots.peri_grp == "pericyte-high"]
            _draw_score_boxes(
                fig.add_subplot(grid[2]) if "chambers" in show else None,
                [high[high.region == region].score.values for region in ["ventricle", "atria"]],
                [regions["ventricle"], regions["atria"]],
                ["ventricles", "atria"],
                "Pericyte-high spots: atria vs ventricles",
                rng,
            )
        if caption:
            fig.text(0.5, 0.035, SIGNATURE_CAPTION, ha="center", fontsize=8.6, color=SIGNATURE_MUTED)
    return fig
