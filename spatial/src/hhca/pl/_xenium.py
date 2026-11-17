"""Panels of the Xenium analysis (Fig 6 c-l, ED9 a-c, ED6 i-j)."""

from __future__ import annotations

import colorsys
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from hhca.pl import style
from hhca.pl._panel import PANEL_DPI, PanelFigure, panel, panel_rc
from hhca.stats import stars
from hhca.tl._rejection import PAIRS, REJECTING, REJECTION_TYPES, SHORT_TYPES, programme_name


def _style_rc() -> dict:
    """The matplotlib settings of ``style.apply()``, as the settings of a panel."""
    with mpl.rc_context(), plt.style.context("default"):
        default = mpl.rcParams.copy()
        style.apply()
        return {key: value for key, value in mpl.rcParams.items() if value != default[key]}


PANEL_RC = _style_rc()
# the larger fonts of the statistics panels
STATS_RC = {**PANEL_RC, "axes.titlesize": 12.5, "axes.labelsize": 11, "xtick.labelsize": 10.5, "ytick.labelsize": 10}
INK, MUTED = style.INK, style.MUTED


class DarkPanelFigure(PanelFigure):
    """Panel on a dark background: it is saved and shown with its own face colour."""

    def savefig(self, fname, **kwargs):
        """Save with the face colour of the figure."""
        kwargs["facecolor"] = self.get_facecolor()
        super().savefig(fname, **kwargs)


@contextmanager
def dark_panel(figsize, rc: dict, facecolor, save_dpi: int = PANEL_DPI):
    """Open a :class:`DarkPanelFigure` of `figsize` inches with the background `facecolor`."""
    with panel_rc(rc):
        yield plt.figure(figsize=figsize, FigureClass=DarkPanelFigure, rc=rc, save_dpi=save_dpi, facecolor=facecolor)


def ct(name: str, default: str = "#B9B9B9") -> str:
    """Colour of a cell type (atlas name or display name of the Xenium panels)."""
    return style.ct(name, default, aliases=style.XENIUM_ALIASES)


# ---------------------------------------------------------------------------------------------------------------
# Fig 6 c: slide with the immune classes, one core enlarged

BACKGROUND = "#0a0a0a"
STRUCTURAL = "#3d3d3d"
INSET_FRAME = "#f0c000"
SLIDE_NAMES = {
    "Plasma Cells": "Plasma cells",
    "CD8 T cells": "CD8 T",
    "CD4 T cells": "CD4 T",
    "Tregs": "Treg",
    "NK cells": "NK",
    "Proliferating CD8 T cells": "prolif. CD8 T",
    "Proliferating CD4 T cells": "prolif. CD4 T",
    "Proliferating Tregs": "prolif. Treg",
    "Proliferating NK cells": "prolif. NK",
}


def _slide_colour(name, lymphoid_colours):
    """Colour of a class on the slide: the lymphoid Level-4 palette, otherwise the Level-3 atlas colour."""
    return lymphoid_colours[name] if name in lymphoid_colours else ct(name)


def _draw_slide_cells(ax, cells, order, lymphoid_colours, background_size, size):
    """Scatter the cells: structural in grey, then the immune classes with the lymphoid ones on top."""
    structural = cells[cells.disp.isna()]
    ax.scatter(structural.x, structural.y, s=background_size, c=STRUCTURAL, linewidths=0, rasterized=True, zorder=1)
    for name in order:
        subset = cells[cells.disp == name]
        if not len(subset):
            continue
        lymphoid = name in lymphoid_colours
        ax.scatter(
            subset.x,
            subset.y,
            s=size * (1.3 if lymphoid else 0.92),
            c=_slide_colour(name, lymphoid_colours),
            alpha=1.0 if lymphoid else 0.55,
            linewidths=0,
            rasterized=True,
            zorder=3 if lymphoid else 2,
        )


def _draw_zoom(ax, core_cells, core, order, lymphoid_colours):
    """Inset with the cells of one core, a 100 um scale bar and a frame at its position on the slide."""
    inset = ax.inset_axes([-0.60, 0.63, 0.58, 0.34])
    inset.set_facecolor(BACKGROUND)
    _draw_slide_cells(inset, core_cells, order, lymphoid_colours, 3.0, 7.5)
    inset.set_aspect("equal")
    inset.invert_yaxis()
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color(INSET_FRAME)
        spine.set_linewidth(1.6)
    x0, x1 = core_cells.x.min(), core_cells.x.max()
    bar = 100.0
    inset.plot([x1 - bar - 0.04 * (x1 - x0), x1 - 0.04 * (x1 - x0)], [core_cells.y.max()] * 2, color="white", lw=2.2)
    inset.text(
        x1 - 0.04 * (x1 - x0) - bar / 2,
        core_cells.y.max(),
        "100 µm",
        color="white",
        fontsize=8,
        ha="center",
        va="bottom",
    )
    inset.set_title(f"{core}, high-grade cellular rejection", fontsize=9, color="white", pad=3)
    _, connector_1, connector_2 = mark_inset(ax, inset, loc1=2, loc2=3, fc="none", ec=INSET_FRAME, lw=1.0)
    connector_1.set_visible(False)
    connector_2.set_visible(False)


def _draw_slide_legend(ax, counts, lymphoid_colours, other_immune):
    """Legend of the classes present on the slide with their cell numbers."""
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            mfc=_slide_colour(name, lymphoid_colours),
            mec="none",
            ms=8,
            label=f"{SLIDE_NAMES.get(name, name)} ({counts[name]:,})",
        )
        for name in lymphoid_colours
        if counts[name] > 0
    ]
    handles.append(Line2D([0], [0], marker="", ls="", label=""))
    handles += [
        Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            mfc=_slide_colour(name, lymphoid_colours),
            mec="none",
            ms=8,
            alpha=0.55,
            label=f"{SLIDE_NAMES.get(name, name)} ({counts[name]:,})",
        )
        for name in other_immune
        if counts[name] > 0
    ]
    handles.append(Line2D([0], [0], marker="o", ls="", mfc=STRUCTURAL, mec="none", ms=8, label="structural / other"))
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=9,
        handletextpad=0.3,
        labelspacing=0.42,
        title="atlas cell types\n(lymphoid at Level 4)",
        title_fontsize=9.5,
        facecolor=BACKGROUND,
        edgecolor="none",
    )
    for text in legend.get_texts():
        text.set_color("white")
    legend.get_title().set_color("white")


def slide_immune_cells(cells, zoom_core, lymphoid, other_immune):
    """Fig 6 c: the slide with the most cores, immune cells coloured by class, and one core enlarged.

    Structural cells are grey; the lymphoid Level-4 classes are drawn on top of the other immune classes.

    Parameters
    ----------
    cells
        One row per cell (:func:`hhca.tl.spatial_classes`): ``Sample``, ``slide``, ``x``, ``y`` and the class
        ``fig6c``.
    zoom_core
        The core shown enlarged.
    lymphoid
        The lymphoid Level-4 classes that are drawn.
    other_immune
        The other immune classes (Level 3).
    """
    lymphoid_colours = {name: colour for name, colour in style.LYMPHOID.items() if name in set(lymphoid)}
    order = list(other_immune) + list(lymphoid_colours)
    slide = cells.groupby("slide").Sample.nunique().idxmax()
    slide_cells = cells[cells.slide == slide].copy()
    slide_cells["disp"] = slide_cells.fig6c.where(slide_cells.fig6c.notna(), None)
    counts = {name: int((slide_cells.disp == name).sum()) for name in order}
    with dark_panel((6.8, 9.4), PANEL_RC, BACKGROUND) as fig:
        ax = fig.subplots()
        ax.set_facecolor(BACKGROUND)
        _draw_slide_cells(ax, slide_cells, order, lymphoid_colours, 0.7, 1.7)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        _draw_zoom(ax, slide_cells[slide_cells.Sample == zoom_core], zoom_core, order, lymphoid_colours)
        _draw_slide_legend(ax, counts, lymphoid_colours, other_immune)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 6 d, ED9 a: overview of the cores

GRADE_BLUES = ["#deebf7", "#9ecae1", "#4292c6", "#08519c"]
OVERVIEW_REJECTION = {
    "no_rejection": "#cccccc",
    "cellular_rejection": "#3878b0",
    "antibody_rejection": "#B2182B",
    "mixed_rejection": "#8161bc",
}
PAMR = ["pAMR0", "pAMR1-i", "pAMR1-h", "pAMR2"]
PAMR_COLOURS = dict(zip(PAMR, ["#fee6ce", "#fdae6b", "#e6550d", "#8c2d04"]))
OVERVIEW_RESOLUTION = {"resolved": "#4a9d5b", "persistence": "#B2182B", "na": "#efefef"}
REJECTION_TITLES = [
    ("cellular_rejection", "Cellular (ACR)"),
    ("antibody_rejection", "Antibody (AMR)"),
    ("mixed_rejection", "Mixed"),
]


def core_overview(overview, timing):
    """Fig 6 d, ED9 a: clinical tracks, immune fraction, foci and immune composition of the biopsy cores.

    Parameters
    ----------
    overview
        One row per core in drawing order (:func:`hhca.tl.core_overview`).
    timing
        ``pre`` or ``post`` treatment, for the axis label.
    """
    d = overview
    classes = [c[6:] for c in d.columns if c.startswith("comp::")]
    comp = d[[f"comp::{c}" for c in classes]].values
    n = len(d)
    ro = d.prt.map({k: i for i, k in enumerate(REJECTING)}).values
    bnd = np.where(np.diff(ro) != 0)[0] + 0.0
    with panel((15.5, 8.6), PANEL_RC) as fig:
        gs = gridspec.GridSpec(
            10,
            1,
            height_ratios=[0.5, 0.5, 0.5, 0.5, 0.34, 1.5, 1.5, 1.5, 0.34, 2.1],
            hspace=0.10,
            left=0.12,
            right=0.75,
            top=0.96,
            bottom=0.05,
        )

        def strip(ax, vals, cmap, label):
            for i, v in enumerate(vals):
                ax.add_patch(
                    plt.Rectangle(
                        (i, 0), 1, 1, fc=(cmap.get(v, "#efefef") if isinstance(cmap, dict) else cmap[int(v)]), ec="none"
                    )
                )
            ax.set_xlim(0, n)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10)
            for b in bnd:
                ax.axvline(b + 1, color="k", lw=1.1)

        axs = [fig.add_subplot(gs[i]) for i in range(4)]
        strip(axs[0], d.prt.values, OVERVIEW_REJECTION, "Rejection type")
        strip(axs[1], d.g.values, GRADE_BLUES, "ISHLT cellular grade")
        strip(axs[2], d.pamr.values, PAMR_COLOURS, "pAMR grade")
        strip(axs[3], d.res.values, OVERVIEW_RESOLUTION, "Resolution")
        for grp, lab_ in REJECTION_TITLES:
            idx = np.where(d.prt.values == grp)[0]
            if len(idx):
                axs[0].text(
                    (idx.min() + idx.max()) / 2 + 0.5,
                    1.6,
                    lab_,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                    color=OVERVIEW_REJECTION[grp],
                )

        def bars(ax, vals, label, logy=False):
            ax.bar(np.arange(n) + 0.5, vals, width=0.9, color=[GRADE_BLUES[gg] for gg in d.g.values], edgecolor="none")
            ax.set_xlim(0, n)
            ax.set_xticks([])
            if logy:
                ax.set_yscale("log")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_ylabel(label, fontsize=10)
            for b in bnd:
                ax.axvline(b + 1, color="k", lw=1.1)

        bars(fig.add_subplot(gs[5]), d.immune_fraction.values, "Immune\nfraction")
        bars(fig.add_subplot(gs[6]), d.dbscan_frac_in_agg.values, "Frac. immune\nin foci")
        bars(fig.add_subplot(gs[7]), d.largest_cells.values, "Largest focus\n(cells, log)", logy=True)
        axc = fig.add_subplot(gs[9])
        bottom = np.zeros(n)
        for j, c in enumerate(classes):
            axc.bar(
                np.arange(n) + 0.5, comp[:, j], bottom=bottom, width=0.96, color=style.COMPOSITION[c], edgecolor="none"
            )
            bottom += comp[:, j]
        axc.set_xlim(0, n)
        axc.set_ylim(0, 1)
        axc.set_xticks([])
        axc.set_ylabel("Immune\ncomposition", fontsize=10)
        for sp in ("top", "right"):
            axc.spines[sp].set_visible(False)
        for b in bnd:
            axc.axvline(b + 1, color="k", lw=1.1)
        axc.set_xlabel(
            f"{timing}-treatment biopsy cores grouped by rejection type, ordered by ISHLT grade",
            fontsize=10,
            labelpad=3,
        )
        legends = [
            (
                "Rejection type",
                [
                    Patch(fc=OVERVIEW_REJECTION[k], label=v)
                    for k, v in [("no_rejection", "No rejection"), *REJECTION_TITLES]
                ],
                0.99,
            ),
            (
                "Resolution",
                [
                    Patch(fc=OVERVIEW_RESOLUTION["resolved"], label="resolved"),
                    Patch(fc=OVERVIEW_RESOLUTION["persistence"], label="persistence"),
                ],
                0.82,
            ),
            ("ISHLT grade", [Patch(fc=GRADE_BLUES[i], label=str(i)) for i in range(4)], 0.70),
            ("pAMR grade", [Patch(fc=PAMR_COLOURS[k], label=k) for k in PAMR], 0.53),
            ("Immune composition", [Patch(fc=style.COMPOSITION[c], label=c) for c in classes], 0.36),
        ]
        for title, handles, y in legends:
            fs = 9.5 if title == "Immune composition" else 10.5
            lg = fig.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(0.765, y),
                fontsize=fs,
                title=title,
                title_fontsize=11,
                handlelength=1.2,
                handletextpad=0.5,
                labelspacing=0.24 if title == "Immune composition" else 0.3,
                frameon=False,
            )
            fig.add_artist(lg)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 6 e: foci against grade

REJECTION_LEGEND = {
    "no_rejection": "no rejection",
    "cellular_rejection": "cellular",
    "antibody_rejection": "antibody",
    "mixed_rejection": "mixed",
}


def foci_by_grade(cores, rho, p):
    """Fig 6 e: share of immune cells in foci against the ISHLT cellular grade, one point per core.

    Parameters
    ----------
    cores
        Per-core table of :func:`hhca.tl.core_grades`: ``grade``, ``rej`` and ``ff``.
    rho, p
        Spearman correlation of grade and ``ff`` and its p value.
    """
    with panel((5.0, 4.6), STATS_RC) as fig:
        ax = fig.subplots()
        rng = np.random.default_rng(0)
        for rejection in REJECTION_TYPES:
            group = cores[cores.rej == rejection]
            if not len(group):
                continue
            ax.scatter(
                group.grade + rng.normal(0, 0.085, len(group)),
                group.ff,
                s=22,
                color=style.REJECTION[rejection],
                alpha=0.8,
                linewidths=0,
                label=REJECTION_LEGEND[rejection],
            )
        unlabelled = cores[cores.rej.isna()]
        if len(unlabelled):
            ax.scatter(
                unlabelled.grade + rng.normal(0, 0.085, len(unlabelled)),
                unlabelled.ff,
                s=22,
                color="#B9B9B9",
                alpha=0.8,
                linewidths=0,
            )
        grade_median = cores.groupby("grade").ff.median()
        ax.plot(grade_median.index, grade_median.values, color=INK, lw=1.4, zorder=5, label="grade median")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["0R", "1R", "2R", "3R"])
        ax.set_xlabel("ISHLT cellular grade")
        ax.set_ylabel("fraction of immune cells in foci")
        ax.text(0.03, 0.97, f"ρ={rho:.2f} {stars(p)}", transform=ax.transAxes, va="top", fontsize=11)
        ax.set_title("Immune clustering in foci versus cellular grade", fontsize=12.5, pad=10)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right", handletextpad=0.4, labelspacing=0.35)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 6 f, g: before and after treatment


def pre_post(pairs, tests, column, ylabel):
    """Fig 6 f, g: a fraction before and after treatment, one line per patient, by outcome of the rejection.

    Parameters
    ----------
    pairs
        Paired values per patient (:func:`hhca.tl.pre_post_pairs`).
    tests
        Outcome -> Wilcoxon test of this fraction (``{"resolved": {"p": ...}, "persistence": {...}}``).
    column
        ``immf`` (immune fraction, Fig 6 f) or ``tf`` (T-cell fraction, Fig 6 g).
    ylabel
        Name of the fraction.
    """
    with panel((5.2, 4.9), STATS_RC) as fig:
        ax = fig.subplots()
        for group, (x_pre, x_post) in [("resolved", (0, 1)), ("persistence", (2.6, 3.6))]:
            colour = style.RESOLUTION[group]
            patients = pairs[pairs.res == group]
            pre, post = patients[f"{column}_pre"].values, patients[f"{column}_post"].values
            p = tests[group]["p"]
            for value_pre, value_post in zip(pre, post):
                ax.plot([x_pre, x_post], [value_pre, value_post], color=colour, lw=0.9, alpha=0.45, zorder=1)
            ax.scatter(np.full(len(pre), x_pre), pre, s=24, color=colour, alpha=0.85, lw=0, zorder=3)
            ax.scatter(np.full(len(post), x_post), post, s=24, color=colour, alpha=0.85, lw=0, zorder=3)
            for x, values in ((x_pre, pre), (x_post, post)):
                ax.plot([x - 0.16, x + 0.16], [np.median(values)] * 2, color=INK, lw=2.2, zorder=4)
            ax.text(
                (x_pre + x_post) / 2,
                1.02,
                f"{group} (n={len(pre)})\nΔp={p:.2g} {stars(p)}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=10,
                color=colour,
            )
        ax.set_xticks([0, 1, 2.6, 3.6])
        ax.set_xticklabels(["pre", "post", "pre", "post"])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.capitalize()}: resolved vs persistence", fontsize=12.5, pad=30)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 6 h, i, ED9 b, c: cytokine programmes by rejection type


def programme_heatmap(zscores, tests):
    """Fig 6 h: medians of the cytokine programmes per rejection type, z-scored per programme.

    Stars give the q value of each rejecting type against no rejection.

    Parameters
    ----------
    zscores, tests
        The two tables of :func:`hhca.tl.programmes_vs_no_rejection`.
    """
    q_values = {(row.programme, row.group): row.q for row in tests.itertuples()}
    programmes = list(zscores.index)
    with panel((4.8, 6.6), STATS_RC) as fig:
        ax = fig.subplots()
        vmax = float(np.nanmax(np.abs(zscores.values)))
        image = ax.imshow(zscores.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(4))
        ax.set_xticklabels(SHORT_TYPES, rotation=40, ha="right")
        ax.set_yticks(range(len(programmes)))
        ax.set_yticklabels([programme.replace("score_", "") for programme in programmes])
        for i, programme in enumerate(programmes):
            for j, group in enumerate(REJECTING, start=1):
                mark = stars(q_values[(programme, group)])
                ax.text(
                    j,
                    i,
                    mark,
                    ha="center",
                    va="center",
                    fontsize=9 if mark != "ns" else 7,
                    color="white" if abs(zscores.values[i, j]) > 0.62 * vmax else INK,
                    fontweight="bold" if mark != "ns" else "normal",
                )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
        colorbar.set_label("z-score", fontsize=10)
        ax.set_title("Cytokine-response programs by rejection type", fontsize=12, pad=10)
    return fig


def _programme_boxplot(ax, scores, tests, programme, ylabel=True, seed=0):
    """Boxplot of one programme over the rejection types with the significant pairs marked."""
    k = f"score_{programme}"
    v = [scores.loc[scores.rej == g, k].values for g in REJECTION_TYPES]
    kruskal_p, kruskal_q = tests.loc[k, "kruskal_p"], tests.loc[k, "kruskal_q_BH_over_18"]
    pair_q = [tests.loc[k, f"q_{SHORT_TYPES[i]}_vs_{SHORT_TYPES[j]}"] for i, j in PAIRS]
    rng = np.random.default_rng(seed)
    bp = ax.boxplot(
        v, positions=range(4), widths=0.6, patch_artist=True, showfliers=False, medianprops={"color": INK, "lw": 1.5}
    )
    for pt_, g in zip(bp["boxes"], REJECTION_TYPES):
        pt_.set_facecolor(style.REJECTION[g])
        pt_.set_alpha(0.55)
        pt_.set_edgecolor(MUTED)
    for i, z in enumerate(v):
        ax.scatter(
            rng.normal(i, 0.07, len(z)), z, s=13, color=style.REJECTION[REJECTION_TYPES[i]], alpha=0.7, lw=0, zorder=3
        )
    ymx = max(np.max(z) for z in v)
    ymn = min(np.min(z) for z in v)
    ry = ymx - ymn
    sig = sorted([(i, j, q) for (i, j), q in zip(PAIRS, pair_q) if q < 0.05], key=lambda t: (t[1] - t[0], t[0]))
    lvl = 0
    for i, j, q_ in sig:
        y = ymx + 0.07 * ry + lvl * 0.105 * ry
        ax.plot([i, i, j, j], [y, y + 0.022 * ry, y + 0.022 * ry, y], color=INK, lw=1.1)
        ax.text((i + j) / 2, y + 0.028 * ry, stars(q_), ha="center", va="bottom", fontsize=10)
        lvl += 1
    top_br = ymx + 0.07 * ry + (max(lvl - 1, 0)) * 0.105 * ry + (0.085 * ry if lvl else 0)
    ax.set_ylim(ymn - 0.06 * ry, max(top_br, ymx + 0.07 * ry) + 0.20 * ry)
    ax.text(
        0.02,
        0.985,
        f"Kruskal p={kruskal_p:.1e}, q={kruskal_q:.1e} {stars(kruskal_q)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    ax.set_xlim(-0.62, 3.62)
    ax.set_xticks(range(4))
    ax.set_xticklabels(SHORT_TYPES)
    ax.set_title(programme_name(programme), fontsize=12.5, pad=8)
    if ylabel:
        ax.set_ylabel("programme score")


def programme_scores(scores, tests, programmes, shape=None):
    """Fig 6 i, ED9 b-c: cytokine-programme scores of the biopsies by rejection type.

    One box per rejection type with the biopsies as points, the Kruskal-Wallis p and q, and brackets for the
    pairwise comparisons with q < 0.05.

    Parameters
    ----------
    scores
        Per-biopsy table with ``rej`` and the score columns ``score_<programme>``.
    tests
        Table of :func:`hhca.tl.programmes_by_rejection`, indexed by ``column``.
    programmes
        One programme (Fig 6 i) or a list of programmes (ED9 b, c).
    shape
        Rows and columns of the grid for a list of programmes.
    """
    if isinstance(programmes, str):
        with panel((3.9, 3.95), STATS_RC) as fig:
            _programme_boxplot(fig.subplots(), scores, tests, programmes)
        return fig
    n_rows, n_cols = shape
    with panel((3.9 * n_cols, 3.95 * n_rows), STATS_RC) as fig:
        axes = fig.subplots(n_rows, n_cols)
        for index, (programme, ax) in enumerate(zip(programmes, axes.ravel())):
            _programme_boxplot(ax, scores, tests, programme, ylabel=(index % n_cols == 0), seed=index)
        fig.tight_layout(h_pad=2.4, w_pad=2.0)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 6 k, l: drug targets

DRUG_RC = {**PANEL_RC, "axes.titlesize": 13, "axes.labelsize": 11, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5}


def drug_targets(zscores):
    """Fig 6 k: drug-target scores per cell type, z-scored per drug over the cell types.

    Parameters
    ----------
    zscores
        Drugs x cell types (:func:`hhca.tl.group_zscores`).
    """
    values, drugs, cell_types = zscores.values, list(zscores.index), list(zscores.columns)
    vmax = np.nanmax(np.abs(values))
    with panel((7.6 * len(cell_types) / 18 + 0.4, 4.3), DRUG_RC) as fig:
        ax = fig.subplots()
        image = ax.imshow(values, aspect="auto", cmap=style.DIVERGING, vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(cell_types)))
        ax.set_xticklabels(cell_types, rotation=40, ha="right")
        ax.set_yticks(range(len(drugs)))
        ax.set_yticklabels(drugs)
        ax.set_xticks(np.arange(-0.5, len(cell_types), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(drugs), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.0)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title("Immunosuppressant drug-target expression by cell type", fontsize=13, pad=10)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
        colorbar.set_label("z-score", fontsize=10.5)
        colorbar.ax.tick_params(labelsize=9.5)
    return fig


def atg_score(biopsies, column, xlabel, rho, p):
    """Fig 6 l: ATG drug-target score of the biopsies against a cell fraction, coloured by ISHLT grade.

    Parameters
    ----------
    biopsies
        Per-biopsy table (:func:`hhca.tl.biopsy_cohort`) with ``atg`` and ``grade``.
    column, xlabel
        The fraction (``immf`` or ``tf``) and its name.
    rho, p
        Spearman correlation of the fraction with the score and its p value.
    """
    complete = biopsies.dropna(subset=[column, "atg"])
    with panel((4.6, 4.4), STATS_RC) as fig:
        ax = fig.subplots()
        for grade in sorted(complete.grade.dropna().unique()):
            group = complete[complete.grade == grade]
            ax.scatter(
                group[column],
                group.atg,
                s=34,
                color=style.GRADE.get(int(grade), "#CCC"),
                alpha=0.85,
                linewidths=0,
                label=f"{int(grade)}R",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ATG drug-target score")
        ax.text(0.03, 0.97, f"ρ={rho:.2f} {stars(p)}", transform=ax.transAxes, va="top", fontsize=11)
        ax.set_title(f"ATG target vs {xlabel}", fontsize=12.5, pad=10)
        ax.legend(frameon=False, fontsize=9, title="ISHLT grade", title_fontsize=9, loc="lower right")
    return fig


# ---------------------------------------------------------------------------------------------------------------
# ED6 i: healthy section

HEALTHY_PAGE = (503.748, 481.3632)
HEALTHY_AX_MAIN = (17.0017334906, 7.2, 179.1165330187, 453.456)
HEALTHY_AX_ZOOM = (228.276, 263.97, 173.88, 173.88)
HEALTHY_ROI_PT = (52.74848, 295.052365, 103.259915, 345.5638)
HEALTHY_BAR_MAIN_PT = (28.667448, 58.034561, 33.097717)
HEALTHY_BAR_ZOOM_PT = (373.849953, 394.068558, 272.057442)
HEALTHY_LEGEND_PT = (400.372, 246.856)
HEALTHY_S_MAIN = (0.6, 1.15)
HEALTHY_S_ZOOM = (4.25, 7.5)
HEALTHY_LEGEND_MS, HEALTHY_LEGEND_SPACING = 4.32, 0.52
HEALTHY_RC = {**PANEL_RC, "savefig.pad_inches": 0.0, "legend.handlelength": 1.0, "savefig.bbox": None}


def _on_dark(hex_colour, sat=1.45, val=1.18):
    """A palette colour with more saturation and brightness, for a black background."""
    r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return colorsys.hsv_to_rgb(h, min(1.0, s * sat), min(1.0, v * val))


def _draw_classes(ax, xy, classes, masks, palette, sizes):
    """Plot the cells class by class; the most frequent class (first) gets the smaller marker `sizes[0]`."""
    for name in classes:
        area = sizes[0] if name == classes[0] else sizes[1]
        mask = masks[name]
        ax.plot(
            xy[mask, 0], xy[mask, 1], marker="o", ls="", ms=area**0.5, mfc=palette[name], mec="none", rasterized=True
        )


def healthy_section(cells, negative_label):
    """ED6 i: the healthy section coloured by display class, with a vascular field enlarged.

    The geometry of the page, the enlarged field and the scale bars are given in page points.

    Parameters
    ----------
    cells
        One row per cell: ``x``, ``y`` (um) and ``display`` (:func:`hhca.tl.endothelial_gate`).
    negative_label
        The display class of cells with an endothelial label that fail the marker gate (drawn grey).
    """
    xy = cells[["x", "y"]].to_numpy(float)
    label = cells.display.to_numpy(dtype=object)

    classes = list(pd.Series(label).value_counts().index)
    palette = {name: _on_dark(ct(name)) for name in classes}
    palette[negative_label] = (0.42, 0.42, 0.42)
    member = {name: label == name for name in classes}

    page_width, page_height = HEALTHY_PAGE
    ax_main, ax_zoom, roi = HEALTHY_AX_MAIN, HEALTHY_AX_ZOOM, HEALTHY_ROI_PT
    margin_x, margin_y = 0.05 * np.ptp(xy[:, 0]), 0.05 * np.ptp(xy[:, 1])
    xlim = (xy[:, 0].min() - margin_x, xy[:, 0].max() + margin_x)
    ylim = (xy[:, 1].max() + margin_y, xy[:, 1].min() - margin_y)
    pt_per_um = ax_main[2] / (xlim[1] - xlim[0])

    def to_data(x_pt, y_pt):
        """Data coordinates (um) of a position given in page points."""
        return xlim[0] + (x_pt - ax_main[0]) / pt_per_um, ylim[0] - (y_pt - ax_main[1]) / pt_per_um

    rx0, ry1 = to_data(roi[0], roi[1])
    rx1, ry0 = to_data(roi[2], roi[3])
    in_roi = (xy[:, 0] >= rx0) & (xy[:, 0] <= rx1) & (xy[:, 1] >= ry0) & (xy[:, 1] <= ry1)

    with dark_panel((page_width / 72, page_height / 72), HEALTHY_RC, "black", save_dpi=400) as fig:
        fig.patches.append(
            Rectangle((0, 0), 1, 1, transform=fig.transFigure, facecolor="black", edgecolor="none", zorder=-10)
        )
        ax = fig.add_axes(
            [ax_main[0] / page_width, ax_main[1] / page_height, ax_main[2] / page_width, ax_main[3] / page_height],
            facecolor="black",
        )
        _draw_classes(ax, xy, classes, member, palette, HEALTHY_S_MAIN)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False, edgecolor="white", linewidth=1.0))
        bar_x0, bar_y = to_data(HEALTHY_BAR_MAIN_PT[0], HEALTHY_BAR_MAIN_PT[2])
        bar_x1, _ = to_data(HEALTHY_BAR_MAIN_PT[1], HEALTHY_BAR_MAIN_PT[2])
        ax.plot([bar_x0, bar_x1], [bar_y, bar_y], color="white", lw=2.2, solid_capstyle="butt")
        fig.text(
            31.187 / page_width, 38.012 / page_height, "500 µm", color="white", fontsize=7.2, ha="left", va="baseline"
        )

        zoom = fig.add_axes(
            [ax_zoom[0] / page_width, ax_zoom[1] / page_height, ax_zoom[2] / page_width, ax_zoom[3] / page_height],
            facecolor="black",
        )
        _draw_classes(zoom, xy, classes, {name: member[name] & in_roi for name in classes}, palette, HEALTHY_S_ZOOM)
        zoom.set_xlim(rx0, rx1)
        zoom.set_ylim(ry1, ry0)
        zoom.set_xticks([])
        zoom.set_yticks([])
        for name in ("top", "right"):
            zoom.spines[name].set_visible(False)
        for name in ("left", "bottom"):
            zoom.spines[name].set_color("white")
            zoom.spines[name].set_linewidth(1.0)
        zoom.set_title("Zoom, vascular niche (single cells)", color="white", fontsize=7.6, pad=3.0)
        zoom_pt_per_um = ax_zoom[2] / (rx1 - rx0)
        zoom_bar_x0 = rx0 + (HEALTHY_BAR_ZOOM_PT[0] - ax_zoom[0]) / zoom_pt_per_um
        zoom_bar_x1 = rx0 + (HEALTHY_BAR_ZOOM_PT[1] - ax_zoom[0]) / zoom_pt_per_um
        zoom_bar_y = ry1 - (HEALTHY_BAR_ZOOM_PT[2] - ax_zoom[1]) / zoom_pt_per_um
        zoom.plot([zoom_bar_x0, zoom_bar_x1], [zoom_bar_y, zoom_bar_y], color="white", lw=2.2, solid_capstyle="butt")
        fig.text(
            372.639 / page_width, 277.087 / page_height, "100 µm", color="white", fontsize=6.8, ha="left", va="baseline"
        )
        for y_roi, y_zoom in ((roi[1], ax_zoom[1]), (roi[3], ax_zoom[1] + ax_zoom[3])):
            fig.add_artist(
                Line2D(
                    [roi[2] / page_width, ax_zoom[0] / page_width],
                    [y_roi / page_height, y_zoom / page_height],
                    transform=fig.transFigure,
                    color="white",
                    lw=0.7,
                    ls=(0, (4, 2)),
                    solid_capstyle="round",
                )
            )
        fig.text(
            7.2 / page_width,
            469.06945 / page_height,
            "Xenium single cells, healthy heart, CellPin annotation (HHCA Level 3)",
            color="white",
            fontsize=7.4,
            fontweight="bold",
            ha="left",
            va="baseline",
        )
        fig.legend(
            handles=[
                Line2D([], [], marker="o", ls="", ms=HEALTHY_LEGEND_MS, color=palette[name], label=name)
                for name in classes
            ],
            loc="upper left",
            bbox_to_anchor=(HEALTHY_LEGEND_PT[0] / page_width, HEALTHY_LEGEND_PT[1] / page_height),
            frameon=False,
            fontsize=6.2,
            labelcolor="white",
            labelspacing=HEALTHY_LEGEND_SPACING * 0.8,
            borderaxespad=0.0,
        )
    return fig


# ---------------------------------------------------------------------------------------------------------------
# ED6 j: two cores of the rejection cohort

LINEAGE_RC = {**PANEL_RC, "savefig.bbox": None}
LINEAGE_PAGE_ASPECT = 308 / 452
LINEAGE_WIDTH = 5.68
LINEAGE_HEIGHT = LINEAGE_WIDTH / LINEAGE_PAGE_ASPECT
LINEAGE_FREE_HEIGHT = 0.22
LINEAGE_LEFT = 0.05
LINEAGE_TOP = 0.90


def _draw_core(fig, cells, rect, title_y):
    """One core on a black axes at `rect`: structural cells in grey, immune lineages in colour, 100 um bar."""
    immune = cells.ed6j != "structural / other"
    grade = int(cells.core_grade.iloc[0])
    fig.text(
        0.075,
        title_y,
        f"ISHLT grade {grade}R, immune {100 * immune.mean():.0f}% (n={len(cells):,})",
        color="white",
        fontsize=9.2,
        ha="left",
        va="center",
    )
    ax = fig.add_axes(rect, facecolor="black")
    structural = cells[~immune]
    ax.scatter(structural.x, structural.y, s=2.2, c="#6a6a6a", linewidths=0, rasterized=True, zorder=1)
    for lineage in list(style.LINEAGE)[:-1]:
        subset = cells[cells.ed6j == lineage]
        ax.scatter(subset.x, subset.y, s=7.0, c=style.LINEAGE[lineage], linewidths=0, rasterized=True, zorder=2)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    x0, x1 = cells.x.min(), cells.x.max()
    y0, y1 = cells.y.min(), cells.y.max()
    ax.set_xlim(x0 - 0.08 * (x1 - x0), x1 + 0.08 * (x1 - x0))
    ax.set_ylim(y1 + 0.10 * (y1 - y0), y0 - 0.06 * (y1 - y0))
    bar_x, bar_y = x0 - 0.02 * (x1 - x0), y1 + 0.06 * (y1 - y0)
    ax.plot([bar_x, bar_x + 100], [bar_y, bar_y], color="white", lw=2.4, solid_capstyle="butt", zorder=5)
    ax.text(bar_x + 50, bar_y - 0.015 * (y1 - y0), "100 µm", color="white", fontsize=7.5, ha="center", va="bottom")


def core_lineages(cells, cores):
    """ED6 j: immune lineages on two biopsy cores of different ISHLT cellular grade.

    Parameters
    ----------
    cells
        One row per cell (:func:`hhca.tl.spatial_classes`): ``Sample``, ``x``, ``y``, ``core_grade`` and the
        lineage ``ed6j``.
    cores
        The two cores, top to bottom.
    """
    cells = cells[cells.Sample.isin(cores)]
    # the two axes share one width; their heights follow the aspect ratios of the cores
    aspect = {}
    for core in cores:
        core_cells = cells[cells.Sample == core]
        x0, x1 = core_cells.x.min(), core_cells.x.max()
        y0, y1 = core_cells.y.min(), core_cells.y.max()
        aspect[core] = (y1 - y0) / (x1 - x0)
    aspect_1, aspect_2 = aspect[cores[0]], aspect[cores[1]]
    height_1 = (1 - LINEAGE_FREE_HEIGHT) / (1 + aspect_2 / aspect_1)
    height_2 = (1 - LINEAGE_FREE_HEIGHT) - height_1
    axes_width = height_1 / ((LINEAGE_WIDTH / LINEAGE_HEIGHT) * aspect_1)
    top_2 = LINEAGE_TOP - height_1 - 0.06
    left = LINEAGE_LEFT
    rects = [[left, LINEAGE_TOP - height_1, axes_width, height_1], [left, top_2 - height_2, axes_width, height_2]]
    title_y = [LINEAGE_TOP + 0.030, top_2 + 0.030]

    with dark_panel((LINEAGE_WIDTH, LINEAGE_HEIGHT), LINEAGE_RC, "black") as fig:
        fig.text(
            0.04,
            0.965,
            "Rejection Xenium, immune infiltrate by cellular grade",
            color="white",
            fontsize=10.5,
            ha="left",
            va="center",
        )
        for core, rect, y in zip(cores, rects, title_y):
            _draw_core(fig, cells[cells.Sample == core], rect, y)
        handles = [
            Line2D([0], [0], marker="o", ls="", mfc=colour, mec="none", ms=6.5, label=name)
            for name, colour in style.LINEAGE.items()
        ]
        fig.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(left + axes_width + 0.05, LINEAGE_TOP),
            frameon=False,
            fontsize=8.2,
            labelcolor="white",
            title="immune lineage",
            title_fontsize=8.6,
            labelspacing=0.55,
            handletextpad=0.5,
            borderaxespad=0,
        )
        fig.legends[-1].get_title().set_color("white")
    return fig
