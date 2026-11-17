"""Panels of the atlas donor analysis: sex and hormones (Fig 2 h-l) and the hormone relay (Fig 4 l)."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from hhca.pl import style
from hhca.pl._panel import panel
from hhca.stats import stars
from hhca.tl._hormones import cell_family

# ---------------------------------------------------------------------------------------------------------------
# Fig 2 h, i: sex differences in hormone reception and production per cell system

SYSTEM_LABEL = {
    "vascular": "Vascular",
    "macrophage": "Myeloid/Mac",
    "cardiomyocyte": "Cardiomyocyte",
    "neural": "Neural",
}
MARKER_AREA = 260.0  # marker area (pt^2) at mean score 1
DOTS_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "axes.titleweight": "regular",
    "axes.labelcolor": style.INK,
    "text.color": style.INK,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "xtick.color": style.INK,
    "ytick.color": style.INK,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "patch.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.handlelength": 1.2,
    "legend.handletextpad": 0.5,
    "legend.borderpad": 0.2,
    "legend.labelspacing": 0.3,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": False,
}
# title, figure size and axes position of the two panels; the production panel carries the colour bar and legend
DOTS_LAYOUT = {
    "received": {
        "title": "Reception (receptor expr.)",
        "figsize": (6.7, 4.3),
        "adjust": {"left": 0.13, "right": 0.99, "top": 0.9, "bottom": 0.42},
        "legend": False,
    },
    "produced": {
        "title": "Production (biosynthesis expr.)",
        "figsize": (8.2, 4.3),
        "adjust": {"left": 0.10, "right": 0.85, "top": 0.9, "bottom": 0.42},
        "legend": True,
    },
}


def _add_colorbar_and_size_legend(fig, cmap, norm):
    """Colour bar of the sex difference and the legend of marker size and outline, right of the dots."""
    cax = fig.add_axes([0.905, 0.60, 0.014, 0.28])
    cb = fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), cax=cax)
    cb.set_ticks([-0.02, 0.0, 0.02])
    cb.ax.tick_params(labelsize=7)
    cb.set_label("female − male (per-donor score)", fontsize=7.2)
    cb.ax.text(
        0.5,
        1.06,
        "female↑",
        transform=cb.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=style.SEX["female"],
        fontweight="bold",
    )
    cb.ax.text(
        0.5,
        -0.06,
        "male↑",
        transform=cb.ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color=style.SEX["male"],
        fontweight="bold",
    )
    handles = [
        Line2D([0], [0], marker="o", ls="", mfc="#D9D9D9", mec="none", ms=np.sqrt(MARKER_AREA * v), label=f"{v:.2f}")
        for v in (0.17, 0.33, 0.67)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            mfc="#D9D9D9",
            mec="#111111",
            mew=1.3,
            ms=np.sqrt(MARKER_AREA * 0.33),
            label="p<0.05",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.875, 0.50),
        frameon=False,
        fontsize=7.2,
        title="mean score",
        title_fontsize=7.6,
        labelspacing=1.0,
        handletextpad=0.8,
        borderaxespad=0,
    )


def hormone_dots(sex_differences, gate, side="received"):
    """Fig 2 h, i: sex difference of every shown hormone per cell system, as a dot matrix.

    Systems in rows, hormones in columns ordered by their relevance. Colour: female minus male mean per-donor
    score; size: mean of the two; outline: OLS ``score ~ C(sex) + age`` with p < 0.05.

    Parameters
    ----------
    sex_differences
        Table of :func:`hhca.tl.sex_differences` for both panels (the colour scale is shared).
    gate
        Gate table of this panel (:func:`hhca.tl.relevance_gate`).
    side
        ``"received"`` (Fig 2 h) or ``"produced"`` (Fig 2 i, with the colour bar and the size legend).
    """
    layout = DOTS_LAYOUT[side]
    systems = list(SYSTEM_LABEL)
    significance = sex_differences.set_index(["panel", "system", "hormone"])
    difference = (sex_differences.mean_female - sex_differences.mean_male).abs()
    vmax = max(0.03, float(np.nanpercentile(difference, 80)))
    norm = Normalize(-vmax, vmax)
    cmap = plt.get_cmap(style.DIVERGING)
    shown = gate[gate.shown_in_figure].copy()
    hormones = list(shown.groupby("hormone")["mean_or_max_strength_R"].max().sort_values(ascending=False).index)
    shown = shown.set_index(["system", "hormone"])
    with panel(layout["figsize"], DOTS_RC) as fig:
        ax = fig.subplots()
        fig.subplots_adjust(**layout["adjust"])
        for yi, system in enumerate(systems):
            for xi, hormone in enumerate(hormones):
                if (system, hormone) not in shown.index:
                    continue
                key = (side, system, hormone)
                mean_male = float(significance.loc[key, "mean_male"]) if key in significance.index else np.nan
                mean_female = float(significance.loc[key, "mean_female"]) if key in significance.index else np.nan
                p = float(significance.loc[key, "ols_p"]) if key in significance.index else 1.0
                if not np.isfinite(mean_male + mean_female):
                    continue
                ax.scatter(
                    [xi],
                    [yi],
                    s=MARKER_AREA * (mean_male + mean_female) / 2,
                    c=[cmap(norm(mean_female - mean_male))],
                    edgecolors="#111111" if p < 0.05 else "none",
                    linewidths=1.3 if p < 0.05 else 0,
                    zorder=3,
                )
        ax.set_xlim(-0.6, len(hormones) - 0.4)
        ax.set_ylim(len(systems) - 0.4, -0.6)
        ax.set_xticks(range(len(hormones)))
        ax.set_xticklabels(hormones, rotation=55, ha="right", fontsize=7.4, rotation_mode="anchor")
        ax.set_yticks(range(len(systems)))
        ax.set_yticklabels([SYSTEM_LABEL[s] for s in systems], fontsize=9.0)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(layout["title"], fontsize=9.0, pad=8, color=style.INK)
        if layout["legend"]:
            _add_colorbar_and_size_legend(fig, cmap, norm)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 2 j, k, l: per-donor gene-set scores by sex

SCORE_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "savefig.bbox": "tight",
}
SCORE_INK, SCORE_MUTED, WHISKER = "#222222", "#7A7A7A", "#8a8a8a"


def _despine(ax):
    """Hide the top and right spines and darken the other two."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#4d4d4d")
        ax.spines[s].set_linewidth(1.0)


def sex_boxes(female, male, title, p, g, seed=0):
    """Fig 2 j, l: a per-donor score of the female and the male donors as boxes with jittered points.

    Parameters
    ----------
    female, male
        Scores of the adult female and male donors (:func:`hhca.tl.split_by_sex`).
    title
        Title of the panel.
    p, g
        Mann-Whitney p value and Hedges g (female minus male) written into the panel.
    seed
        Random seed of the jitter.
    """
    rng = np.random.default_rng(seed)  # jitter of the points
    data, names = [female, male], ["female", "male"]
    with panel((5.2, 5.9), SCORE_RC) as fig:
        ax = fig.subplots()
        bp = ax.boxplot(data, positions=[0, 1], widths=0.52, patch_artist=True, showfliers=False, whis=1.5, zorder=2)
        for i, name in enumerate(names):
            col = style.SEX[name]
            bp["boxes"][i].set(facecolor=matplotlib.colors.to_rgba(col, 0.22), edgecolor=col, linewidth=1.3)
            bp["medians"][i].set(color="#111111", linewidth=2.4)
            for whisker in (bp["whiskers"][2 * i], bp["whiskers"][2 * i + 1]):
                whisker.set(color=WHISKER, linewidth=1.2)
            for cap in (bp["caps"][2 * i], bp["caps"][2 * i + 1]):
                cap.set(color=WHISKER, linewidth=1.2)
        for i, (vals, name) in enumerate(zip(data, names)):
            jitter = (rng.random(len(vals)) - 0.5) * 0.30
            ax.scatter(
                np.full(len(vals), i) + jitter, vals, s=26, color=style.SEX[name], alpha=0.80, linewidths=0.0, zorder=3
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Female\n(n={len(female)})", f"Male\n(n={len(male)})"], fontsize=13, color=SCORE_INK)
        ax.set_xlim(-0.7, 1.7)
        ax.set_ylabel("per-donor gene-set score", fontsize=13, color=SCORE_INK)
        ax.set_title(title, fontsize=15.5, color=SCORE_INK, pad=14)
        ax.tick_params(axis="y", labelsize=11.5)
        ax.margins(y=0.06)
        _despine(ax)
        txt = f"Mann-Whitney  p = {p:.1e}  {stars(p)}\nHedges g = {g:+.2f}  (female-higher)"
        ax.text(
            0.5,
            0.985,
            txt,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12.5,
            color=SCORE_INK,
            linespacing=1.5,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f0f0f0", "edgecolor": "#c0c0c0", "linewidth": 1.0},
        )
        fig.tight_layout()
    return fig


def sex_by_age(band_scores, bands, p_values, title):
    """Fig 2 k: mean and SEM of a per-donor score per age band and sex.

    Parameters
    ----------
    band_scores
        One (female scores, male scores) pair per age band (:func:`hhca.tl.split_by_sex`).
    bands
        The age bands as (from, to) pairs.
    p_values
        Mann-Whitney p value of female against male per band (NaN where it was not tested), shown as stars.
    title
        Title of the panel.
    """
    band_labels = [f"{lo}-{hi}" for lo, hi in bands]
    x = np.arange(len(bands))
    series = {}
    for name in ("female", "male"):
        means, sems = [], []
        for female, male in band_scores:
            v = female if name == "female" else male
            means.append(np.mean(v))
            sems.append(np.std(v, ddof=1) / np.sqrt(len(v)))
        series[name] = (np.array(means), np.array(sems))
    with panel((6.6, 5.9), SCORE_RC) as fig:
        ax = fig.subplots()
        for name in ("female", "male"):
            mu, se = series[name]
            ax.errorbar(
                x,
                mu,
                yerr=se,
                color=style.SEX[name],
                lw=2.6,
                marker="o",
                ms=8,
                mfc=style.SEX[name],
                mec="white",
                mew=0.8,
                capsize=4,
                capthick=1.6,
                elinewidth=1.6,
                zorder=3,
                label=name.capitalize(),
            )
        female_mu, female_se = series["female"]
        male_mu, male_se = series["male"]
        ymax = float(np.nanmax([(female_mu + female_se).max(), (male_mu + male_se).max()]))
        for i, p in enumerate(p_values):
            s = stars(p) if p == p else "ns"
            ytop = max(female_mu[i] + female_se[i], male_mu[i] + male_se[i]) + 0.012
            ax.text(
                i, ytop, s, ha="center", va="bottom", fontsize=16 if s != "ns" else 14, weight="bold", color=SCORE_INK
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{label}\nF{len(female)}/M{len(male)}" for label, (female, male) in zip(band_labels, band_scores)],
            fontsize=12,
            color=SCORE_INK,
        )
        ax.set_xlim(-0.5, len(bands) - 0.5)
        ax.set_ylim(0, ymax * 1.16)
        ax.set_xlabel("Age band (years)", fontsize=13, color=SCORE_INK)
        ax.set_ylabel("per-donor gene-set score", fontsize=13, color=SCORE_INK)
        ax.set_title(title, fontsize=15.5, color=SCORE_INK, pad=12)
        ax.tick_params(axis="y", labelsize=11.5)
        _despine(ax)
        ax.legend(loc="upper left", fontsize=12.5, frameon=False, handletextpad=0.6, borderpad=0.2, labelspacing=0.5)
        ax.text(
            0.985,
            0.02,
            "mean ± SEM per band\nstars: Mann-Whitney F vs M",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.5,
            color=SCORE_MUTED,
            style="italic",
            linespacing=1.4,
        )
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 4 l: hormone relay, producing cell states -> hormones -> receiving cell states

RELAY_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
X_PRODUCER, X_HORMONE, X_RECEIVER = 0.0, 1.0, 2.0  # x positions of the three columns
RELAY = {
    "figsize": (19.0, 3.9),
    "producer_face": "#74add1",
    "producer_edge": "#4575b4",
    "receiver_face": "#66bd63",  # receiving-cell dots of a family without a colour
    "receiver_edge": "#1a9850",
    "hormone_face": "#fff3cf",
    "hormone_edge": "#d9a441",
    "hormone_text": "#5c4400",
    "edge_color": "#7f7f7f",  # producing side
    "node_size": 42,
    "edge_lw_range": (0.4, 3.0),
    "edge_alpha_range": (0.12, 0.85),
    "curve_rad": 0.10,
    "label_fontsize": 6.5,
    "hormone_fontsize": 7.2,
    "header_fontsize": 11.0,
}


def _family_grouped(cells, strength_max, families, gap=0.9, lo=0.02, hi=0.98):
    """Cells grouped by family (config order) and, within a family, by decreasing strength.

    Family blocks are separated by a gap. Returns the ordered cells and their y positions (name -> y).
    """
    family_order = [*families, "Other"]
    family_of = {c: cell_family(c, families) for c in cells}

    def sort_key(c):
        return (family_order.index(family_of[c]), -float(strength_max.get(c, 0.0)))

    ordered = sorted(cells, key=sort_key)
    # slot positions, with a gap wherever the family changes
    pos, cur, prev = [], 0.0, None
    for c in ordered:
        family = family_of[c]
        if prev is not None and family != prev:
            cur += gap
        pos.append(cur)
        cur += 1.0
        prev = family
    pos = np.asarray(pos, float)
    span = pos.max() if pos.max() > 0 else 1.0
    ys = hi - (pos / span) * (hi - lo)
    return ordered, dict(zip(ordered, ys))


def _order_hormones(hormones, receiver_best, receiver_edges, receiver_cells):
    """Hormones ordered by the mean position of their receiving cells; ties keep the stronger hormone first."""
    by_strength = sorted(hormones, key=lambda h: receiver_best.get(h, 0.0), reverse=True)
    cell_pos = {c: i for i, c in enumerate(receiver_cells)}
    receivers_of = receiver_edges.groupby("Hormone")["Celltype"].apply(list).to_dict()

    def mean_receiver_pos(h):
        positions = [cell_pos[c] for c in receivers_of.get(h, []) if c in cell_pos]
        return np.mean(positions) if positions else 1e9

    return sorted(by_strength, key=mean_receiver_pos)


def _unit_scale(series):
    """Function scaling a strength to 0-1 over the range of `series` (0.5 when there are no edges)."""
    if len(series) == 0:
        return lambda v: 0.5
    lo, hi = float(series.min()), float(series.max())
    rng = (hi - lo) or 1.0
    return lambda v: (float(v) - lo) / rng


def _between(limits, t):
    """Value at the share t of a (low, high) range."""
    return limits[0] + (limits[1] - limits[0]) * t


def _draw_relay_edges(ax, producer_edges, receiver_edges, y_producer, y_hormone, y_receiver, receiver_family):
    """Grey edges from the producing states to the hormones, family-coloured edges on to the receiving states."""
    scale = _unit_scale(producer_edges["Strength"])
    for _, edge in producer_edges.iterrows():
        cell, hormone = edge["Celltype"], edge["Hormone"]
        if cell not in y_producer or hormone not in y_hormone:
            continue
        t = scale(edge["Strength"])
        ax.add_patch(
            FancyArrowPatch(
                (X_PRODUCER + 0.012, y_producer[cell]),
                (X_HORMONE - 0.085, y_hormone[hormone]),
                connectionstyle=f"arc3,rad={RELAY['curve_rad']}",
                arrowstyle="-",
                lw=_between(RELAY["edge_lw_range"], t),
                color=RELAY["edge_color"],
                alpha=_between(RELAY["edge_alpha_range"], t),
                zorder=1,
            )
        )
    scale = _unit_scale(receiver_edges["Strength"])
    for _, edge in receiver_edges.iterrows():
        cell, hormone = edge["Celltype"], edge["Hormone"]
        if cell not in y_receiver or hormone not in y_hormone:
            continue
        t = scale(edge["Strength"])
        ax.add_patch(
            FancyArrowPatch(
                (X_HORMONE + 0.085, y_hormone[hormone]),
                (X_RECEIVER - 0.012, y_receiver[cell]),
                connectionstyle=f"arc3,rad={-RELAY['curve_rad']}",
                arrowstyle="-",
                lw=_between(RELAY["edge_lw_range"], t),
                color=style.CELL_FAMILY.get(receiver_family.get(cell, "Other"), RELAY["edge_color"]),
                alpha=min(0.95, _between(RELAY["edge_alpha_range"], t) + 0.12),
                zorder=2,
            )
        )


def _draw_relay_nodes(ax, producer_cells, y_producer, receiver_cells, y_receiver, receiver_family):
    """Producing nodes (left, labels to the left) and receiving nodes (right, labels to the right)."""
    if producer_cells:
        ax.scatter(
            [X_PRODUCER] * len(producer_cells),
            [y_producer[c] for c in producer_cells],
            s=RELAY["node_size"],
            c=RELAY["producer_face"],
            edgecolors=RELAY["producer_edge"],
            linewidths=0.7,
            zorder=3,
        )
        for c in producer_cells:
            ax.text(
                X_PRODUCER - 0.03,
                y_producer[c],
                c,
                ha="right",
                va="center",
                fontsize=RELAY["label_fontsize"],
                color="#222222",
                zorder=4,
            )
    ax.scatter(
        [X_RECEIVER] * len(receiver_cells),
        [y_receiver[c] for c in receiver_cells],
        s=RELAY["node_size"],
        c=[style.CELL_FAMILY.get(receiver_family[c], RELAY["receiver_face"]) for c in receiver_cells],
        edgecolors="#333333",
        linewidths=0.6,
        zorder=3,
    )
    for c in receiver_cells:
        ax.text(
            X_RECEIVER + 0.03,
            y_receiver[c],
            c,
            ha="left",
            va="center",
            fontsize=RELAY["label_fontsize"],
            color="#222222",
            zorder=4,
        )


def hormone_relay(producer_edges, receiver_edges, families, hormones=None):
    """Fig 4 l: three-column network of producing cell states, hormones and receiving cell states.

    Edges are curved, with width and opacity following the hormone2cell strength (scaled per side). Cell states
    are grouped by cell family; edges to and nodes of the receiving states carry the colour of their family.
    Hormones are ordered by the mean position of their receiving states, and a hormone without a drawn producer
    gets a grey box.

    Parameters
    ----------
    producer_edges, receiver_edges
        Edges of :func:`hhca.tl.relay_edges`.
    families
        Mapping of cell family -> patterns on the Level-4 type names, in the order of the node blocks.
    hormones
        Hormone ids in the order that breaks ties between hormones of equal strength (default: order of the
        receiver edges).
    """
    display_name = receiver_edges.drop_duplicates("Hormone").set_index("Hormone")["hormone_display"].to_dict()
    received = set(receiver_edges["Hormone"])
    shown = [h for h in (hormones or dict.fromkeys(receiver_edges["Hormone"])) if h in received]
    receiver_cells, y_receiver = _family_grouped(
        list(receiver_edges["Celltype"].unique()),
        receiver_edges.groupby("Celltype")["Strength"].max().to_dict(),
        families,
    )
    producer_cells, y_producer = list(producer_edges["Celltype"].unique()), {}
    if producer_cells:
        producer_cells, y_producer = _family_grouped(
            producer_cells, producer_edges.groupby("Celltype")["Strength"].max().to_dict(), families
        )
    receiver_best = receiver_edges.groupby("Hormone")["Strength"].max()
    hormone_order = _order_hormones(shown, receiver_best, receiver_edges, receiver_cells)
    n = len(hormone_order)
    y_hormone = dict(zip(hormone_order, np.array([0.5]) if n <= 1 else np.linspace(0.98, 0.02, n)))
    receiver_family = {c: cell_family(c, families) for c in receiver_cells}
    produced = set(producer_edges["Hormone"])

    with panel(RELAY["figsize"], RELAY_RC) as fig:
        ax = fig.subplots()
        _draw_relay_edges(ax, producer_edges, receiver_edges, y_producer, y_hormone, y_receiver, receiver_family)
        _draw_relay_nodes(ax, producer_cells, y_producer, receiver_cells, y_receiver, receiver_family)
        for hormone in hormone_order:
            ax.text(
                X_HORMONE,
                y_hormone[hormone],
                display_name.get(hormone, hormone),
                ha="center",
                va="center",
                fontsize=RELAY["hormone_fontsize"],
                color=RELAY["hormone_text"],
                zorder=5,
                bbox={
                    "boxstyle": "round,pad=0.32",
                    "fc": RELAY["hormone_face"],
                    "ec": RELAY["hormone_edge"] if hormone in produced else "#bbbbbb",
                    "lw": 1.0,
                },
            )
        for x, text, color in [
            (X_PRODUCER, "Producing cell states", RELAY["producer_edge"]),
            (X_HORMONE, "Hormone", RELAY["hormone_text"]),
            (X_RECEIVER, "Receiving cell states", RELAY["receiver_edge"]),
        ]:
            ax.text(
                x,
                1.0 + 0.055,
                text,
                ha="center",
                va="bottom",
                fontsize=RELAY["header_fontsize"],
                fontweight="bold",
                color=color,
            )
        present = [f for f in [*families, "Other"] if f in {receiver_family[c] for c in receiver_cells}]
        handles = [Line2D([0], [0], color=style.CELL_FAMILY[f], lw=3.0, label=f) for f in present]
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.075),
            ncol=min(len(handles), 6),
            fontsize=7.5,
            handlelength=2.0,
            columnspacing=1.4,
            frameon=False,
            title="edge / receiving-cell colour = receiving L4 family   (line width ∝ hormone2cell strength)",
            title_fontsize=7.5,
        )
        ax.set_xlim(-1.02, 3.02)
        ax.set_ylim(-0.05, 1.13)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        ax.get_legend().set_bbox_to_anchor((0.5, -0.185), transform=ax.transAxes)
        ax.set_xlim(-0.55, 2.55)  # crop the empty margins left and right
    return fig
