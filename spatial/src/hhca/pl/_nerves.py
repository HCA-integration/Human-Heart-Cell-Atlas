"""Panels of the nerve and macrophage analysis (Fig 5 b-h, j-l)."""

from __future__ import annotations

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle

from hhca.pl import style
from hhca.pl._panel import panel
from hhca.stats import stars
from hhca.tl._nerves import size_class_labels

FIELD_RC = {**style.PANEL_RC, "savefig.pad_inches": 0.02}
# short names of the Level-3 types in the key of the fields
FIELD_LABELS = {
    "Neural Cells": "Neural cells",
    "Macrophages": "Macrophages",
    "Capillary Endothelial cells": "Capillary EC",
    "Pericytes": "Pericytes",
    "Vascular Smooth Muscle cells": "VSMC",
    "Arterial Endothelial cells": "Arterial EC",
    "Venous Endothelial cells": "Venous EC",
}
SIGNIFICANT = "#B2182B"
NOT_SIGNIFICANT = "#D8A0A6"


def _scale_bar(ax, length_um, x0, x1, y0, y1, colour="white"):
    """Scale bar with its length in the lower left of a field (y axis inverted: y1 at the bottom)."""
    width, height = x1 - x0, y1 - y0
    bar_x, bar_y = x0 + 0.06 * width, y1 - 0.07 * height
    ax.add_patch(
        Rectangle((bar_x, bar_y), length_um, 0.012 * height, facecolor=colour, edgecolor="black", lw=0.4, zorder=6)
    )
    ax.text(
        bar_x + length_um / 2,
        bar_y - 0.015 * height,
        f"{length_um:g} µm",
        color=colour,
        ha="center",
        va="bottom",
        fontsize=6.5,
        zorder=6,
    )


def _squares(ax, x, y, colours, alpha=0.85, zorder=3, size=2.0):
    """Filled 2 um bins as squares centred on (x, y)."""
    bins = [Rectangle((bin_x - size / 2, bin_y - size / 2), size, size) for bin_x, bin_y in zip(x, y)]
    ax.add_collection(PatchCollection(bins, facecolor=colours, edgecolor="none", alpha=alpha, zorder=zorder))


def _hatched(ax, x, y, zorder=4, size=2.0):
    """Unassigned 2 um bins as black diagonal hatching, as in Fig 4 f."""
    bins = [Rectangle((bin_x - size / 2, bin_y - size / 2), size, size) for bin_x, bin_y in zip(x, y)]
    ax.add_collection(
        PatchCollection(bins, facecolor="none", edgecolor="#111111", linewidth=0, hatch="////", zorder=zorder)
    )


# ---------------------------------------------------------------------------------------------------------------
# Fig 5 b: a nerve next to a vessel


def nerve_on_he(field, neural_type="Neural Cells"):
    """Fig 5 b: a field of the H&E (top) and the same field with its neural 2 um bins filled (bottom).

    Parameters
    ----------
    field
        Field of :func:`hhca.tl.registered_field`.
    neural_type
        Level-3 label of the bins that are drawn.
    """
    x0, x1, y1, y0 = field["extent"]
    bin_x, bin_y = field["bx"], field["by"]
    shown = (field["blab"] == neural_type) & (bin_x >= x0 - 1) & (bin_x < x1 + 1) & (bin_y >= y0 - 1) & (bin_y < y1 + 1)
    with panel((2.3, 4.7), FIELD_RC) as fig:
        axes = fig.subplots(2, 1)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01, hspace=0.05)
        for ax in axes:
            ax.imshow(field["he"], extent=[x0, x1, y1, y0], interpolation="nearest")
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_aspect("equal")
            ax.axis("off")
        _squares(axes[1], bin_x[shown], bin_y[shown], style.SPATIAL[neural_type], alpha=0.75)
        _scale_bar(axes[0], 25, x0, x1, y0, y1)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 5 c: neural and vascular bins in a window

NEURAL_MARK = "#d62728"
VASCULAR_MARK = "#19b3ff"


def nerves_and_vessels(field, vascular_types, neural_type="Neural Cells"):
    """Fig 5 c: neural (red stars) and vascular (blue squares) 2 um bins of a window on the H&E.

    Parameters
    ----------
    field
        Window of :func:`hhca.tl.registered_field`.
    vascular_types
        Level-3 labels drawn as vascular.
    """
    x, y, labels = field["bx"], field["by"], field["blab"]
    x0, x1, y1, y0 = field["extent"]
    neural = labels == neural_type
    vascular = np.isin(labels, vascular_types)
    other = ~neural & ~vascular
    with panel((6.6, 6.6), style.PANEL_RC) as fig:
        ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
        ax.imshow(field["he"], extent=[x0, x1, y1, y0], interpolation="antialiased", alpha=0.85)
        ax.scatter(x[other], y[other], s=0.5, c="#888888", alpha=0.05, linewidths=0)
        ax.scatter(x[vascular], y[vascular], s=2.2, c=VASCULAR_MARK, marker="s", linewidths=0)
        ax.scatter(x[neural], y[neural], s=14, c=NEURAL_MARK, marker="*", edgecolors="k", linewidths=0.3)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 5 d: distance of nerves to vessels

NEAR_COLOURS = ["#4d6fb0", "#4aa3a2"]


def nerve_vessel_distance(d_nerve, d_background, near_um=(8, 16)):
    """Fig 5 d: distance from nerves to the nearest vascular bin, against their random placements.

    Overlaid density histograms with the two medians (dashed) and the distances `near_um` (dotted).

    Parameters
    ----------
    d_nerve
        Distance per nerve (um).
    d_background
        Distance per random placement of a nerve shape (um).
    """
    d_nerve, d_background = np.asarray(d_nerve), np.asarray(d_background)
    median_nerve, median_background = float(np.median(d_nerve)), float(np.median(d_background))
    nerve_colour = style.NERVE_PLACEMENT["nerves"]
    with panel((5.4, 4.6), style.PANEL_RC) as fig:
        ax = fig.subplots()
        bins = np.arange(0, 52, 1.5)
        ax.hist(
            d_background,
            bins=bins,
            density=True,
            color=style.NERVE_PLACEMENT["random position"],
            alpha=0.75,
            label=f"same nerves, random tissue position (n={len(d_background):,})",
        )
        ax.hist(d_nerve, bins=bins, density=True, color=nerve_colour, alpha=0.72, label=f"nerves (n={len(d_nerve):,})")
        ax.axvline(median_nerve, color=nerve_colour, ls="--", lw=1.6, label=f"nerve median {median_nerve:.1f} µm")
        ax.axvline(
            median_background, color="#333333", ls="--", lw=1.6, label=f"background median {median_background:.1f} µm"
        )
        for distance, colour in zip(near_um, NEAR_COLOURS):
            ax.axvline(distance, color=colour, ls=":", lw=1.4, label=f"{distance:g} µm")
        ax.set_xlabel("distance from nerve to nearest vascular bin (µm)", fontsize=10)
        ax.set_ylabel("density", fontsize=10)
        ax.set_xlim(0, 51)
        ax.legend(fontsize=7.6, frameon=False, loc="upper right")
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 5 e: myelination by nerve size


def myelination_by_size(nerves, size_edges, myelinated_min=3):
    """Fig 5 e: myelination score of the nerves by size class, with the share of myelinating nerves on top.

    Parameters
    ----------
    nerves
        Per-nerve table of :func:`hhca.tl.nerve_myelination`.
    size_edges
        Edges of the size classes in bins.
    myelinated_min
        Myelin counts from which a nerve is myelinating (dashed line).
    """
    labels = size_class_labels(size_edges)
    groups = [nerves[nerves.szbin == label].logmy.values for label in labels]
    with panel((5.2, 3.9), style.PANEL_RC) as fig:
        ax = fig.subplots()
        ax.boxplot(
            groups,
            positions=np.arange(len(labels)),
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": style.INK, "lw": 1.4},
            boxprops={"facecolor": style.NERVE_SIZE, "edgecolor": style.MUTED, "lw": 0.9},
            whiskerprops={"color": style.MUTED, "lw": 0.9},
            capprops={"color": style.MUTED, "lw": 0.9},
        )
        rng = np.random.default_rng(0)
        for i, group in enumerate(groups):
            jitter = rng.normal(i, 0.07, len(group))
            ax.scatter(jitter, group, s=4, color=style.INK, alpha=0.18, linewidths=0, zorder=3)
        threshold = np.log1p(myelinated_min)
        ax.axhline(threshold, color=SIGNIFICANT, lw=1.1, ls="--", zorder=2)
        ax.text(
            len(labels) - 0.5,
            threshold + 0.07,
            "myelin+ threshold",
            color=SIGNIFICANT,
            fontsize=8.0,
            ha="right",
            va="bottom",
        )
        for i, label in enumerate(labels):
            pct = 100 * nerves[nerves.szbin == label].myelinating.mean()
            ax.text(
                i,
                ax.get_ylim()[1] * 0.97,
                f"{pct:.0f}%",
                ha="center",
                va="top",
                fontsize=8.6,
                color=style.INK,
                weight="semibold",
            )
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("nerve size (number of neural bins)")
        ax.set_ylabel("myelination score  log1p(Σ myelin counts)")
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 5 f: a myelinating nerve

FADED_HE = 0.45


def myelin_on_he(field):
    """Fig 5 f: a field of the H&E (top) and its neural 2 um bins coloured by their myelin score (bottom).

    The colour scale runs from 0 to the 95th percentile of the positive scores of the field, at least to 2.

    Parameters
    ----------
    field
        Field of :func:`hhca.tl.registered_field` with the scores of :func:`hhca.tl.local_myelin`.
    """
    x0, x1, y1, y0 = field["extent"]
    inside = (field["nx"] >= x0) & (field["nx"] <= x1) & (field["ny"] >= y0) & (field["ny"] <= y1)
    x, y, score = field["nx"][inside], field["ny"][inside], field["nscore"][inside]
    vmax = max(2.0, float(np.percentile(score[score > 0], 95)) if (score > 0).any() else 2.0)
    with panel((3.0, 4.9), style.PANEL_RC) as fig:
        ax_he = fig.add_axes([0.02, 0.51, 0.74, 0.48])
        ax_score = fig.add_axes([0.02, 0.01, 0.74, 0.48])
        for ax, alpha in ((ax_he, 1.0), (ax_score, FADED_HE)):
            ax.imshow(field["he"], extent=[x0, x1, y1, y0], interpolation="nearest", alpha=alpha)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(0.6)
            _scale_bar(ax, 20, x0, x1, y0, y1)
        order = np.argsort(score)
        points = ax_score.scatter(
            x[order],
            y[order],
            c=score[order],
            s=22,
            marker="s",
            cmap=style.MYELIN_SCORE,
            vmin=0,
            vmax=vmax,
            linewidths=0.0,
        )
        colourbar = fig.colorbar(points, cax=fig.add_axes([0.82, 0.30, 0.05, 0.40]))
        colourbar.ax.tick_params(labelsize=7)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 5 g: neurotoxic drug score by myelination


def neurotoxic_score(per_nerve, p, score="cipn_score"):
    """Fig 5 g: drug2cell score over the neurotoxic drugs in myelinated and non-myelinated nerves.

    Parameters
    ----------
    per_nerve
        Per-nerve table with ``group`` (``myelinated`` / ``non_myelinated``) and the score.
    p
        p value of the comparison, shown as stars.
    """
    myelinated = per_nerve.loc[per_nerve.group == "myelinated", score].values
    non_myelinated = per_nerve.loc[per_nerve.group == "non_myelinated", score].values
    with panel((4.2, 4.6), style.PANEL_RC) as fig:
        ax = fig.subplots()
        positions = [1, 2]
        violins = ax.violinplot([myelinated, non_myelinated], positions=positions, widths=0.82, showextrema=False)
        for body in violins["bodies"]:
            body.set_facecolor("#aec7e8")
            body.set_edgecolor("#5b7fb0")
            body.set_alpha(0.62)
            body.set_linewidth(0.8)
        rng = np.random.default_rng(0)
        for position, scores, group in [(1, myelinated, "myelinated"), (2, non_myelinated, "non_myelinated")]:
            jitter = position + rng.uniform(-0.13, 0.13, len(scores))
            ax.scatter(jitter, scores, s=14, color=style.MYELINATION[group], alpha=0.6, edgecolors="none", zorder=3)
            ax.plot([position - 0.34, position + 0.34], [np.median(scores)] * 2, color="#37527a", lw=2.0, zorder=4)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"myelinated\n(n={len(myelinated)})", f"non-myel.\n(n={len(non_myelinated)})"], fontsize=9.5
        )
        ax.set_ylabel("CIPN drug2cell score\n(mean over neurotoxic drugs)", fontsize=10)
        ax.set_ylim(bottom=-0.1)
        top = max(myelinated.max(), non_myelinated.max())
        ax.plot([1, 1, 2, 2], [top * 1.04, top * 1.07, top * 1.07, top * 1.04], color=style.INK, lw=0.8)
        ax.text(1.5, top * 1.08, stars(p), ha="center", va="bottom", fontsize=10)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 5 h: neurotoxic drugs one by one


def neurotoxic_drugs(drugs):
    """Fig 5 h: difference of the drug2cell score between myelinated and non-myelinated nerves, per drug.

    Drugs with a detected target; bars are dark and starred by the adjusted p value.

    Parameters
    ----------
    drugs
        Per-drug table of :func:`hhca.tl.drug_differences`.
    """
    drugs = drugs[drugs.n_detected > 0].copy().sort_values("diff").reset_index(drop=True)
    with panel((6.4, 0.42 * len(drugs) + 1.4), style.PANEL_RC) as fig:
        ax = fig.subplots()
        y = np.arange(len(drugs))
        colours = [
            SIGNIFICANT if (np.isfinite(drug.fdr) and drug.fdr < 0.05) else NOT_SIGNIFICANT
            for drug in drugs.itertuples()
        ]
        ax.barh(y, drugs["diff"], color=colours, edgecolor="white", lw=0.6, height=0.68)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{drug.drug} ({drug.cls})" for drug in drugs.itertuples()], fontsize=9.5)
        for i, drug in enumerate(drugs.itertuples()):
            ax.text(
                drug.diff + (0.002 if drug.diff >= 0 else -0.002),
                i,
                stars(drug.fdr) if np.isfinite(drug.fdr) else "ns",
                va="center",
                ha="left" if drug.diff >= 0 else "right",
                fontsize=9,
                color=style.INK,
            )
        ax.axvline(0, color=style.MUTED, lw=0.8)
        ax.set_xlabel("Δ target score (myelinated − non-myelinated)", fontsize=10.5)
        ax.tick_params(labelsize=9.5)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 5 j: macrophages and vessels on the H&E

HATCHED_RC = {"hatch.linewidth": 0.6, "hatch.color": "#111111", **FIELD_RC}


def macrophage_fields(fields, show_types):
    """Fig 5 j: fields of the H&E with the 2 um bins of some cell types filled and the unassigned bins hatched.

    Parameters
    ----------
    fields
        Fields of :func:`hhca.tl.registered_field`, drawn from top to bottom.
    show_types
        Level-3 types whose bins are filled, in the order of the key.
    """
    with panel((2.3, 2.25 * len(fields) + 0.5), HATCHED_RC) as fig:
        axes = fig.subplots(len(fields), 1)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.12, hspace=0.06)
        for ax, field in zip(np.atleast_1d(axes), fields):
            x0, x1, y1, y0 = field["extent"]
            ax.imshow(field["he"], extent=[x0, x1, y1, y0], interpolation="nearest")
            shown = np.isin(field["blab"], show_types)
            colours = [style.SPATIAL[level3_type] for level3_type in field["blab"][shown]]
            _squares(ax, field["bx"][shown], field["by"][shown], colours, alpha=0.75)
            _hatched(ax, field["ux"], field["uy"])
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_aspect("equal")
            ax.axis("off")
            _scale_bar(ax, 50, x0, x1, y0, y1)
        handles = [
            Patch(facecolor=style.SPATIAL[level3_type], label=FIELD_LABELS[level3_type]) for level3_type in show_types
        ]
        handles.append(Patch(facecolor="white", edgecolor="#111111", hatch="////", linewidth=0.5, label="unassigned"))
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.0))
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 5 k: perineural macrophages


def perineural_counts(n_perineural, n_other):
    """Fig 5 k: number and share of the perineural and of the other macrophage pseudo-cells."""
    n = [n_perineural, n_other]
    with panel((2.9, 3.6), style.PANEL_RC) as fig:
        ax = fig.subplots()
        ax.bar([0, 1], n, color=list(style.NICHE.values()), edgecolor="black", lw=0.6, width=0.62)
        for i in range(2):
            ax.text(
                i,
                n[i] + 0.012 * max(n),
                f"n={n[i]:,}\n{100 * n[i] / sum(n):.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["perineural", "perivascular\nand interstitial"])
        ax.set_ylabel("macrophage pseudo-cells")
        ax.set_ylim(0, max(n) * 1.15)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Fig 5 l: markers of perineural macrophages


def perineural_markers(effects, markers):
    """Fig 5 l: difference of the macrophage markers between perineural and the other pseudo-cells.

    Bars give the section-adjusted coefficient of the perineural indicator, stars its adjusted p value.

    Parameters
    ----------
    effects
        Per-marker table of :func:`hhca.tl.marker_effects` (``marker``, ``beta``, ``padj``).
    markers
        Markers from top to bottom.
    """
    effects = effects.set_index("marker").reindex(markers)
    beta, padj = effects["beta"].to_numpy(), effects["padj"].to_numpy()
    with panel((3.6, 3.6), style.PANEL_RC) as fig:
        ax = fig.subplots()
        y = np.arange(len(markers))[::-1]
        ax.barh(y, beta, color=style.NICHE["perineural"], edgecolor="black", lw=0.6, height=0.62)
        limit = max(np.nanmax(np.abs(beta)) * 1.35, 0.3)
        for position, value, q in zip(y, beta, padj):
            ax.text(
                value + (0.04 * limit if value >= 0 else -0.04 * limit),
                position,
                stars(q) if np.isfinite(q) else "ns",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8,
            )
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(markers, fontstyle="italic")
        ax.set_xlim(-limit, limit)
        ax.set_xlabel("perineural vs perivascular+interstitial\n(Δ log1p CP10k, slide-adjusted)")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# SPP1 around nerves of increasing minimum size (results/nerve_macrophage/macrophage/figures)

MIN_SIZE_COLOURS = ["#999999", "#56B4E9", "#0072B2", "#D55E00"]
MIN_SIZE_RC = {
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def nerve_size_effect(profiles, effects, centres, gene="SPP1"):
    """Expression of a marker against the distance to nerves of increasing minimum size, and its slope.

    Parameters
    ----------
    profiles
        Minimum nerve size -> mean expression per distance bin (:func:`hhca.tl.expression_by_distance`).
    effects
        Slope per marker and minimum nerve size (``marker``, ``k``, ``beta_uni``, ``ci_lo``, ``ci_hi``,
        ``p_uni``).
    centres
        Position of each distance bin on the axis (um).
    """
    min_sizes = list(profiles)
    with panel((12.6, 5.0), MIN_SIZE_RC) as fig:
        axes = fig.subplots(1, 2)
        ax = axes[0]
        for k, colour in zip(min_sizes, MIN_SIZE_COLOURS):
            ax.errorbar(
                np.array(centres),
                profiles[k]["mean"].values,
                yerr=profiles[k]["sem"].values,
                marker="o",
                ms=5,
                lw=1.8,
                capsize=2.5,
                color=colour,
                label=f"nerve = >={k} combined bins" if k > 1 else "nerve = any single bin",
            )
        ax.set_xlabel("distance to nearest nerve (um)")
        ax.set_ylabel(f"mean {gene} log1p(CP10k)")
        ax.set_title(
            f"(a) Perineural {gene} drop under stricter nerve definitions\n"
            "requiring real nerves (>=k combined neural bins), eps=6 um"
        )
        ax.legend(frameon=False, fontsize=7.6)

        ax = axes[1]
        slopes = effects[effects.marker == gene]
        x = np.arange(len(min_sizes))
        ax.errorbar(
            x,
            slopes.beta_uni.values,
            yerr=[slopes.beta_uni.values - slopes.ci_lo.values, slopes.ci_hi.values - slopes.beta_uni.values],
            marker="o",
            ms=8,
            lw=0,
            elinewidth=1.6,
            capsize=4,
            color=MIN_SIZE_COLOURS[-1],
        )
        ax.axhline(0, color="#999", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([f"k>={k}" for k in min_sizes])
        ax.set_xlabel("minimum nerve cluster size (combined neural bins)")
        ax.set_ylabel(f"standardized {gene} slope per 1 SD nerve distance\n(negative = {gene} higher near nerves)")
        ax.set_title(
            f"(b) {gene}-near-nerve slope vs nerve-definition strictness\n"
            "slide FE + library; attenuates ~3x but stays significant to k>=5, null at k>=10"
        )
        for slope in slopes.itertuples():
            ax.annotate(
                f"p={slope.p_uni:.0e}",
                (min_sizes.index(slope.k), slope.beta_uni),
                textcoords="offset points",
                xytext=(8, 0),
                fontsize=7,
                va="center",
            )
        fig.suptitle(
            f"Robustness of the perineural {gene} finding to requiring real nerves "
            "(clusters of combined neural bins, not stray single bins)",
            fontsize=11,
            y=1.0,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
