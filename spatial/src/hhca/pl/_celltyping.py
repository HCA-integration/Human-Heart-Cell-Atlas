"""Plots of the cell-typing notebook: H&E fields with the 2 um bins on top, in the look of Fig 1 j and Fig 4 f."""

from __future__ import annotations

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle

from hhca.pl import style
from hhca.pl._panel import panel
from hhca.tl._bins import CARDIOMYOCYTES

FIELD_RC = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "pdf.fonttype": 42}
SCALE_BAR_UM = 20.0


def _squares(ax, x, y, colours, alpha, zorder):
    """Filled 2 um bins as squares centred on (x, y)."""
    bins = [Rectangle((xx - 1, yy - 1), 2, 2) for xx, yy in zip(x, y)]
    ax.add_collection(PatchCollection(bins, facecolor=colours, edgecolor="none", alpha=alpha, zorder=zorder))


def _draw_field(ax, field, flagged, hide, palette):
    """One field: H&E, the bins of the shown Level-3 types, the flagged bins and a scale bar."""
    x0, x1, y0, y1 = field["ext"]
    ax.imshow(field["he"], extent=[x0, x1, y1, y0], interpolation="lanczos", zorder=0)
    labels = field["cats"][field["code"]]
    shown = ~np.isin(labels, hide)
    _squares(ax, field["x"][shown], field["y"][shown], [palette[label] for label in labels[shown]], 0.85, 2)
    if flagged is not None and len(flagged):
        colours = [style.SPILLOVER[flag] for flag in flagged["flag"]]
        bins = [Rectangle((x - 1, y - 1), 2, 2) for x, y in zip(flagged["x_um"], flagged["y_um"])]
        ax.add_collection(PatchCollection(bins, facecolor=colours, edgecolor=style.INK, linewidth=0.3, zorder=3))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar_x, bar_y = x0 + 0.06 * (x1 - x0), y1 - 0.06 * (y1 - y0)
    ax.plot([bar_x, bar_x + SCALE_BAR_UM], [bar_y, bar_y], color="black", lw=3, solid_capstyle="butt", zorder=5)
    ax.text(
        bar_x + SCALE_BAR_UM / 2,
        bar_y - 0.025 * (y1 - y0),
        f"{SCALE_BAR_UM:g} µm",
        ha="center",
        va="bottom",
        fontsize=8,
        zorder=5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1},
    )
    return set(labels[shown])


def he_fields(fields, titles, flagged=None, hide=CARDIOMYOCYTES, palette=None):
    """Fields of the registered H&E with the 2 um bins on top.

    The bins are drawn as filled squares in the colour of their Level-3 type, as in the spatial panels; the bins
    of the types in `hide` (cardiomyocytes, which fill the tissue) are left out so that the H&E stays visible.
    Bins that carry a spillover flag can be drawn on top in the colour of their flag, with a dark outline.

    Parameters
    ----------
    fields
        Fields of :func:`hhca.tl.he_window`.
    titles
        One title per field.
    flagged
        Per field, a table of flagged bins (``x_um``, ``y_um``, ``flag``) or None.
    hide
        Level-3 types that are not drawn.
    palette
        Level-3 type -> colour (default: the palette of the spatial panels).
    """
    palette = palette or style.SPATIAL
    flagged = flagged or [None] * len(fields)
    with panel((3.3 * len(fields), 3.95), FIELD_RC, save_dpi=200) as fig:
        axes = fig.subplots(1, len(fields), squeeze=False)[0]
        present, flags_present = set(), set()
        for ax, field, title, flags in zip(axes, fields, titles, flagged):
            present |= _draw_field(ax, field, flags, hide, palette)
            if flags is not None:
                flags_present |= set(flags["flag"])
            ax.set_title(title, fontsize=10, pad=4)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.2, wspace=0.04)
        handles = [
            Patch(facecolor=palette[label], edgecolor="none", alpha=0.85, label=label)
            for label in palette
            if label in present
        ]
        handles += [
            Patch(
                facecolor=style.SPILLOVER[flag],
                edgecolor=style.INK,
                linewidth=0.5,
                label=f"flagged: {style.SPILLOVER_LABELS[flag]}",
            )
            for flag in style.SPILLOVER
            if flag in flags_present
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=min(4, max(2, len(fields) * 2)),
            frameon=False,
            fontsize=7.5,
            handlelength=1.2,
            columnspacing=1.2,
        )
    return fig
