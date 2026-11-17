"""Colours and matplotlib defaults shared by the figure panels and the notebooks.

from hhca.pl import style

style.apply()  # rcParams for panels drawn at their final size
style.ct("Fibroblasts")  # colour of a Level-3 type in the atlas palette (short aliases accepted)
style.palette("spatial")  # Level-3 type -> colour on tissue, as in the spatial panels
style.notebook()  # look of the exploratory plots in the notebooks
style.set_colors(adata, "L3_label", style.SPATIAL)  # colours of a categorical obs column for scanpy plots
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import to_hex, to_rgb

# Level-3 colours on tissue: the palette of the spatial panels (Fig 1 j, Fig 3 e, Fig 4 f, Fig 5 b, j, ED6 f)
SPATIAL = {
    "Adipocytes": "#1f77b4",
    "Arterial Endothelial cells": "#aec7e8",
    "Atrial Cardiomyocytes": "#b5b5b5",
    "B cells": "#ffbb78",
    "Capillary Endothelial cells": "#2ca02c",
    "Dendritic cells": "#98df8a",
    "Endocardial Endothelial Cells": "#d62728",
    "Epicardium": "#ff9896",
    "Fibroblasts": "#9467bd",
    "ILC": "#c5b0d5",
    "Lymphatic Endothelial Cells": "#8c564b",
    "Lymphoid Cells (Proliferating)": "#c49c94",
    "Macrophages": "#e377c2",
    "Mast cells": "#f7b6d2",
    "Monocytes": "#7f7f7f",
    "NK cells": "#c7c7c7",
    "Neural Cells": "#bcbd22",
    "Pericytes": "#dbdb8d",
    "Plasma Cells": "#17becf",
    "T cells": "#9edae5",
    "Vascular Smooth Muscle cells": "#393b79",
    "Venous Endothelial cells": "#5254a3",
    "Ventricular Cardiomyocytes": "#b5b5b5",
}

# Level-3 cell-type palette of the atlas (non-spatial panels)
CT = {
    "Adipocytes": "#d7b699",
    "Atrial Cardiomyocytes": "#eea2a4",
    "Ventricular Cardiomyocytes": "#eabcc5",
    "Endocardial Endothelial Cells": "#7ac4bd",
    "Lymphatic Endothelial Cells": "#26386a",
    "Arterial Endothelial cells": "#4292a6",
    "Capillary Endothelial cells": "#96dcdd",
    "Venous Endothelial cells": "#6b9acb",
    "Epicardium": "#cdc6a8",
    "B cells": "#de91b3",
    "Plasma Cells": "#b471ee",
    "Dendritic cells": "#c2eec1",
    "Macrophages": "#90ba80",
    "Mast cells": "#62a65e",
    "Monocytes": "#4da79d",
    "ILC": "#a565bc",
    "Lymphoid Cells (Proliferating)": "#e9bd96",
    "NK cells": "#d19faf",
    "T cells": "#d587a4",
    "Neural Cells": "#f3e5b8",
    "Fibroblasts": "#e7af83",
    "Pericytes": "#dbcff0",
    "Vascular Smooth Muscle cells": "#8161bc",
}
ALIASES = {
    "Ventricular CM": "Ventricular Cardiomyocytes",
    "Atrial CM": "Atrial Cardiomyocytes",
    "VSMC": "Vascular Smooth Muscle cells",
    "Arterial EC": "Arterial Endothelial cells",
    "Capillary EC": "Capillary Endothelial cells",
    "Venous EC": "Venous Endothelial cells",
    "Endocardial EC": "Endocardial Endothelial Cells",
    "Lymphatic EC": "Lymphatic Endothelial Cells",
    "Macrophage": "Macrophages",
    "T cell": "T cells",
    "NK cell": "NK cells",
    "Lymphoid Proliferating": "Lymphoid Cells (Proliferating)",
}

DIVERGING = "RdBu_r"  # z-scores and log fold-changes (red high, blue low)
SEQUENTIAL = "magma"  # scores on tissue (Fig 3 d)
INK = "#2B2B2B"
MUTED = "#7A7A7A"

# the four CMFLAG spillover flags on tissue: a flag takes the colour of the cell type whose transcripts spill;
# cardiomyocytes are grey in the spatial palette, so their flag takes the ink and stands out against the tissue
SPILLOVER = {
    "flag_cm_spillover": INK,
    "flag_vsmc_spillover": SPATIAL["Vascular Smooth Muscle cells"],
    "flag_spill_artEC": SPATIAL["Arterial Endothelial cells"],
    "flag_spill_venEC": SPATIAL["Venous Endothelial cells"],
}
SPILLOVER_LABELS = {
    "flag_cm_spillover": "cardiomyocyte",
    "flag_vsmc_spillover": "VSMC",
    "flag_spill_artEC": "arterial EC",
    "flag_spill_venEC": "venous EC",
}
# bins kept and removed by the clean-up: no panel colours them, so they take the ink scale (removed = darker)
KEPT_REMOVED = {"kept": MUTED, "removed": INK}

# gene-high bins against the other bins (Fig 3 f-h)
HIGH_LOW = {"high": "#C2362F", "low": "0.72"}
# NPPA-high cardiomyocyte bins against all other bins of a section, in the colours of Fig 3 f-h
NPPA_HIGH = {"NPPA-high CM": HIGH_LOW["high"], "other": HIGH_LOW["low"]}
# vascular subtypes in the composition curves (Fig 4 g; Okabe-Ito)
VASCULAR_SUBTYPE = {
    "VSMC (mural)": "#D55E00",
    "Pericyte (mural)": "#E69F00",
    "Arterial EC": "#0072B2",
    "Capillary EC": "#009E73",
    "Venous EC": "#56B4E9",
}
# vessel-wall compartments of the 2 um bins (hhca.tl.wall_compartments) on tissue: the Level-3 colours of the
# spatial panels; capillary endothelium with a pericyte nearby takes the capillary colour, without one the same
# colour mixed half with white (lighten)
WALL_COMPARTMENT = {
    "Vascular Smooth Muscle cells": SPATIAL["Vascular Smooth Muscle cells"],
    "Arterial Endothelial cells": SPATIAL["Arterial Endothelial cells"],
    "Pericytes": SPATIAL["Pericytes"],
    "Capillary EC (pericyte+)": SPATIAL["Capillary Endothelial cells"],
    "Capillary EC (pericyte-)": "#95d095",
    "Venous Endothelial cells": SPATIAL["Venous Endothelial cells"],
    "Lymphatic Endothelial Cells": SPATIAL["Lymphatic Endothelial Cells"],
    "Endocardial Endothelial Cells": SPATIAL["Endocardial Endothelial Cells"],
}
# calibre classes: no panel colours them, so they take the ink scale of the panels (larger = darker)
CALIBRE_CLASS = {"small_resistance": "#C9C9C9", "intermediate": MUTED, "large_conductance": INK}
# compartments of the composition bars (ED6 h), bottom to top
COMPARTMENT = {
    "CM": "#E8A0B0",
    "Fibroblast": "#E8A45C",
    "Endothelial": "#7ECECA",
    "Pericyte": "#C9B8E8",
    "SMC": "#8A63C4",
    "Myeloid": "#7CB668",
    "Lymphoid": "#CF6F9B",
    "Mast": "#3F7D54",
    "Adipocyte": "#CB8F5E",
    "Neural": "#E8CF5A",
    "Epicardium": "#B3A86A",
}
# deconvolution methods: the panels name the chosen method in red and the others in black (ED6 g)
METHOD = {
    "spotiphy": "#B2182B",
    "RCTD": INK,
    "cell2location": INK,
    "ucaspatial": INK,
    "stereoscope": INK,
    "SPOTlight": INK,
    "NNLS": INK,
}
# atrial and ventricular genes of the chamber panel (Fig 3 c; Okabe-Ito orange and blue)
CHAMBER_GENES = {"atria": "#D55E00", "ventricle": "#0072B2"}
# spots of atria and ventricles, and pericyte-low spots (ED8 b-c)
CHAMBER_REGION = {"atria": "#C0392B", "ventricle": "#2166AC"}
PERICYTE_LOW = "#BDBDBD"
# cardiomyocyte-dominant Visium spots against the others: the spots that Fig 3 c keeps are grey, the rest red
CM_DOMINANT = {"cardiomyocyte-dominant": HIGH_LOW["low"], "other": HIGH_LOW["high"]}

# neural bins in a nerve and single neural bins on tissue: the neural colour of the spatial panels and grey
NERVE_BINS = {"in a nerve": SPATIAL["Neural Cells"], "single bins": MUTED}
# nerves against the same nerve shapes at random tissue positions (Fig 5 d)
NERVE_PLACEMENT = {"nerves": "#c2668f", "random position": "#c9c9c9"}
# nerve size classes (boxes of Fig 5 e)
NERVE_SIZE = "#d9c561"
# myelinated and non-myelinated nerves (dots of Fig 5 g)
MYELINATION = {"myelinated": "#B2182B", "non_myelinated": "#8aa0c0"}
MYELIN_SCORE = "viridis"  # myelin counts on tissue (Fig 5 f)
# perineural macrophage pseudo-cells against the perivascular and interstitial ones (Fig 5 k, l)
NICHE = {"perineural": "#d97b2b", "rest": "#8c8c8c"}
# Xenium panels (Fig 6, ED9, ED6 i-j): display names of cell types on top of ALIASES
XENIUM_ALIASES = {
    "Cardiomyocytes": "Ventricular Cardiomyocytes",
    "Smooth muscle": "Vascular Smooth Muscle cells",
    "Neural": "Neural Cells",
    "Plasma cells": "Plasma Cells",
    "Lymphoid (prolif.)": "Lymphoid Cells (Proliferating)",
}
# Level-3 types of the Xenium panels under their atlas and their display names (Fig 6 c, ED6 i)
XENIUM = {**CT, **{alias: CT[name] for alias, name in {**ALIASES, **XENIUM_ALIASES}.items()}}
# the two Xenium datasets in the exploratory plots: healthy section and rejection cohort
XENIUM_DATASET = {"healthy section": INK, "rejection cohort": MUTED}
# lymphoid Level-4 classes on the slide (Fig 6 c)
LYMPHOID = {
    "CD8 T cells": "#C4453A",
    "Proliferating CD8 T cells": "#FF5A4D",
    "CD4 T cells": "#2E6FA8",
    "Proliferating CD4 T cells": "#6FC0FF",
    "Tregs": "#7F9A00",
    "Proliferating Tregs": "#C8F03C",
    "NK cells": "#0E8F9E",
    "Proliferating NK cells": "#3FE0F0",
}
# classes of the cells of a core as in Fig 6 c (hhca.tl.spatial_classes): immune Level-3 types, the lymphoid
# Level-4 classes on top, and the structural cells in grey
IMMUNE_CLASS = {**XENIUM, **LYMPHOID, "structural / other": MUTED}
# immune composition of the biopsy cores (Fig 6 d, ED9 a)
COMPOSITION = {
    "CD4 T": "#9ecae1",
    "prolif. CD4 T": "#fb6a4a",
    "CD8 T": "#3878b0",
    "prolif. CD8 T": "#cb181d",
    "Treg": "#08306b",
    "prolif. Treg": "#67000d",
    "Macrophages": "#d98cb3",
    "Monocytes": "#636363",
    "NK": "#bdbdbd",
    "prolif. NK": "#7a3fa8",
    "Dendritic cells": "#74c476",
    "B cells": "#d9a441",
    "Plasma Cells": "#2fb8c4",
    "Other": "#e6e6e6",
}
# immune lineages on the cores (ED6 j)
LINEAGE = {
    "T / NK": "#fa3580",
    "B / Plasma": "#d279ff",
    "Macrophage": "#00d158",
    "Monocyte / DC": "#00bbd2",
    "Mast": "#ff8b00",
    "Lymphoid (prolif.)": "#ffc100",
    "structural / other": "#959595",
}
# rejection type (Fig 6 e, i, ED9 b-c), outcome of the rejection episode (Fig 6 f-g), ISHLT cellular grade (Fig 6 l)
REJECTION = {
    "no_rejection": "#B9B9B9",
    "cellular_rejection": "#3B72B0",
    "antibody_rejection": "#B2182B",
    "mixed_rejection": "#8161BC",
}
RESOLUTION = {"resolved": "#2166AC", "persistence": "#B2182B"}
GRADE = {0: "#F4C6D7", 1: "#D48CAE", 2: "#A8557F", 3: "#6B2E4E"}
# immune foci: no panel colours them, so they take the ink scale of the panels (in a focus = darkest)
FOCUS = {"immune cell in a focus": INK, "immune cell outside foci": MUTED, "other cell": "#DDDDDD"}

# donors of the atlas: sex (Fig 2 h-l), direction of an age effect (Fig 5 m, n; the light shades mark effects that
# are significant after FDR only) and cell families of the Level-4 states (Fig 4 l; tab10)
SEX = {"female": "#B2182B", "male": "#2166AC"}
AGE_DIRECTION = {"up_with_age": "#762A83", "down_with_age": "#1B7837"}
AGE_DIRECTION_LIGHT = {"up_with_age": "#C2A5CF", "down_with_age": "#A6DBA0"}
# diverging map in the direction colours: green (down), white, purple (up)
AGE_DIVERGING = mpl.colors.LinearSegmentedColormap.from_list(
    "age_div", ["#1B7837", "#5AAE61", "#A6DBA0", "#F5F5F5", "#C2A5CF", "#9970AB", "#762A83"]
)
CELL_FAMILY = {
    "Endothelial": "#1f77b4",
    "Pericyte": "#9467bd",
    "VSMC": "#8c564b",
    "Cardiomyocyte": "#7f7f7f",
    "Fibroblast": "#2ca02c",
    "Immune": "#ff7f0e",
    "Neural": "#e377c2",
    "Adipocyte": "#bcbd22",
    "Epicardium": "#17becf",
    "Other": "#aaaaaa",
}

PALETTES = {
    "spatial": SPATIAL,
    "atlas": CT,
    "kept_removed": KEPT_REMOVED,
    "high_low": HIGH_LOW,
    "nppa_high": NPPA_HIGH,
    "vascular_subtype": VASCULAR_SUBTYPE,
    "wall_compartment": WALL_COMPARTMENT,
    "calibre_class": CALIBRE_CLASS,
    "nerve_bins": NERVE_BINS,
    "nerve_placement": NERVE_PLACEMENT,
    "myelination": MYELINATION,
    "niche": NICHE,
    "xenium": XENIUM,
    "xenium_dataset": XENIUM_DATASET,
    "lymphoid": LYMPHOID,
    "immune_class": IMMUNE_CLASS,
    "composition": COMPOSITION,
    "lineage": LINEAGE,
    "rejection": REJECTION,
    "resolution": RESOLUTION,
    "grade": GRADE,
    "focus": FOCUS,
    "compartment": COMPARTMENT,
    "method": METHOD,
    "chamber_genes": CHAMBER_GENES,
    "chamber_region": CHAMBER_REGION,
    "cm_dominant": CM_DOMINANT,
    "spillover": SPILLOVER,
    "sex": SEX,
    "age_direction": AGE_DIRECTION,
    "age_direction_light": AGE_DIRECTION_LIGHT,
    "cell_family": CELL_FAMILY,
}


def palette(kind: str = "spatial") -> dict[str, str]:
    """Category -> colour of one of the palettes of the panels.

    ``"spatial"``: Level-3 types on tissue; ``"atlas"``: Level-3 types elsewhere; ``"kept_removed"``: bins kept
    and removed by the clean-up; ``"high_low"``: gene-high against the other bins; ``"nppa_high"``: NPPA-high
    cardiomyocyte bins against the others; ``"vascular_subtype"``: the composition curves;
    ``"wall_compartment"``: vessel-wall compartments on tissue; ``"calibre_class"``: calibre classes;
    ``"nerve_bins"``: neural bins in a nerve and single ones; ``"nerve_placement"``: nerves and their random
    placements; ``"myelination"``: myelinated and non-myelinated nerves; ``"niche"``: perineural and other
    macrophages.
    Xenium panels: ``"xenium"`` (Level-3 types under atlas and display names), ``"xenium_dataset"`` (healthy
    section and rejection cohort), ``"lymphoid"`` (lymphoid Level 4), ``"immune_class"`` (cell classes of a core
    as in Fig 6 c), ``"composition"`` (immune classes of a core), ``"lineage"``, ``"rejection"``,
    ``"resolution"``, ``"grade"`` and ``"focus"`` (immune foci).
    Deconvolution panels: ``"compartment"`` (compartments of the composition bars), ``"method"``
    (deconvolution methods), ``"chamber_genes"`` (atrial and ventricular genes), ``"chamber_region"``
    (spots of atria and ventricles) and ``"cm_dominant"`` (cardiomyocyte-dominant spots against the others).
    ``"spillover"``: the four CMFLAG spillover flags on tissue.
    Atlas donor panels: ``"sex"``, ``"age_direction"`` and ``"age_direction_light"`` (direction of an age effect,
    light when significant after FDR only) and ``"cell_family"`` (cell families of the Level-4 states).
    """
    return PALETTES[kind]


def set_colors(adata, key: str, palette, default: str | None = None) -> None:
    """Colour the categories of ``adata.obs[key]`` in scanpy and squidpy plots.

    Writes ``adata.uns[f"{key}_colors"]``, one colour per category in the order of the categories, as scanpy
    expects it. A column that is not categorical is made categorical first.

    Parameters
    ----------
    adata
        AnnData with the column in ``obs``.
    key
        Column of ``obs``.
    palette
        Category -> colour, e.g. :data:`SPATIAL` or :data:`WALL_COMPARTMENT`.
    default
        Colour of a category missing from `palette`; without it, a missing category raises a KeyError.
    """
    if adata.obs[key].dtype.name != "category":
        adata.obs[key] = adata.obs[key].astype("category")
    categories = adata.obs[key].cat.categories
    if default is None:
        colours = [palette[name] for name in categories]
    else:
        colours = [palette.get(name, default) for name in categories]
    adata.uns[f"{key}_colors"] = colours


def lighten(colour, amount: float = 0.5) -> str:
    """A colour mixed with white (`amount` = share of white), as hex."""
    r, g, b = to_rgb(colour)
    return to_hex((r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount))


def ct(name: str, default: str = "#B9B9B9", aliases: dict[str, str] | None = None) -> str:
    """Colour of a Level-3 cell type; short aliases such as "VSMC" and the names in `aliases` are accepted."""
    if name in CT:
        return CT[name]
    if name in ALIASES:
        return CT[ALIASES[name]]
    if aliases and name in aliases:
        return CT[aliases[name]]
    return default


# matplotlib settings of the panels that are drawn at their final size: the font sizes are the sizes on the page
PANEL_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "axes.titleweight": "regular",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "xtick.color": INK,
    "ytick.color": INK,
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


def apply() -> None:
    """Set the matplotlib defaults of the panels (:data:`PANEL_RC`)."""
    mpl.rcParams.update(PANEL_RC)


def notebook() -> None:
    """Look of the exploratory plots in the notebooks: the font of the panels, open axes, white background."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.titleweight": "regular",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
        }
    )
