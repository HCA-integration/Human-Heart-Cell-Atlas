"""Figure panels, in the style of ``scanpy.pl``: each function draws one publication panel from the objects of the
analysis and returns the figure.

    import hhca

    fig = hhca.pl.composition_along_calibre(bins, edges)
    hhca.pl.save_panel(fig, "Fig4", "g")  # results/figures/Fig4/panel_g.{png,pdf}

The same functions are called by the panel scripts in ``figures/`` and by the notebooks, where the returned
:class:`PanelFigure` is shown as it is saved.

Layout of the package
    ``hhca.pl.<panel>``   panel functions, one module per analysis, imported on first use: ``_cardiomyocytes.py``
                          (Fig 1 j, Fig 3 d-l, ED6 d-f), ``_vessels.py`` (Fig 4 f-k), ``_nerves.py`` (Fig 5 b-h,
                          j-l), ``_deconvolution.py`` (Fig 3 c, ED6 c, g, h, ED8 b-c), ``_xenium.py`` (Fig 6 c-l,
                          ED9 a-c, ED6 i-j), ``_atlas.py`` (Fig 2 h-l, Fig 4 l), ``_age.py`` (Fig 5 m, n) and
                          ``_celltyping.py`` (H&E fields of the cell-typing notebook)
    ``hhca.pl.PanelFigure``   the figure of one panel (``_panel.py``)
    ``hhca.pl.save_panel``, ``figure_dir``, ``save_figure``   writing panels (``_save.py``)
    ``hhca.pl.style``     palettes, colour maps and matplotlib defaults of the panels and the notebooks
"""

import importlib

from hhca.pl import style
from hhca.pl._panel import PanelFigure
from hhca.pl._save import figure_dir, save_figure, save_panel

# panel function -> module that defines it
_PANELS = {
    "celltype_fields": "_cardiomyocytes",
    "expressing_bins": "_cardiomyocytes",
    "factor_concordance": "_cardiomyocytes",
    "label_markers": "_cardiomyocytes",
    "niche_distance": "_cardiomyocytes",
    "section_map": "_cardiomyocytes",
    "slope_concordance": "_cardiomyocytes",
    "stress_fields": "_cardiomyocytes",
    "transmural_modules": "_cardiomyocytes",
    "composition_along_calibre": "_vessels",
    "drug_calibre": "_vessels",
    "raas_heatmap": "_vessels",
    "raas_means": "_vessels",
    "vessel_fields": "_vessels",
    "wall_markers": "_vessels",
    "macrophage_fields": "_nerves",
    "myelin_on_he": "_nerves",
    "myelination_by_size": "_nerves",
    "nerve_on_he": "_nerves",
    "nerve_size_effect": "_nerves",
    "nerve_vessel_distance": "_nerves",
    "nerves_and_vessels": "_nerves",
    "neurotoxic_drugs": "_nerves",
    "neurotoxic_score": "_nerves",
    "perineural_counts": "_nerves",
    "perineural_markers": "_nerves",
    "chamber_markers": "_deconvolution",
    "method_composition": "_deconvolution",
    "method_maps": "_deconvolution",
    "pericyte_signature": "_deconvolution",
    "spot_pies": "_deconvolution",
    "atg_score": "_xenium",
    "core_lineages": "_xenium",
    "core_overview": "_xenium",
    "drug_targets": "_xenium",
    "foci_by_grade": "_xenium",
    "healthy_section": "_xenium",
    "pre_post": "_xenium",
    "programme_heatmap": "_xenium",
    "programme_scores": "_xenium",
    "slide_immune_cells": "_xenium",
    "hormone_dots": "_atlas",
    "hormone_relay": "_atlas",
    "sex_boxes": "_atlas",
    "sex_by_age": "_atlas",
    "age_programmes": "_age",
    "age_trajectories": "_age",
    "he_fields": "_celltyping",
}

__all__ = ["PanelFigure", "figure_dir", "save_figure", "save_panel", "style", *sorted(_PANELS)]


def __getattr__(name):
    if name in _PANELS:
        return getattr(importlib.import_module(f"hhca.pl.{_PANELS[name]}"), name)
    raise AttributeError(f"module 'hhca.pl' has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_PANELS])
