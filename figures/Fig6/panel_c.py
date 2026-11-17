"""Fig 6 c: whole-slide Xenium TMA with HHCA-transferred immune labels, lymphoid at Level 4, one high-grade zoom.

The slide with the most cores is drawn with structural cells in grey and immune cells coloured by type; the
zoom shows one core with high-grade cellular rejection.

Inputs:
    results/xenium/downstream/tables/spatial_cells.csv.gz (spatial_cells.py)
    config/xenium_downstream.yaml: figures.fig6c_zoom_core, labels.lymphoid (6 classes, no Treg),
    classes.slide_immune

Outputs:
    results/figures/Fig6/panel_c.{png,pdf}

Usage: python figures/Fig6/panel_c.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.config import load_config
from hhca.pl import save_panel


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    cfg = load_config("xenium_downstream")
    cells = pd.read_csv(hhca.results_dir("xenium", "downstream", "tables", "spatial_cells.csv.gz"))
    zoom_core = cfg["figures"]["fig6c_zoom_core"]
    fig = hhca.pl.slide_immune_cells(cells, zoom_core, cfg["labels"]["lymphoid"], cfg["classes"]["slide_immune"])
    print("zoom core:", zoom_core, "->", save_panel(fig, "Fig6", "c"))


if __name__ == "__main__":
    main()
