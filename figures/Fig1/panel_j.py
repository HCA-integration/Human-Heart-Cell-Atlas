"""Fig 1 j: four 92 um fields of H0054 endo, microscopy H&E with the CMFLAG-clean FICTURE 2 um bins on top.

The fields are the gallery crops located on the registered H&E (scripts/cardiomyocyte/fig1j_locate.py); their
centres are set in config/cardiomyocyte.yaml (fig1j.centres_um). Cardiomyocyte bins are not drawn, other Level-3
bins are semi-transparent and the target type is opaque. The legend label "Neural cells" stands for the HHCA
class Neural Cells (glia / Schwann cells).

Inputs: results/cardiomyocyte/fields/fig1j_<name>.npz (scripts/cardiomyocyte/extract_fields.py).
Outputs: results/figures/Fig1/panel_j.{png,pdf}

Usage: python figures/Fig1/panel_j.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import numpy as np

import hhca
from hhca.config import load_config
from hhca.pl import save_panel


def main():
    """Draw the four fields side by side."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    cfg = load_config("cardiomyocyte")
    tiles = cfg["fig1j"]["tiles"]
    fields = {
        name: np.load(hhca.results_dir("cardiomyocyte", "fields", f"fig1j_{name}.npz"), allow_pickle=False)
        for name in tiles
    }
    fig = hhca.pl.celltype_fields(fields, tiles)
    print(save_panel(fig, "Fig1", "j"))


if __name__ == "__main__":
    main()
