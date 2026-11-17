"""ED6 f: whole-section atlas-guided FICTURE at 2 um on H0054 endo (CMFLAG-clean bins; cardiomyocytes light
grey) with a 340 um vascular detail field (config ed6f).

The whole section is drawn from a 4 um label raster (a pixel shows the most frequent non-cardiomyocyte label of
its bins, else cardiomyocyte), the detail field from the bins.

Inputs: results/cardiomyocyte/fields/ed6f.npz (scripts/cardiomyocyte/extract_fields.py)
Outputs: results/figures/ED6/panel_f.{png,pdf}

Usage: python figures/ED6/panel_f.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import numpy as np

import hhca
from hhca.pl import save_panel


def main():
    """Draw the whole section, the detail field and the lines connecting them."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    section = np.load(hhca.results_dir("cardiomyocyte", "fields", "ed6f.npz"), allow_pickle=False)
    zx, zy, zw = section["zoom"]
    half = zw / 2
    detail = {
        "x": section["zx"],
        "y": section["zy"],
        "code": section["zcode"],
        "cats": section["cats"],
        "ext": np.array([zx - half, zx + half, zy - half, zy + half]),
    }
    fig = hhca.pl.section_map(section, detail)
    print(save_panel(fig, "ED6", "f"))


if __name__ == "__main__":
    main()
