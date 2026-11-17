"""Fig 3 e: NPPA- and NPPB-expressing 2 um bins over the FICTURE Level-3 map of the subendocardial field of
H0054 endo (500 um, centre (5250, 3250) um; config fig3e), CMFLAG-clean bins, Level-3 palette of the spatial panels.

Inputs: results/cardiomyocyte/fields/fig3e.npz (scripts/cardiomyocyte/extract_fields.py)
Outputs: results/figures/Fig3/panel_e.{png,pdf}

Usage: python figures/Fig3/panel_e.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import numpy as np

import hhca
from hhca.config import load_config
from hhca.pl import save_panel


def main():
    """Draw one map per gene: the Level-3 bins with the expressing bins on top."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    cfg = load_config("cardiomyocyte")
    genes = cfg["fig3e"]["genes"]
    field = np.load(hhca.results_dir("cardiomyocyte", "fields", "fig3e.npz"), allow_pickle=False)
    fig = hhca.pl.expressing_bins(field, genes)
    counts = {gene: int((field[gene] > 0).sum()) for gene in genes}
    print(save_panel(fig, "Fig3", "e"), "| bins", len(field["x"]), "|", counts)


if __name__ == "__main__":
    main()
