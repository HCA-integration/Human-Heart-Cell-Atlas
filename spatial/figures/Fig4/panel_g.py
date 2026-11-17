"""Fig 4g: vascular composition of vessel structures along calibre (11 sections pooled).

Mean % per calibre bin with its 95% CI (bins with >= 20 structures), log calibre axis, larger vessels left.

Inputs: results/vascular/composition/composition_by_diameter_bin.csv (scripts/vascular/composition_curves.py).
Outputs: results/figures/Fig4/panel_g.{png,pdf}

Usage: python figures/Fig4/panel_g.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import hhca
from hhca.config import load_config
from hhca.pl import save_panel


def main():
    """Draw and save the panel."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    bins = pd.read_csv(hhca.results_dir("vascular", "composition", "composition_by_diameter_bin.csv"))
    diameter_bins = load_config("vascular")["composition"]["diameter_bins"]
    edges = np.geomspace(diameter_bins["low"], diameter_bins["high"], diameter_bins["n_edges"])
    fig = hhca.pl.composition_along_calibre(bins, edges)
    save_panel(fig, "Fig4", "g")


if __name__ == "__main__":
    main()
