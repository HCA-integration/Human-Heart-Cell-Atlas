"""Fig 5 l: macrophage marker difference, perineural vs perivascular + interstitial pseudo-cells.

Horizontal bars for SPP1 LYVE1 F13A1 C1QA MRC1 FOLR2 (config macrophage.markers). Slide-adjusted OLS of log1p CP10k
on the perineural indicator (+ log library, slide fixed effects, HC3 errors); bar = coefficient (difference in log1p
CP10k), stars = BH over the six markers.

Inputs: results/nerve_macrophage/kl/l_regression.csv (scripts/nerve_macrophage/perineural_de.py)
Outputs: results/figures/Fig5/panel_l.{png,pdf}

Usage: python figures/Fig5/panel_l.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.config import load_config
from hhca.pl import save_panel


def main():
    """Draw and save the panel."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    regression = pd.read_csv(hhca.results_dir("nerve_macrophage", "kl", "l_regression.csv"))
    effects = regression.rename(columns={"beta_perineural": "beta", "padj_perineural": "padj"})
    fig = hhca.pl.perineural_markers(effects, load_config("nerve_macrophage")["macrophage"]["markers"])
    print(save_panel(fig, "Fig5", "l"))


if __name__ == "__main__":
    main()
