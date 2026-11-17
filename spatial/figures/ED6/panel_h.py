"""ED6 h: mean cell-type composition per deconvolution method (11 slides, unweighted mean of the slide means).

The 11 compartments are the groups of Level-3 types of config/deconvolution.yaml (comparison.compartments).

Inputs:  results/deconvolution/tables_precomputed/composition_mean_by_method.csv
Outputs: results/figures/ED6/panel_h.{png,pdf}

Usage: python figures/ED6/panel_h.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.pl import save_panel


def main():
    """Draw the stacked compartment bars, one bar per method."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    composition = pd.read_csv(
        hhca.results_dir("deconvolution", "tables_precomputed", "composition_mean_by_method.csv"), index_col=0
    )
    fig = hhca.pl.method_composition(composition)
    print(save_panel(fig, "ED6", "h"))


if __name__ == "__main__":
    main()
