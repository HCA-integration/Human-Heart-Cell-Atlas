"""Fig 6 d: pre-treatment core overview (clinical tracks, immune fraction, foci, largest focus, Level-4 composition).

Inputs:
    results/xenium/downstream/stats/fig6d_overview_pre.csv (fig6d_overview.py)

Outputs:
    results/figures/Fig6/panel_d.{png,pdf}

Usage: python figures/Fig6/panel_d.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.pl import save_panel


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    overview = pd.read_csv(hhca.results_dir("xenium", "downstream", "stats", "fig6d_overview_pre.csv"))
    fig = hhca.pl.core_overview(overview, "pre")
    print(f"pre: {len(overview)} cores ->", save_panel(fig, "Fig6", "d"))


if __name__ == "__main__":
    main()
