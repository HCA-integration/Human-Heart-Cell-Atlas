"""Fig 6 h: cytokine-response programme medians by rejection type (row z-score).

Stars: Mann-Whitney U of each rejection type vs no rejection, Benjamini-Hochberg over the 54 tests.

Inputs:
    results/xenium/downstream/stats/fig6_h_matrix.csv, fig6_h_tests.csv (fig6_stats.py)

Outputs:
    results/figures/Fig6/panel_h.{png,pdf}

Usage: python figures/Fig6/panel_h.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.pl import save_panel


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    stats = hhca.results_dir("xenium", "downstream", "stats")
    zscores = pd.read_csv(f"{stats}/fig6_h_matrix.csv", index_col=0)
    tests = pd.read_csv(f"{stats}/fig6_h_tests.csv")
    fig = hhca.pl.programme_heatmap(zscores, tests)
    print(f"{int((tests.q < 0.05).sum())}/{len(tests)} significant ->", save_panel(fig, "Fig6", "h"))


if __name__ == "__main__":
    main()
