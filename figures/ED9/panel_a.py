"""ED9 a: post-treatment core overview (the drawing of Fig 6 d with timing = post).

Inputs:
    results/xenium/downstream/stats/fig6d_overview_post.csv (fig6d_overview.py)

Outputs:
    results/figures/ED9/panel_a.{png,pdf}

Usage: python figures/ED9/panel_a.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.pl import save_panel


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    overview = pd.read_csv(hhca.results_dir("xenium", "downstream", "stats", "fig6d_overview_post.csv"))
    fig = hhca.pl.core_overview(overview, "post")
    print(f"post: {len(overview)} cores ->", save_panel(fig, "ED9", "a"))


if __name__ == "__main__":
    main()
