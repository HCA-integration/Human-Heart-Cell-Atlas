"""Fig 6 f (immune fraction) and g (T-cell fraction): pre vs post, resolved vs persistence, 46 paired patients.

One line per patient from the pre- to the post-treatment biopsy; Wilcoxon signed-rank test pre vs post within
each outcome group.

Inputs:
    results/xenium/downstream/stats/fig6_fg_pairs.csv, fig6_stats.json (fig6_stats.py)

Outputs:
    results/figures/Fig6/panel_f.{png,pdf}, panel_g.{png,pdf}

Usage: python figures/Fig6/panel_fg.py
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.pl import save_panel


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    stats = hhca.results_dir("xenium", "downstream", "stats")
    pairs = pd.read_csv(f"{stats}/fig6_fg_pairs.csv")
    with open(f"{stats}/fig6_stats.json") as handle:
        tests = json.load(handle)["fg"]
    for column, ylabel, panel in [("immf", "immune fraction", "f"), ("tf", "T-cell fraction", "g")]:
        by_outcome = {group: tests[f"{panel}_{group}"] for group in ("resolved", "persistence")}
        fig = hhca.pl.pre_post(pairs, by_outcome, column, ylabel)
        print(save_panel(fig, "Fig6", panel))


if __name__ == "__main__":
    main()
