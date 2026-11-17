"""Fig 2 h, i: sex differences in hormone reception (h) and production (i) per cell system, LV nuclei, 280 donors.

Dots shown = relevance gate (scripts/atlas/hormone_relevance_gate.py); colour = female - male per-donor mean
score; size = mean of the two; outline = OLS score ~ C(sex) + age, p < 0.05
(scripts/atlas/hormone_sexdiff_significance.py).

Inputs: results/atlas/hormones/sexdiff_280/hormone_sexdiff_significance_ALLDOTS.csv,
        results/atlas/hormones/g7_sexdiff/hormone_sexdiff_relevancegated__{received,produced}.csv
Outputs: results/figures/Fig2/panel_{h,i}.{png,pdf}

Usage: python figures/Fig2/panel_hi.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.pl import save_panel


def main():
    """Draw panels h and i."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    folder = hhca.results_dir("atlas", "hormones")
    sex_differences = pd.read_csv(f"{folder}/sexdiff_280/hormone_sexdiff_significance_ALLDOTS.csv")
    for side, letter in (("received", "h"), ("produced", "i")):
        gate = pd.read_csv(f"{folder}/g7_sexdiff/hormone_sexdiff_relevancegated__{side}.csv")
        print(save_panel(hhca.pl.hormone_dots(sex_differences, gate, side), "Fig2", letter))


if __name__ == "__main__":
    main()
