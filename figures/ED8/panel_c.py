"""ED8 c: atrial-pericyte signature score in pericyte-high Kanemaru Visium spots, ventricles vs atria.

Pericyte-high = top within-region tertile of the total cell2location pericyte abundance; two-sided
Mann-Whitney U test and difference of medians (the third part of the figure of
scripts/deconvolution/build_ED8c.py, same geometry and the same jitter of the points).

Inputs:  results/deconvolution/downstream/ed8/ED8c_perspot.csv.gz
Outputs: results/figures/ED8/panel_c.{png,pdf}

Usage: python figures/ED8/panel_c.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.pl import save_panel


def main():
    """Draw the box plot of the signature score in pericyte-high spots, ventricles vs atria."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    spots = pd.read_csv(hhca.results_dir("deconvolution", "downstream", "ed8", "ED8c_perspot.csv.gz"), index_col=0)
    fig = hhca.pl.pericyte_signature(spots=spots, show=["chambers"])
    print(save_panel(fig, "ED8", "c"))


if __name__ == "__main__":
    main()
