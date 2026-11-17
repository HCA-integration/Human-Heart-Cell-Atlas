"""ED8 b: atrial-pericyte signature genes by chamber on the Kanemaru Visium sections.

Mean z-scored expression of the 29 signature genes measured in Visium over all spots of LA, RA, LV and RV
(the first part of the figure of scripts/deconvolution/build_ED8c.py, same geometry).

Inputs:  results/deconvolution/downstream/ed8/ED8c_gene_by_chamber_z.csv
Outputs: results/figures/ED8/panel_b.{png,pdf}

Usage: python figures/ED8/panel_b.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.pl import save_panel


def main():
    """Draw the signature genes x chambers heatmap."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    chamber_means = pd.read_csv(
        hhca.results_dir("deconvolution", "downstream", "ed8", "ED8c_gene_by_chamber_z.csv"), index_col=0
    )
    fig = hhca.pl.pericyte_signature(chamber_means, show=["genes"])
    print(save_panel(fig, "ED8", "b"), f"| {chamber_means.shape[0]} genes")


if __name__ == "__main__":
    main()
