"""Fig 4h: vascular marker genes, connexins, junctions and caveolins across vessel-wall compartments.

Heatmap of the row z-scores of mean log1p CP10k on the pooled 29-gene matrix, gene-group colour bars and key,
tall colourbar.

Inputs: results/vascular/markers/connexin_tj_pooled_matrix.csv (scripts/vascular/marker_pool.py)
Outputs: results/figures/Fig4/panel_h.{png,pdf}

Usage: python figures/Fig4/panel_h.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.pl import save_panel


def main():
    """Draw and save the panel."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    pooled = pd.read_csv(hhca.results_dir("vascular", "markers", "connexin_tj_pooled_matrix.csv"))
    fig = hhca.pl.wall_markers(pooled)
    save_panel(fig, "Fig4", "h")


if __name__ == "__main__":
    main()
