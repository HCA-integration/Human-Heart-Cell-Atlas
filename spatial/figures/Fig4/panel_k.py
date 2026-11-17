"""Fig 4k: calibre selectivity of vascular drug-target classes (large minus small, 11 sections).

Colours: mural targets in the VSMC colour, endothelial targets in the arterial-EC colour; stars from the
BH-adjusted p. Drug scores: hand-curated ChEMBL target-gene sets per class, mean log1p CP10k per structure,
OLS large vs small with richness and slide, BH over the 11 classes.

Inputs: results/vascular/drug/clean/tables/drug_class_by_caliber_summary.csv (scripts/vascular/drug_caliber.py)
Outputs: results/figures/Fig4/panel_k.{png,pdf}

Usage: python figures/Fig4/panel_k.py
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
    summary = pd.read_csv(hhca.results_dir("vascular", "drug", "clean", "tables", "drug_class_by_caliber_summary.csv"))
    fig = hhca.pl.drug_calibre(summary)
    save_panel(fig, "Fig4", "k")


if __name__ == "__main__":
    main()
