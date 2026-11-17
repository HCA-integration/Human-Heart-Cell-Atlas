"""Fig 6 k: immunosuppressant drug-target score by cell type (lymphoid at Level 4), z-scored per drug across cell types.

Curated drug -> target-gene sets scored with sc.tl.score_genes (not drug2cell / ChEMBL); cell types at lymphoid
Level 4 and Level 3 for the rest; 16 columns (no Treg).

Inputs:
    results/xenium/downstream/tables/drug_target_matrix_z.csv (drug_targets.py)

Outputs:
    results/figures/Fig6/panel_k.{png,pdf}

Usage: python figures/Fig6/panel_k.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.pl import save_panel


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    zscores = pd.read_csv(hhca.results_dir("xenium", "downstream", "tables", "drug_target_matrix_z.csv"), index_col=0)
    fig = hhca.pl.drug_targets(zscores)
    print(zscores.shape, "->", save_panel(fig, "Fig6", "k"))


if __name__ == "__main__":
    main()
