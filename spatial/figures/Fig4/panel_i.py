"""Fig 4i: RAAS row z-scores (ACE, AGTR1, ACE2, ENPEP) of the richness-adjusted marginal means across vessel types.

Peak cell per gene outlined; artery-vein colour bar below.

Inputs: results/vascular/raas/model/raas_emm_by_gene_vessel.csv (scripts/vascular/raas_model.py)
Outputs: results/figures/Fig4/panel_i.{png,pdf}

Usage: python figures/Fig4/panel_i.py
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
    emm = pd.read_csv(hhca.results_dir("vascular", "raas", "model", "raas_emm_by_gene_vessel.csv"))
    fig = hhca.pl.raas_heatmap(emm)
    save_panel(fig, "Fig4", "i")


if __name__ == "__main__":
    main()
