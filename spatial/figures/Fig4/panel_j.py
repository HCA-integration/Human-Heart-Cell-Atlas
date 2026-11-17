"""Fig 4j: RAAS marginal means on the adjusted-CP10k scale (ACE, ACE2, AGTR1; AGT on the right axis).

adj_cp10k with its 95 % interval: ACE / ACE2 / AGTR1 on the left axis, AGT as a grey band on the right axis;
an artery-capillary-vein sketch along the top.

Inputs: results/vascular/raas/model/raas_emm_by_gene_vessel.csv (scripts/vascular/raas_model.py)
Outputs: results/figures/Fig4/panel_j.{png,pdf}

Usage: python figures/Fig4/panel_j.py
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
    fig = hhca.pl.raas_means(emm)
    save_panel(fig, "Fig4", "j")


if __name__ == "__main__":
    main()
