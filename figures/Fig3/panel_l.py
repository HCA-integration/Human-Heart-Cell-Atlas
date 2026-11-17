"""Fig 3 l: DVP protein vs Visium HD transcript transmural slopes of cardiomyocyte genes (Spearman rho, p, sign
agreement; OLS dashed line; sign-concordant quadrants shaded).

Substrate and gene set come from config panels: fig3l_substrate (cmflag_2um_3donor = 2 um CMFLAG-clean ventricular
CM bins of H0054, H0063 and AH10) and fig3l_gene_set (dvp_rule_cmsafe = Methods DVP rule and cm_safe).

Inputs: results/cardiomyocyte/fig3l/fig3l_concordance_genes.csv (scripts/cardiomyocyte/fig3l_concordance.py)
Outputs: results/figures/Fig3/panel_l.{png,pdf}

Usage: python figures/Fig3/panel_l.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.config import load_config
from hhca.pl import save_panel


def main():
    """Scatter the protein against the transcript slopes of the configured gene set."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    choice = load_config("cardiomyocyte")["panels"]
    slopes = pd.read_csv(hhca.results_dir("cardiomyocyte", "fig3l", "fig3l_concordance_genes.csv"))
    slopes = slopes[(slopes.substrate == choice["fig3l_substrate"]) & (slopes.gene_set == choice["fig3l_gene_set"])]
    fig = hhca.pl.slope_concordance(slopes)
    print(save_panel(fig, "Fig3", "l"))


if __name__ == "__main__":
    main()
