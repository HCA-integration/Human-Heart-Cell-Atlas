"""ED6 e: concordance between the atlas-guided Visium HD FICTURE factors and the snRNA Level-3 atlas marker genes
(Pearson r in marker-gene specificity space; rows = FICTURE factors, columns = atlas types; diagonal boxed, a dot
marks the best-matching type where it is off the diagonal).

Definition: scripts/cardiomyocyte/ed6e_concordance.py; variant: config panels.ed6e_variant (posterior = FICTURE
decode posterior counts per factor, 11 slides).

Inputs: results/cardiomyocyte/ed6e/concordance_<variant>.csv
Outputs: results/figures/ED6/panel_e.{png,pdf}

Usage: python figures/ED6/panel_e.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import hhca
from hhca.config import load_config
from hhca.pl import save_panel


def main():
    """Draw the factor x atlas-type correlation matrix."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    cfg = load_config("cardiomyocyte")
    variant = cfg["panels"]["ed6e_variant"]
    r = pd.read_csv(hhca.results_dir("cardiomyocyte", "ed6e", f"concordance_{variant}.csv"), index_col=0)
    fig = hhca.pl.factor_concordance(r, cfg["short_names"])
    print(save_panel(fig, "ED6", "e"))


if __name__ == "__main__":
    main()
