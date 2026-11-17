"""Fig 5 m: Spearman rho of the per-donor cytokine-response score against age, per immune cell type (280 donors).

Inputs: results/atlas/age_cytokine_effects_280.csv.
Outputs: results/figures/Fig5/panel_m.{png,pdf}.

Usage: python figures/Fig5/panel_m.py
"""

import argparse

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import hhca
from hhca.config import load_config
from hhca.pl import save_panel


def main():
    """Draw panel m."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    params = load_config("config")["cytokine"]
    effects = pd.read_csv(hhca.results_dir("atlas", "age_cytokine_effects_280.csv"))
    matrix = hhca.tl.programme_matrix(effects, params["age_cell_types"], params["alpha"])
    print("programmes kept:", list(matrix["rho"].index))
    save_panel(hhca.pl.age_programmes(matrix), "Fig5", "m")


if __name__ == "__main__":
    main()
