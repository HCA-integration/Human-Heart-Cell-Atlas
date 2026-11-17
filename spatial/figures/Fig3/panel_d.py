"""Fig 3 d: ventricular-cardiomyocyte stress score across the wall of donor H0054 (sections near the epicardium,
in the mid-myocardium and near the endocardium), CMFLAG-clean vCM 2 um bins on a 5 um display grid.

Each 650 um field is shown as its left 255 um x lower 575 um strip, magma, colour range -0.13 to 0.30 (config
fig3d). Score: mean z of log1p CP10k over the gene set of config panels.fig3d_geneset against the pooled H0054
reference (scripts/cardiomyocyte/stress_score.py).

Inputs: results/cardiomyocyte/fig3d/stress.npz
Outputs: results/figures/Fig3/panel_d.{png,pdf}

Usage: python figures/Fig3/panel_d.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import numpy as np

import hhca
from hhca.config import load_config
from hhca.pl import save_panel


def main():
    """Draw the three strips with a scale bar and the colour bar."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    cfg = load_config("cardiomyocyte")
    geneset, params = cfg["panels"]["fig3d_geneset"], cfg["fig3d"]
    stress = np.load(hhca.results_dir("cardiomyocyte", "fig3d", "stress.npz"))
    grids = {layer: stress[f"{geneset}_{layer}_grid"] for layer in params["sections"]}
    fields = {layer: stress[f"{geneset}_{layer}_ext"] for layer in params["sections"]}
    fig = hhca.pl.stress_fields(grids, fields, params["strip"], params["vmin_vmax"])
    print(save_panel(fig, "Fig3", "d"))


if __name__ == "__main__":
    main()
