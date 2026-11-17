"""Fig 5 n: LOESS age trajectories of the macrophage cytokine-response programmes that change with age (280 donors).

Cell type, age grid and LOESS settings: config/config.yaml (cytokine.trajectories).

Inputs: results/atlas/age_cytokine_effects_280.csv, results/atlas/per_donor_cytokine_scores_280.npz.
Outputs: results/figures/Fig5/panel_n.{png,pdf}.

Usage: python figures/Fig5/panel_n.py
"""

import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import hhca
from hhca.config import load_config
from hhca.donors import read_programme_scores
from hhca.pl import save_panel


def main():
    """Draw panel n."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    cfg = load_config("config")
    params = cfg["cytokine"]["trajectories"]
    effects = pd.read_csv(hhca.results_dir("atlas", "age_cytokine_effects_280.csv"))
    significant = hhca.tl.changing_programmes(effects, params["cell_type"], cfg["cytokine"]["alpha"])
    print(significant[["feature", "q", "p_sens", "robust", "direction"]].to_string(index=False))
    donors, scores = read_programme_scores(hhca.results_dir("atlas", "per_donor_cytokine_scores_280.npz"))
    trajectories = hhca.tl.age_trajectories(
        scores,
        donors,
        list(significant["feature"]),
        params["cell_type"],
        np.linspace(*params["age_grid"]),
        frac=params["frac"],
        n_boot=params["n_boot"],
        seed=params["seed"],
        adult_age=cfg["donors"]["adult_age"],
    )
    save_panel(hhca.pl.age_trajectories(significant, trajectories), "Fig5", "n")


if __name__ == "__main__":
    main()
