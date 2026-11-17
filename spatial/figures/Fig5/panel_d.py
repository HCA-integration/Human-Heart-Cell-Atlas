"""Fig 5 d: proximity of nerves to vessels.

Per nerve, the shortest distance from a nerve's 2 um bins to the nearest vascular bin, against the same nerve shapes
placed at random tissue positions; overlaid density histograms, medians, 8 / 16 um lines.

Inputs: results/nerve_macrophage/d/per_nerve.npz (scripts/nerve_macrophage/nerve_vessel_distance_pernerve.py)
Outputs: results/figures/Fig5/panel_d.{png,pdf}

Usage: python figures/Fig5/panel_d.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import numpy as np

import hhca
from hhca.config import load_config
from hhca.pl import save_panel


def main():
    """Draw and save the panel."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    distances = np.load(hhca.results_dir("nerve_macrophage", "d", "per_nerve.npz"))
    near_um = load_config("nerve_macrophage")["proximity"]["near_um"]
    fig = hhca.pl.nerve_vessel_distance(distances["d_nerve"], distances["d_background"], near_um)
    print(save_panel(fig, "Fig5", "d"))


if __name__ == "__main__":
    main()
