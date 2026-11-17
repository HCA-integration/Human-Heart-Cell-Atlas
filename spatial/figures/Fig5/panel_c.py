"""Fig 5 c: neural (red stars) and vascular (blue squares) 2 um bins in an 800 um window on the registered H&E.

Slide VS_BO_HDZ150319_34_myo; bins = CMFLAG-clean 2 um h5ad of the full-slide decode; vascular = arterial,
capillary, venous EC, VSMC, pericytes (config proximity.vascular_types); microscopy H&E through the CMFLAG
registration.

Inputs: results/nerve_macrophage/fields/c_window.npz (scripts/nerve_macrophage/fields.py)
Outputs: results/figures/Fig5/panel_c.{png,pdf}

Usage: python figures/Fig5/panel_c.py
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
    cfg = load_config("nerve_macrophage")
    field = np.load(hhca.results_dir("nerve_macrophage", "fields", "c_window.npz"), allow_pickle=False)
    fig = hhca.pl.nerves_and_vessels(field, cfg["proximity"]["vascular_types"], cfg["nerve"]["neural_type"])
    print(save_panel(fig, "Fig5", "c"))


if __name__ == "__main__":
    main()
