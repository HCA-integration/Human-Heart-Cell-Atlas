"""Fig 5 b: nerve next to a vessel on the microscopy H&E (top) and the same field with the neural 2 um bins (bottom).

Slide VS_BO_HDZ150319_34_myo; bins = CMFLAG-clean 2 um h5ad, label Neural Cells; H&E through the CMFLAG
registration. Field = config fields.b.

Inputs: results/nerve_macrophage/fields/b_field.npz (scripts/nerve_macrophage/fields.py)
Outputs: results/figures/Fig5/panel_b.{png,pdf}

Usage: python figures/Fig5/panel_b.py
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
    field = np.load(hhca.results_dir("nerve_macrophage", "fields", "b_field.npz"), allow_pickle=False)
    fig = hhca.pl.nerve_on_he(field, load_config("nerve_macrophage")["nerve"]["neural_type"])
    print(save_panel(fig, "Fig5", "b"))


if __name__ == "__main__":
    main()
