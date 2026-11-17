"""Fig 5 j: FICTURE-called macrophages and vascular cell types on the microscopy H&E of VS_BO_HDZ150319_34_myo.

Macrophages (pink), capillary EC (green), pericytes (beige), VSMC (purple) and arterial EC (lilac); in-tissue 2 um
bins without a cell-type assignment hatched (no decode pixel, or removed by the four CMFLAG spillover flags; Fig 4 f
convention). Bins = CMFLAG-clean 2 um h5ad, H&E through the CMFLAG registration. Fields = config fields.j.

Inputs: results/nerve_macrophage/fields/j_<id>.npz (scripts/nerve_macrophage/fields.py), config/nerve_macrophage.yaml
Outputs: results/figures/Fig5/panel_j.{png,pdf}

Usage: python figures/Fig5/panel_j.py
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
    cfg = load_config("nerve_macrophage")["fields"]["j"]
    fields = [
        np.load(hhca.results_dir("nerve_macrophage", "fields", f"j_{field_id}.npz"), allow_pickle=False)
        for field_id in cfg["fields"]
    ]
    fig = hhca.pl.macrophage_fields(fields, cfg["show"])
    print(save_panel(fig, "Fig5", "j"))


if __name__ == "__main__":
    main()
