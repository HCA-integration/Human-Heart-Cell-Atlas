"""Fig 5 f: myelinating nerve on the microscopy H&E (top) and its neural bins coloured by myelin score (bottom).

Slide VS_BO_HDZ150319_34_myo. Bottom: faded H&E with the neural 2 um bins coloured by the myelination marker score
(MBP, PLP1, MOG, MPZ, PMP22 counts within 4 um of the bin), shared colour bar. H&E through the CMFLAG registration,
bins = CMFLAG-clean 2 um h5ad. Field = config fields.f. Colour scale 0 .. max(2, p95 of the positive bin scores in
the field).

Inputs: results/nerve_macrophage/fields/f_field.npz (scripts/nerve_macrophage/fields.py)
Outputs: results/figures/Fig5/panel_f.{png,pdf}

Usage: python figures/Fig5/panel_f.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import numpy as np

import hhca
from hhca.pl import save_panel


def main():
    """Draw and save the panel."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    field = np.load(hhca.results_dir("nerve_macrophage", "fields", "f_field.npz"), allow_pickle=False)
    fig = hhca.pl.myelin_on_he(field)
    print(save_panel(fig, "Fig5", "f"))


if __name__ == "__main__":
    main()
