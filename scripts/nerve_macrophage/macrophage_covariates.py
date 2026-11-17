#!/usr/bin/env python
"""Per-cell covariates of the macrophage pseudo-cells (Fig 5 k-l): distances to nerve and vessel bins, markers.

Distances are measured from each pseudo-cell centroid to the nearest 2 um bin of a class in the CMFLAG-clean
2 um h5ads (obs L3_label, x_um, y_um; the same micrometre frame as the centroids; hhca.tl.distance_to_bins):
dist_nerve_um (Neural Cells), dist_vessel_pix_um (VSMC, pericytes, capillary, arterial, venous, lymphatic and
endocardial EC) and dist_mural_um (VSMC, arterial and venous EC). Markers (LYVE1, FOLR2, MRC1, F13A1, SPP1, CCR2,
C1QA, VCAN) are given per cell as raw counts, log1p CP10k and detection (hhca.tl.marker_expression). The nearest
vessel structure and its calibre class (dist_vessel_struct_um, class_near) are carried over from the pseudo-cell
h5ad. Types and markers: config `macrophage`.

Inputs: the pooled pseudo-cell h5ad of macrophage_pseudocells.py and the 2 um h5ads.
Outputs: <outdir>/tables/macrophage_percell_covariates.csv

Usage: python macrophage_covariates.py --cache Macrophage_pseudocells_allslides.h5ad --h5ad '<dir>/{s}.h5ad'
           --outdir DIR
"""

import argparse
import os

from hhca import thread_env

os.environ.update(thread_env(1))

import anndata as ad
import pandas as pd

import hhca
from hhca.config import load_config

# columns of the pseudo-cell h5ad and the distances written to the table, followed by the marker columns
COLUMNS = [
    "slide",
    "cell_type",
    "centroid_x_um",
    "centroid_y_um",
    "n_bins",
    "total_counts",
    "dist_vessel_struct_um",
    "diam_near",
    "class_near",
    "dist_vessel_pix_um",
    "dist_mural_um",
    "dist_nerve_um",
]


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", required=True, help="pooled macrophage pseudo-cell h5ad")
    parser.add_argument("--h5ad", required=True, help="CMFLAG-clean 2 um h5ad, path template with {s} for the slide")
    parser.add_argument("--outdir", required=True, help="output folder; tables/ is created in it")
    return parser.parse_args()


def main():
    """Write the per-cell covariate table of the pooled macrophage pseudo-cells."""
    args = parse_args()
    cfg = load_config("nerve_macrophage")
    macrophage = cfg["macrophage"]
    os.makedirs(f"{args.outdir}/tables", exist_ok=True)
    cells = ad.read_h5ad(args.cache)
    expression = hhca.tl.marker_expression(cells, macrophage["covariate_markers"])
    table = pd.concat([cells.obs, expression], axis=1)
    sections = {slide: ad.read_h5ad(args.h5ad.format(s=slide), backed="r") for slide in table.slide.unique()}
    n_neural = hhca.tl.distance_to_bins(table, sections, cfg["nerve"]["neural_type"], "dist_nerve_um")
    hhca.tl.distance_to_bins(table, sections, macrophage["vessel_bins"]["broad"], "dist_vessel_pix_um")
    hhca.tl.distance_to_bins(table, sections, macrophage["vessel_bins"]["mural"], "dist_mural_um")
    print("neural bins per slide:", n_neural.to_dict())
    print("slides without neural bins:", list(n_neural.index[n_neural == 0]))
    table = table[COLUMNS + list(expression.columns)].copy()
    table.to_csv(f"{args.outdir}/tables/macrophage_percell_covariates.csv")
    print("\nwrote", f"{args.outdir}/tables/macrophage_percell_covariates.csv", table.shape)
    print(table[["dist_vessel_struct_um", "dist_vessel_pix_um", "dist_mural_um", "dist_nerve_um"]].describe())


if __name__ == "__main__":
    main()
