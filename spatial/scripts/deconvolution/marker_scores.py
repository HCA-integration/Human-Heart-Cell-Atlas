#!/usr/bin/env python
"""Per-spot marker scores of every cell type, used to check estimated proportions against marker support.

Markers = the 15 genes per Level-3 type with the largest log fold change against the other types in the shared
reference (log1p CP10k); scores = scanpy score_genes on each slide (types with fewer than 3 markers on the
slide are NaN).

Inputs:  the shared reference h5ad, <data>/<slide>/slide.h5ad
Outputs: <outdir>/<slide>.csv.gz (spots x cell types), the marker lists (--markers-json)

Usage: marker_scores.py --ref REF.h5ad --data DIR --celltypes CT.txt --outdir DIR --markers-json JSON [--slides a,b]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import scanpy as sc

import bench_common as bc
from hhca.log import get_logger


def parse_args():
    """Command-line arguments."""
    parser = bc.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]), slide=False)
    parser.add_argument("--outdir", required=True, help="output directory of the per-slide score tables")
    parser.add_argument("--markers-json", required=True, help="output json with the marker genes per cell type")
    parser.add_argument("--slides", default=",".join(bc.SLIDES), help="comma-separated slide names")
    return parser.parse_args()


def compute_markers(markers_json, n_top=15):
    """Top `n_top` genes per cell type by log fold change against the other types; also written to json."""
    ref = bc.load_ref()
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)
    cell_types = sorted(ref.obs[bc.LABEL].astype(str).unique())
    X = ref.X
    means = {}
    for cell_type in cell_types:
        means[cell_type] = np.asarray(X[(ref.obs[bc.LABEL].values == cell_type)].mean(0)).ravel()
    type_means = pd.DataFrame(means, index=ref.var_names)  # genes x types
    markers = {}
    for cell_type in cell_types:
        rest = (type_means.sum(1) - type_means[cell_type]) / (type_means.shape[1] - 1)
        fold_change = type_means[cell_type] - rest
        markers[cell_type] = list(fold_change.sort_values(ascending=False).head(n_top).index)
    with open(markers_json, "w") as handle:
        json.dump(markers, handle, indent=2)
    return markers


def main():
    """Pick the markers on the reference and score them on every slide."""
    args = parse_args()
    bc.configure(args)
    slides = [slide for slide in args.slides.split(",") if slide]
    os.makedirs(args.outdir, exist_ok=True)
    log = get_logger()

    markers = compute_markers(args.markers_json)
    for slide in slides:
        spots = bc.load_slide(slide)
        sc.pp.normalize_total(spots, target_sum=1e4)
        sc.pp.log1p(spots)
        scores = {}
        present = set(spots.var_names)
        for cell_type, marker_genes in markers.items():
            genes = [gene for gene in marker_genes if gene in present]
            if len(genes) >= 3:
                sc.tl.score_genes(spots, genes, score_name="_s", ctrl_size=50)
                scores[cell_type] = spots.obs["_s"].values.copy()
            else:
                scores[cell_type] = np.full(spots.n_obs, np.nan)
        score_df = pd.DataFrame(scores, index=spots.obs_names)
        score_df.index.name = "barcode"
        score_df.to_csv(f"{args.outdir}/{slide}.csv.gz", compression="gzip")
        log(f"{slide}: marker scores {score_df.shape}")
    log("done")


if __name__ == "__main__":
    main()
