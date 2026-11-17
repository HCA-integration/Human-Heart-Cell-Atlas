#!/usr/bin/env python
"""Spotiphy deconvolution of one slide: marker selection on the shared reference, then proportion estimation.

The seeds (python, numpy, torch, pyro) are set at start-up and again right before the estimation.

Inputs:  the shared reference h5ad, <data>/<slide>/slide.h5ad
Outputs: <out>/spotiphy/<slide>.csv.gz and .json

Usage: run_spotiphy.py --ref REF.h5ad --data DIR --out DIR --celltypes CT.txt [--epochs 8000] SLIDE
"""

import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import bench_common as bc
from hhca.log import get_logger


def parse_args():
    """Command-line arguments."""
    parser = bc.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    parser.add_argument("--epochs", type=int, default=8000, help="training epochs of the proportion model")
    return parser.parse_args()


def main():
    """Select marker genes on the reference, estimate the proportions of one slide and save them."""
    args = parse_args()
    bc.configure(args)
    slide = args.slide
    start = time.time()
    log = get_logger()

    import spotiphy
    import torch

    key = bc.LABEL
    adata_sc = bc.load_ref()
    adata_st = bc.load_slide(slide)
    adata_sc.obs[key] = adata_sc.obs[key].astype(str)
    adata_st.var_names_make_unique()
    log("init: sc", adata_sc.shape, "st", adata_st.shape)
    adata_sc, adata_st = spotiphy.initialization(adata_sc, adata_st, verbose=1)
    type_list = sorted(adata_sc.obs[key].unique().astype(str))
    markers_by_type = spotiphy.sc_reference.marker_selection(
        adata_sc, key_type=key, return_dict=True, n_select=50, threshold_p=0.1, threshold_fold=1.5, q=0.15
    )
    marker_genes = []
    for cell_type in type_list:
        marker_genes.extend(markers_by_type.get(cell_type, []))
    marker_genes = list(dict.fromkeys(marker_genes))  # unique, in order of first appearance
    log("n marker genes:", len(marker_genes))
    adata_sc_markers = adata_sc[:, marker_genes]
    adata_st_markers = adata_st[:, marker_genes]
    sc_ref = spotiphy.construct_sc_ref(adata_sc_markers, key_type=key)
    X = adata_st_markers.X
    X = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("device:", device, "estimating proportion...")
    bc.set_seed(bc.SEED)
    proportions = spotiphy.deconvolution.estimation_proportion(
        X, adata_sc_markers, sc_ref, type_list, key, n_epoch=args.epochs, plot=False, batch_prior=1, device=device
    )
    prop = pd.DataFrame(np.asarray(proportions), index=adata_st.obs_names, columns=type_list)
    bc.save_proportions(
        "spotiphy",
        slide,
        prop,
        time.time() - start,
        len(marker_genes),
        extra={"n_epoch": args.epochs, "device": device},
    )


if __name__ == "__main__":
    main()
