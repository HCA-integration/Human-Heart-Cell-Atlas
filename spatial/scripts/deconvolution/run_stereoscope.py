#!/usr/bin/env python
"""stereoscope, step 2: map one slide with the single-cell model of stereo_train_ref.py.

Inputs:  <models>/stereoscope/rna_model and rna_genes.txt, <data>/<slide>/slide.h5ad
Outputs: <out>/stereoscope/<slide>.csv.gz and .json

Usage: run_stereoscope.py --ref REF.h5ad --data DIR --out DIR --celltypes CT.txt --models DIR [--epochs 2000] SLIDE
"""

import argparse
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import bench_common as bc
from hhca.log import get_logger


def parse_args():
    """Command-line arguments."""
    parser = bc.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    parser.add_argument("--models", required=True, help="model directory holding stereoscope/rna_model")
    parser.add_argument("--epochs", type=int, default=2000, help="training epochs of the spatial model")
    return parser.parse_args()


def main():
    """Map one slide with the trained single-cell model and save the proportions."""
    args = parse_args()
    bc.configure(args)
    slide = args.slide
    model_dir = f"{args.models}/stereoscope"
    start = time.time()
    log = get_logger()

    import scvi

    scvi.settings.seed = bc.SEED
    from scvi.external import RNAStereoscope, SpatialStereoscope

    ref = bc.load_ref()
    genes = bc.read_lines(f"{model_dir}/rna_genes.txt")
    ref = ref[:, genes].copy()
    RNAStereoscope.setup_anndata(ref, labels_key=bc.LABEL)
    rna_model = RNAStereoscope.load(f"{model_dir}/rna_model", adata=ref)
    spots = bc.load_slide(slide)
    missing = [gene for gene in genes if gene not in set(spots.var_names)]
    if missing:
        log("model genes missing from the slide:", len(missing))
    spots = spots[:, [gene for gene in genes if gene in set(spots.var_names)]].copy()
    # genes in the order of the model; genes missing from the slide are zero columns
    if spots.n_vars != len(genes):
        full = sp.lil_matrix((spots.n_obs, len(genes)), dtype=np.float32)
        position = {gene: i for i, gene in enumerate(genes)}
        for j, gene in enumerate(spots.var_names):
            full[:, position[gene]] = spots.X[:, j]
        spots = ad.AnnData(X=full.tocsr(), obs=spots.obs.copy(), var=pd.DataFrame(index=genes), obsm=spots.obsm)
    log(f"{slide}: {spots.n_obs} spots x {spots.n_vars} genes")
    SpatialStereoscope.setup_anndata(spots)
    spatial_model = SpatialStereoscope.from_rna_model(spots, rna_model)
    spatial_model.train(max_epochs=args.epochs)
    prop = spatial_model.get_proportions()  # spots x cell types
    prop.index = spots.obs_names
    bc.save_proportions("stereoscope", slide, prop, time.time() - start, spots.n_vars)


if __name__ == "__main__":
    main()
