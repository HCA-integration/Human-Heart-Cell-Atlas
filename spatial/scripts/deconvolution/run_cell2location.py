#!/usr/bin/env python
"""cell2location, step 2: map one slide with the reference signatures of c2l_train_ref.py.

Proportions = 5 % quantile of the posterior cell abundance, normalised to sum to 1 per spot.

Inputs:  <models>/cell2location/ref_signatures.csv, <data>/<slide>/slide.h5ad
Outputs: <out>/cell2location/<slide>.csv.gz and .json

Usage: run_cell2location.py --ref REF.h5ad --data DIR --out DIR --celltypes CT.txt --models DIR
           [--epochs 15000] [--ncells 8] SLIDE
"""

import argparse
import time

import numpy as np
import pandas as pd

import bench_common as bc
from hhca.log import get_logger


def parse_args():
    """Command-line arguments."""
    parser = bc.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    parser.add_argument("--models", required=True, help="model directory holding cell2location/ref_signatures.csv")
    parser.add_argument("--epochs", type=int, default=15000, help="training epochs of the mapping model")
    parser.add_argument("--ncells", type=int, default=8, help="expected cells per spot (N_cells_per_location)")
    return parser.parse_args()


def main():
    """Map one slide and save the normalised 5 % quantile of the cell abundance."""
    args = parse_args()
    bc.configure(args)
    slide = args.slide
    model_dir = f"{args.models}/cell2location"
    start = time.time()
    log = get_logger()

    import cell2location
    import scvi

    scvi.settings.seed = bc.SEED
    import cell2location.models

    signatures = pd.read_csv(f"{model_dir}/ref_signatures.csv", index_col=0)
    spots = bc.load_slide(slide)
    shared = [gene for gene in signatures.index if gene in set(spots.var_names)]
    spots = spots[:, shared].copy()
    signatures = signatures.loc[shared]
    log(f"{slide}: {spots.n_obs} spots x {len(shared)} shared genes")
    cell2location.models.Cell2location.setup_anndata(adata=spots)
    model = cell2location.models.Cell2location(
        spots, cell_state_df=signatures, N_cells_per_location=args.ncells, detection_alpha=20
    )
    model.train(max_epochs=args.epochs, batch_size=None, train_size=1)
    spots = model.export_posterior(spots, sample_kwargs={"num_samples": 1000, "batch_size": spots.n_obs})
    # 5 % quantile of the cell abundance
    quantile_keys = [key for key in spots.obsm.keys() if "q05_cell_abundance" in key]
    abundance = spots.obsm[quantile_keys[0]] if quantile_keys else spots.obsm["means_cell_abundance_w_sf"]
    abundance = pd.DataFrame(np.asarray(abundance), index=spots.obs_names)
    # obsm columns ('q05cell_abundance_w_sf_<type>') follow the order of the signature columns
    abundance.columns = list(signatures.columns)[: abundance.shape[1]]
    prop = abundance.div(abundance.sum(1).replace(0, 1.0), axis=0)
    bc.save_proportions(
        "cell2location",
        slide,
        prop,
        time.time() - start,
        len(shared),
        extra={"epochs": args.epochs, "N_cells_per_location": args.ncells},
    )


if __name__ == "__main__":
    main()
