#!/usr/bin/env python
"""cell2location, step 1: negative-binomial regression on the reference to get the cell-type signatures.

Trained once on the panel genes; every slide is then mapped with these signatures (run_cell2location.py).

Inputs:  the shared reference h5ad and gene panel
Outputs: <models>/cell2location/ref_signatures.csv (genes x cell types) and regression_model/

Usage: c2l_train_ref.py --ref REF.h5ad --data DIR --panel PANEL.txt --celltypes CT.txt --models DIR [--epochs 250]
"""

import argparse
import os

import bench_common as bc
from hhca.log import get_logger


def parse_args():
    """Command-line arguments."""
    parser = bc.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]), slide=False)
    parser.add_argument("--models", required=True, help="model directory (signatures go to <models>/cell2location/)")
    parser.add_argument("--epochs", type=int, default=250, help="training epochs of the regression model")
    return parser.parse_args()


def main():
    """Train the regression model on the reference and write the signatures."""
    args = parse_args()
    bc.configure(args)
    model_dir = f"{args.models}/cell2location"
    os.makedirs(model_dir, exist_ok=True)
    log = get_logger()

    import cell2location
    import scvi

    scvi.settings.seed = bc.SEED
    import cell2location.models

    ref = bc.load_ref()
    panel = bc.read_panel()
    if panel:
        keep = [gene for gene in panel if gene in set(ref.var_names)]
        ref = ref[:, keep].copy()
    log("ref for regression:", ref.shape)
    cell2location.models.RegressionModel.setup_anndata(adata=ref, labels_key=bc.LABEL)
    model = cell2location.models.RegressionModel(ref)
    model.train(max_epochs=args.epochs)
    ref = model.export_posterior(ref, sample_kwargs={"num_samples": 1000, "batch_size": 2500})
    # expression signature of each cell type (genes x cell types)
    if "means_per_cluster_mu_fg" in ref.varm:
        signatures = ref.varm["means_per_cluster_mu_fg"].copy()
    else:
        keys = [key for key in ref.varm.keys() if "per_cluster_mu_fg" in key]
        signatures = ref.varm[keys[0]].copy()
    # columns named by cell type
    factor_names = list(ref.uns["mod"]["factor_names"])
    columns = [column for column in signatures.columns if column.startswith("means_per_cluster_mu_fg")]
    signatures = signatures[columns]
    signatures.columns = factor_names
    signatures.to_csv(f"{model_dir}/ref_signatures.csv")
    model.save(f"{model_dir}/regression_model", overwrite=True)
    log("wrote signatures", signatures.shape, "->", f"{model_dir}/ref_signatures.csv")


if __name__ == "__main__":
    main()
