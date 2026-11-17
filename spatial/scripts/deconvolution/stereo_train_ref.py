#!/usr/bin/env python
"""stereoscope, step 1: train the single-cell model (RNAStereoscope) on the reference, panel genes only.

Trained once; every slide is then mapped with this model (run_stereoscope.py).

Inputs:  the shared reference h5ad and gene panel
Outputs: <models>/stereoscope/rna_model/ and rna_genes.txt

Usage: stereo_train_ref.py --ref REF.h5ad --data DIR --panel PANEL.txt --celltypes CT.txt --models DIR [--epochs 100]
"""

import argparse
import os

import bench_common as bc
from hhca.log import get_logger


def parse_args():
    """Command-line arguments."""
    parser = bc.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]), slide=False)
    parser.add_argument("--models", required=True, help="model directory (the model goes to <models>/stereoscope/)")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs of the single-cell model")
    return parser.parse_args()


def main():
    """Train the single-cell model on the panel genes and save it with its gene list."""
    args = parse_args()
    bc.configure(args)
    model_dir = f"{args.models}/stereoscope"
    os.makedirs(model_dir, exist_ok=True)
    log = get_logger()

    import scvi

    scvi.settings.seed = bc.SEED
    from scvi.external import RNAStereoscope

    ref = bc.load_ref()
    panel = bc.read_panel()
    keep = [gene for gene in panel if gene in set(ref.var_names)]
    ref = ref[:, keep].copy()
    log("ref for stereoscope:", ref.shape)
    RNAStereoscope.setup_anndata(ref, labels_key=bc.LABEL)
    rna_model = RNAStereoscope(ref)
    rna_model.train(max_epochs=args.epochs)
    rna_model.save(f"{model_dir}/rna_model", overwrite=True)
    with open(f"{model_dir}/rna_genes.txt", "w") as handle:
        handle.write("\n".join(keep) + "\n")
    log("wrote rna_model and rna_genes.txt", len(keep), "genes")


if __name__ == "__main__":
    main()
