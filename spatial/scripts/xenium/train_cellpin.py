#!/usr/bin/env python
"""Train CellPin on the HHCA reference and impute the Xenium data.

The model is smaller than the CellPin paper defaults (n_hidden 1024 -> 256, encoder_layers 16 -> 3, latent
192 -> 48) and trains on CPU.
  --holdout N    hold N panel genes out of the spatial input (they stay in the reference) and report how well
                 CellPin predicts them
  --skip-impute  train and save the model only; the labels come from the panel-encoder embedding
                 (transfer_labels.py), not from the imputed matrix. Used for the 162,638-cell rejection cohort,
                 where the dense imputed matrix would take about 2 GB.

Inputs:
    --ref     reference_subset.h5ad (build_cellpin_reference.py)
    --xenium  xenium_genes.h5ad (build_healthy_h5ad.py / build_rejection_h5ad.py)

Outputs (in --outdir):
    cellpin_model_<tag>.pt
    cellpin_imputed_<tag>.h5ad (unless --skip-impute)
    holdout_eval_<tag>.json, holdout_arrays_<tag>.npz (with --holdout)

Usage: python train_cellpin.py --ref reference_subset.h5ad --xenium xenium_genes.h5ad --outdir DIR
           [--tag full] [--accelerator cpu] [--skip-impute] [--holdout N]
"""

import argparse
import json
import os

import anndata as ad
import cellpin
import cellpin.pp
import numpy as np
import scanpy as sc
import torch
from cellpin.models import CellPin
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

SMALL_CONFIG = {
    "n_latent": 48,
    "n_hidden": 256,
    "encoder_layers": 3,
    "decoder_layers": 2,
    "encoder_dropout": 0.1,
    "drop_path_rate": 0.0,
    "reconstruction_loss": "nb",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", required=True, help="reference_subset.h5ad (build_cellpin_reference.py)")
    parser.add_argument("--xenium", required=True, help="xenium_genes.h5ad")
    parser.add_argument("--outdir", required=True, help="output directory")
    parser.add_argument("--tag", default="full", help="suffix of the output file names")
    parser.add_argument("--holdout", type=int, default=0, help="number of panel genes held out for evaluation")
    parser.add_argument("--pretrain_epochs", type=int, default=12)
    parser.add_argument("--train_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--accelerator", default="auto", help="lightning accelerator (cpu, gpu, auto)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-impute", action="store_true", help="train and save the model only")
    return parser.parse_args()


def to_dense(matrix):
    """A sparse or dense matrix as a dense numpy array."""
    return np.asarray(matrix.todense()) if hasattr(matrix, "todense") else np.asarray(matrix)


def hold_out_genes(sc_adata, st_adata, n_genes, seed):
    """Draw `n_genes` panel genes; return them, their measured counts and the spatial data without them."""
    panel = [g for g in st_adata.var_names if g in set(sc_adata.var_names)]
    rng = np.random.default_rng(seed)
    genes = sorted(rng.choice(panel, size=n_genes, replace=False).tolist())
    # keep the measured counts of the held-out genes as the target
    measured = to_dense(st_adata[:, genes].layers["counts"])
    keep = [g for g in st_adata.var_names if g not in set(genes)]
    return genes, measured, st_adata[:, keep].copy()


def evaluate_holdout(imputed, measured, genes):
    """Per-gene Pearson / Spearman correlation of log1p measured vs imputed counts, with their summary."""
    per_gene = {}
    pearson, spearman = [], []
    for j, gene in enumerate(genes):
        log_measured = np.log1p(measured[:, j].astype(float))
        log_imputed = np.log1p(imputed[:, j].astype(float))
        if log_measured.std() < 1e-9 or log_imputed.std() < 1e-9:
            continue
        r_pearson = pearsonr(log_measured, log_imputed)[0]
        r_spearman = spearmanr(log_measured, log_imputed)[0]
        per_gene[gene] = {
            "pearson": float(r_pearson),
            "spearman": float(r_spearman),
            "measured_mean": float(measured[:, j].mean()),
        }
        pearson.append(r_pearson)
        spearman.append(r_spearman)
    return {
        "n_holdout": len(genes),
        "n_scored": len(pearson),
        "pearson_mean": float(np.mean(pearson)),
        "pearson_median": float(np.median(pearson)),
        "spearman_mean": float(np.mean(spearman)),
        "spearman_median": float(np.median(spearman)),
        "holdout_genes": genes,
        "per_gene": per_gene,
    }


def main():
    args = parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(
        f"[run:{args.tag}] cellpin {getattr(cellpin, '__version__', '?')} torch {torch.__version__} "
        f"scanpy {sc.__version__} threads {torch.get_num_threads()}",
        flush=True,
    )

    print(f"[run:{args.tag}] loading reference + xenium", flush=True)
    sc_adata = ad.read_h5ad(args.ref)
    st_adata = ad.read_h5ad(args.xenium)
    print(f"[run:{args.tag}] ref {sc_adata.shape}  xenium {st_adata.shape}", flush=True)

    holdout_genes = []
    measured_holdout = None
    if args.holdout > 0:
        holdout_genes, measured_holdout, st_adata = hold_out_genes(sc_adata, st_adata, args.holdout, args.seed)
        print(f"[run:{args.tag}] held out {len(holdout_genes)} panel genes; spatial input {st_adata.shape}", flush=True)

    sc_ds, st_ds = cellpin.pp.setup_data(sc_adata=sc_adata, st_adata=st_adata, layer="counts")

    model = CellPin(sc_dataset=sc_ds, config=SMALL_CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[run:{args.tag}] model params: {n_params / 1e6:.2f}M", flush=True)

    model.fit(
        dataset=sc_ds,
        pretrain_epochs=args.pretrain_epochs,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        early_stopping_patience=5,
        save_checkpoints=False,
        output_dir=f"{outdir}/runs/{args.tag}",
        accelerator=args.accelerator,
        devices=1,
        precision="32-true",
        num_workers=0,
    )

    # saved before imputation, so --skip-impute runs keep the model
    model.save(f"{outdir}/cellpin_model_{args.tag}.pt")

    if args.skip_impute:
        print(f"[run:{args.tag}] --skip-impute: model saved, no imputed matrix written", flush=True)
        return

    print(f"[run:{args.tag}] imputing spatial", flush=True)
    loader = DataLoader(st_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    imputed = model.impute(
        loader, obs_adata=st_adata, mc_samples=12, return_norm=True, nb_count_samples=12, return_int=False
    )
    if "spatial" in st_adata.obsm:
        imputed.obsm["spatial"] = st_adata.obsm["spatial"]
    # the 'imputed' layer duplicates X
    if "imputed" in imputed.layers:
        del imputed.layers["imputed"]

    out_path = f"{outdir}/cellpin_imputed_{args.tag}.h5ad"
    imputed.write_h5ad(out_path, compression="gzip")
    print(f"[run:{args.tag}] wrote {out_path} shape={imputed.shape}", flush=True)

    if args.holdout > 0:
        imputed_holdout = to_dense(imputed[:, holdout_genes].X)
        summary = evaluate_holdout(imputed_holdout, measured_holdout, holdout_genes)
        with open(f"{outdir}/holdout_eval_{args.tag}.json", "w") as handle:
            json.dump(summary, handle, indent=2)
        np.savez(
            f"{outdir}/holdout_arrays_{args.tag}.npz",
            genes=np.array(holdout_genes),
            true=measured_holdout,
            imp=imputed_holdout,
        )
        print(
            f"[run:{args.tag}] HELD-OUT  Pearson mean={summary['pearson_mean']:.3f} "
            f"median={summary['pearson_median']:.3f} | Spearman mean={summary['spearman_mean']:.3f} "
            f"(scored {summary['n_scored']}/{len(holdout_genes)})",
            flush=True,
        )


if __name__ == "__main__":
    main()
