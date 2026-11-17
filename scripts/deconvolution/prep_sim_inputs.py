#!/usr/bin/env python
"""Subsample each simulated slide to at most --max-spots pseudo-spots and write the method inputs.

One random stream runs over the slides in the order given (default: sorted names), so the draw of a slide
depends on the slides before it.

Inputs:  <sim>/<slide>/slide.h5ad and truth_proportions.csv.gz (build_sim.py)
Outputs: <out>/<slide>/slide.h5ad, matrix.mtx.gz, genes.txt, barcodes.txt, coords.csv, truth_proportions.csv.gz

Usage: prep_sim_inputs.py --sim DIR --out DIR [--max-spots 5000] [--seed 0] [slide ...]
"""

import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import hhca
from hhca.io import write_mtx_gz


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sim", required=True, help="directory with the simulated slides (<sim>/<slide>/)")
    parser.add_argument("--out", required=True, help="output directory of the method inputs")
    parser.add_argument("--max-spots", type=int, default=5000, help="largest number of pseudo-spots kept per slide")
    parser.add_argument("--seed", type=int, default=0, help="seed of the subsampling")
    parser.add_argument("slides", nargs="*", help="slide names (default: every slide in --sim)")
    return parser.parse_args()


def main():
    """Subsample every simulated slide and write it in the layout of the method inputs."""
    args = parse_args()
    slides = args.slides or sorted(os.listdir(args.sim))
    rng = np.random.default_rng(args.seed)
    for slide in slides:
        slide_dir = f"{args.sim}/{slide}"
        if not os.path.exists(f"{slide_dir}/slide.h5ad"):
            print("skip", slide)
            continue
        adata = sc.read_h5ad(f"{slide_dir}/slide.h5ad")
        truth = pd.read_csv(f"{slide_dir}/truth_proportions.csv.gz", index_col=0)
        adata = hhca.tl.subsample_spots(adata, args.max_spots, rng)
        truth = truth.loc[adata.obs_names]
        out = f"{args.out}/{slide}"
        os.makedirs(out, exist_ok=True)
        adata.X = sp.csr_matrix(adata.X).astype(np.float32)
        adata.write(f"{out}/slide.h5ad")
        write_mtx_gz(adata.X.T, f"{out}/matrix.mtx.gz")
        with open(f"{out}/genes.txt", "w") as handle:
            handle.write("\n".join(map(str, adata.var_names)) + "\n")
        with open(f"{out}/barcodes.txt", "w") as handle:
            handle.write("\n".join(map(str, adata.obs_names)) + "\n")
        xy = adata.obsm["spatial"]
        coords = pd.DataFrame(
            {
                "barcode": adata.obs_names,
                "total_counts": np.asarray(adata.X.sum(1)).ravel(),
                "x": xy[:, 0],
                "y": xy[:, 1],
                "array_row": adata.obs["array_row"].values,
                "array_col": adata.obs["array_col"].values,
            }
        )
        coords.to_csv(f"{out}/coords.csv", index=False)
        truth.to_csv(f"{out}/truth_proportions.csv.gz", compression="gzip")
        print(f"{slide}: {adata.n_obs} spots x {adata.n_vars} genes -> {out}", flush=True)


if __name__ == "__main__":
    main()
