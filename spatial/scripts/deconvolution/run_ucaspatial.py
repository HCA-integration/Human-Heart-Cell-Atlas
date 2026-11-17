#!/usr/bin/env python
"""UCASpatial deconvolution of one slide (entropy-weighted NMF and weighted NNLS, on CPU).

Uses the UCASpatial authors' Python implementation (ucaspatial.py, next to this script) with
meta_filter=False, as in the example of their README.

Inputs:  the shared reference h5ad, <data>/<slide>/slide.h5ad
Outputs: <out>/ucaspatial/<slide>.csv.gz and .json (working files in <out>/ucaspatial/_work_<slide>/)

Usage: run_ucaspatial.py --ref REF.h5ad --data DIR --out DIR --celltypes CT.txt [--random-seed 12345] SLIDE
"""

import argparse
import os
import time

import bench_common as bc
import ucaspatial
from hhca.log import get_logger


def parse_args():
    """Command-line arguments."""
    parser = bc.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    parser.add_argument("--random-seed", type=int, default=12345, help="random_seed passed to UCASpatial")
    return parser.parse_args()


def main():
    """Run UCASpatial on one slide and save the proportions."""
    args = parse_args()
    bc.configure(args)
    slide = args.slide
    start = time.time()
    log = get_logger()

    sc_ref = bc.load_ref()
    spots = bc.load_slide(slide)
    sc_ref.obs[bc.LABEL] = sc_ref.obs[bc.LABEL].astype(str)
    # UCASpatial reads counts from layers['data'] and normalises them itself
    sc_ref.layers["data"] = sc_ref.X.copy()
    work_dir = f"{bc.OUT}/ucaspatial/_work_{slide}"
    os.makedirs(work_dir, exist_ok=True)
    log(f"{slide}: sc_ref {sc_ref.shape} st {spots.shape}")
    model = ucaspatial.UCASpatial(
        sc_ref=sc_ref,
        st_vis=spots,
        clust_vr=bc.LABEL,
        marker_slot="data",
        meta_filter=False,
        downsample_n=1,
        n_cluster=200,
        random_seed=args.random_seed,
        output_path=work_dir,
    )
    result = model.run()
    prop = result["proportions"].copy()
    prop = prop.drop(columns=[column for column in ["residual_ss"] if column in prop.columns])
    prop.index = spots.obs_names[: prop.shape[0]] if prop.shape[0] == spots.n_obs else prop.index
    n_genes = int(getattr(model, "cos_n_genes_user", spots.n_vars) or spots.n_vars)
    bc.save_proportions("ucaspatial", slide, prop, time.time() - start, n_genes)


if __name__ == "__main__":
    main()
