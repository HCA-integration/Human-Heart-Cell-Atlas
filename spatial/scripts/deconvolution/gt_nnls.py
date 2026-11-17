#!/usr/bin/env python
"""NNLS deconvolution of one simulated slide.

Signatures = mean CP10k per cell type in the shared reference (hhca.tl.reference_signatures); genes = the 2,000
with the largest coefficient of variation across types plus cardiac markers; one fit per spot
(hhca.tl.nnls_deconvolution). The reported proportion is the mRNA fraction (weights x signature totals, normalised
per spot).

Inputs:  the shared reference h5ad, <data>/<slide>/slide.h5ad
Outputs: <out>/NNLS/<slide>.csv.gz and .json

Usage: gt_nnls.py --ref REF.h5ad --data DIR --out DIR --celltypes CT.txt [--config CONFIG] SLIDE
"""

import argparse
import os
import time

from hhca import thread_env

# BLAS threads as set by the job, otherwise single-threaded; read when numpy is imported
os.environ.update(thread_env(1, numba=False, keep_existing=True))

import numpy as np
import pandas as pd

import bench_common as bc
import deconv_config
import hhca


def parse_args():
    """Command-line arguments."""
    parser = bc.add_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    return deconv_config.add_argument(parser).parse_args()


def main():
    """Fit every spot of one slide against the reference signatures and save the mRNA fractions."""
    args = parse_args()
    bc.configure(args)
    cfg = deconv_config.load(args.config)
    start = time.time()
    signatures = hhca.tl.reference_signatures(bc.load_ref(), bc.LABEL)
    spots = bc.load_slide(args.slide)
    hhca.tl.nnls_deconvolution(spots, signatures, cfg["methods"]["NNLS"]["n_top_genes"], cfg["markers"]["cardiac"])
    # a row-major table, so that the row sums of bc.save_proportions add the cell types in their order
    proportions = pd.DataFrame(
        np.ascontiguousarray(spots.obsm["NNLS"].to_numpy()), index=spots.obs_names, columns=spots.obsm["NNLS"].columns
    )
    bc.save_proportions("NNLS", args.slide, proportions, time.time() - start, len(spots.uns["NNLS"]["genes"]))


if __name__ == "__main__":
    main()
