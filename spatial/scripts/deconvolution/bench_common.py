#!/usr/bin/env python
"""Shared helpers of the Python deconvolution runners: arguments, seeding, loading and saving.

add_args() adds the common command-line arguments (reference, slide inputs, output root, gene panel, cell-type
list, seed); configure() stores them in this module and seeds the random number generators. The same runners
serve the 11-slide benchmark and the ground-truth simulation; only the paths differ.

Usage: import bench_common as bc; args = bc.add_args(parser).parse_args(); bc.configure(args)
"""

import importlib.util
import json
import os
import random

import numpy as np
import scanpy as sc
import scipy.sparse as sp

REF_H5 = None
DATA = None
OUT = None
LABEL = "cell_type_L3"
PANEL = None
CELLTYPES_FILE = None
SLIDES = [
    "HHH1-LA",
    "HHH2-LV-endo-1",
    "HHH4-RA2",
    "HHH4-RV2",
    "A17-AH1-C1",
    "A18-AH2-D1",
    "HHH2-D1",
    "HHH2-LV-endo-2",
    "HHH4-A1",
    "HHH5-B1",
    "HHH5-C1",
]
SEED = 0


def add_args(parser, slide=True):
    """Add the arguments shared by the runners to `parser` (with the positional slide name unless slide=False)."""
    parser.add_argument("--ref", required=True, help="reference h5ad (raw counts in X, obs[cell_type_L3])")
    parser.add_argument("--data", required=True, help="directory with <slide>/slide.h5ad (prep_inputs.py output)")
    parser.add_argument("--out", default="", help="output root; writes <out>/<method>/<slide>.csv.gz")
    parser.add_argument("--panel", default="", help="shared HVG gene panel (make_gene_panel.py)")
    parser.add_argument("--celltypes", required=True, help="cell-type list (one per line)")
    parser.add_argument("--label", default="cell_type_L3")
    parser.add_argument("--seed", type=int, default=0)
    if slide:
        parser.add_argument("slide")
    return parser


def configure(args):
    """Store the parsed common arguments in this module and seed the random number generators."""
    global REF_H5, DATA, OUT, PANEL, CELLTYPES_FILE, LABEL, SEED
    REF_H5, DATA, OUT = args.ref, args.data, args.out
    PANEL, CELLTYPES_FILE, LABEL, SEED = args.panel, args.celltypes, args.label, args.seed
    set_seed(SEED)


def set_seed(seed):
    """Seed python and numpy, and torch / pyro in the environments that have them."""
    random.seed(seed)
    np.random.seed(seed)
    if importlib.util.find_spec("torch"):
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if importlib.util.find_spec("pyro"):
        import pyro

        pyro.set_rng_seed(seed)


def read_counts(path):
    """An h5ad with X as float32 CSR and unique gene names."""
    adata = sc.read_h5ad(path)
    adata.X = sp.csr_matrix(adata.X).astype(np.float32)
    adata.var_names_make_unique()
    return adata


def load_ref():
    """The single-cell reference."""
    return read_counts(REF_H5)


def load_slide(slide):
    """The spots of one slide."""
    return read_counts(f"{DATA}/{slide}/slide.h5ad")


def read_lines(path):
    """Non-empty lines of a text file, stripped."""
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def read_panel():
    """Gene panel as a list, or None when no panel file was given."""
    if PANEL and os.path.exists(PANEL):
        return read_lines(PANEL)
    return None


def celltypes():
    """Cell types in the order of the cell-type list."""
    return read_lines(CELLTYPES_FILE)


def save_proportions(method, slide, prop_df, runtime_sec, n_genes_used, extra=None):
    """Write <out>/<method>/<slide>.csv.gz (spots x cell types, rows sum to 1) and a small run summary json."""
    method_dir = f"{OUT}/{method}"
    os.makedirs(method_dir, exist_ok=True)
    # every cell type as a column (0 where a method returned none), in the order of the cell-type list
    cell_types = celltypes()
    for cell_type in cell_types:
        if cell_type not in prop_df.columns:
            prop_df[cell_type] = 0.0
    prop_df = prop_df[cell_types]
    row_sums = prop_df.sum(1).replace(0, 1.0)
    prop_df = prop_df.div(row_sums, axis=0)
    prop_df.index.name = "barcode"
    prop_df.to_csv(f"{method_dir}/{slide}.csv.gz", compression="gzip")
    meta = {
        "method": method,
        "slide": slide,
        "runtime_sec": float(runtime_sec),
        "n_genes_used": int(n_genes_used),
        "n_spots": int(prop_df.shape[0]),
        "n_celltypes": int(prop_df.shape[1]),
        "seed": int(SEED),
    }
    if extra:
        meta.update(extra)
    with open(f"{method_dir}/{slide}.json", "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"[save] {method}/{slide}: {prop_df.shape} runtime={runtime_sec:.1f}s genes={n_genes_used}", flush=True)
    return prop_df
