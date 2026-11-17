#!/usr/bin/env python
"""Mean expression profile per atlas cell type: the gene x factor model of the FICTURE decode.

The atlas X (log1p-normalised counts) is streamed in row chunks, each cell is rescaled to CP10k over all its
genes, and the cells are averaged per type at Level 4 and Level 3 in one pass (Unassigned dropped).

Inputs: the HHCA atlas h5ad and the gene set (feature.clean.tsv.gz of build_ficture_input.py --crop).
Outputs: atlas_{L4,L3}_pseudobulk.tsv (genes x types, restricted to the gene set) and
atlas_{L4,L3}_factor_order.json in --outdir; nuclei per type (atlas_{L4,L3}_ncells.csv) and top genes per type
(atlas_{L4,L3}_topgenes.csv) in --tabdir.

Usage: python build_atlas_profiles.py --atlas ATLAS.h5ad --features feature.clean.tsv.gz --outdir DIR --tabdir DIR
"""

import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from hhca.io import decode, read_categorical
from hhca.log import get_logger

LEVELS = {"L4": "Final Level 4_harmonized", "L3": "Final Level 3_harmonized"}
DROPPED_LABELS = {"Unassigned", "<NA>", "nan", "NA"}
CHUNK_ROWS = 20000

log = get_logger("atlas")


def read_levels(atlas):
    """Per level: the kept type names, and for every cell the index of its type (-1 = dropped)."""
    type_names = {}
    type_of_cell = {}
    for level, column in LEVELS.items():
        categories, codes = read_categorical(atlas["obs"], column)
        categories = categories.tolist()
        log(f"{level} ({column}): {len(categories)} categories")
        kept = [i for i, label in enumerate(categories) if label not in DROPPED_LABELS]
        type_names[level] = [categories[i] for i in kept]
        type_of_category = -np.ones(len(categories), dtype=np.int64)
        for type_index, category_index in enumerate(kept):
            type_of_category[category_index] = type_index
        type_of_cell[level] = type_of_category[codes.astype(np.int64)]
    return type_names, type_of_cell


def sum_profiles(atlas, n_obs, n_vars, type_names, type_of_cell):
    """Per level: CP10k sums (types x genes) and nuclei per type, streaming X in row chunks."""
    sums = {level: np.zeros((len(type_names[level]), n_vars), dtype=np.float64) for level in LEVELS}
    n_cells = {level: np.zeros(len(type_names[level]), dtype=np.int64) for level in LEVELS}

    data = atlas["X"]["data"]
    indices = atlas["X"]["indices"]
    indptr = atlas["X"]["indptr"][:]
    n_chunks = (n_obs + CHUNK_ROWS - 1) // CHUNK_ROWS
    for chunk in range(n_chunks):
        row0 = chunk * CHUNK_ROWS
        row1 = min(n_obs, row0 + CHUNK_ROWS)
        p0, p1 = int(indptr[row0]), int(indptr[row1])
        X = sp.csr_matrix(
            (data[p0:p1].astype(np.float64), indices[p0:p1], indptr[row0 : row1 + 1] - p0),
            shape=(row1 - row0, n_vars),
        )
        # CP10k per cell over all genes
        library = np.asarray(X.sum(1)).ravel()
        library[library == 0] = 1.0
        X = sp.diags(1e4 / library) @ X
        for level in LEVELS:
            cell_type = type_of_cell[level][row0:row1]
            kept = cell_type >= 0
            if not kept.any():
                continue
            indicator = sp.csr_matrix(
                (np.ones(kept.sum()), (cell_type[kept], np.nonzero(kept)[0])),
                shape=(len(type_names[level]), row1 - row0),
            )  # types x cells
            sums[level] += (indicator @ X).toarray()
            np.add.at(n_cells[level], cell_type[kept], 1)
        if chunk % 10 == 0 or chunk == n_chunks - 1:
            log(f"chunk {chunk + 1}/{n_chunks} rows[{row0}:{row1}] nnz={p1 - p0}")
    return sums, n_cells


def write_level(level, symbols, types, sums, n_cells, gene_set, outdir, tabdir):
    """Write the model table, the factor order and the two summary tables of one level."""
    mean = sums / np.maximum(n_cells[:, None], 1)  # mean CP10k, types x genes
    profiles = pd.DataFrame(mean.T, index=symbols, columns=types)
    profiles = profiles.groupby(level=0).sum()  # duplicated symbols summed
    in_gene_set = profiles.index.intersection(gene_set)
    profiles = profiles.loc[in_gene_set]
    out = f"{outdir}/atlas_{level}_pseudobulk.tsv"
    profiles.index.name = "gene"
    profiles.to_csv(out, sep="\t", float_format="%.6g")
    nuclei = pd.DataFrame({"factor": types, "n_cells": n_cells})
    nuclei["thin_lt500"] = nuclei["n_cells"] < 500
    nuclei.to_csv(f"{tabdir}/atlas_{level}_ncells.csv", index=False)
    log(
        f"{level}: wrote {out} shape={profiles.shape}; coverage={len(in_gene_set)}/{len(gene_set)} "
        f"({len(in_gene_set) / len(gene_set):.4f}); thin(<500)={int(nuclei.thin_lt500.sum())}/{len(types)}"
    )
    top_genes = {
        name: ", ".join(profiles[name].sort_values(ascending=False).head(12).index) for name in profiles.columns
    }
    pd.DataFrame(
        {
            "factor": list(top_genes),
            "n_cells": [int(n_cells[types.index(name)]) for name in top_genes],
            "top12": list(top_genes.values()),
        }
    ).to_csv(f"{tabdir}/atlas_{level}_topgenes.csv", index=False)
    with open(f"{outdir}/atlas_{level}_factor_order.json", "w") as handle:
        json.dump(types, handle, indent=2)


def parse_args():
    """Command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--atlas", required=True, help="HHCA atlas h5ad; X is read as stored (log1p normcounts)")
    parser.add_argument("--features", required=True, help="feature.clean.tsv.gz of the gene-set crop")
    parser.add_argument("--outdir", required=True, help="folder for the model tables and factor-order files")
    parser.add_argument("--tabdir", required=True, help="folder for the nuclei-per-type and top-gene tables")
    return parser.parse_args()


def main():
    """Average the atlas per type at both levels and write the model tables."""
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.tabdir, exist_ok=True)

    with h5py.File(args.atlas, "r") as atlas:
        n_obs, n_vars = [int(x) for x in atlas["X"].attrs["shape"]]
        symbols = np.array([decode(symbol) for symbol in atlas["var"]["_index"][:]])
        log(f"atlas {n_obs} cells x {n_vars} genes")
        type_names, type_of_cell = read_levels(atlas)
        sums, n_cells = sum_profiles(atlas, n_obs, n_vars, type_names, type_of_cell)

    gene_set = set(pd.read_csv(args.features, sep="\t")["gene"].astype(str))
    log(f"gene set: {len(gene_set)} symbols")

    for level in LEVELS:
        write_level(level, symbols, type_names[level], sums[level], n_cells[level], gene_set, args.outdir, args.tabdir)
    log("done")


if __name__ == "__main__":
    main()
