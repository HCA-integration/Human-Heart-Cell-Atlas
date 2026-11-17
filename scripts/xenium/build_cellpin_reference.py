#!/usr/bin/env python
"""Build the CellPin reference from the HHCA atlas (1.46 M nuclei) with the harmonised Level-3 labels.

The atlas h5ad is read with h5py and streamed in row chunks, so the full matrix is never held in memory:
  * labels: obs `Final Level 3_harmonized` (without 'Unassigned'), Level 4 carried along;
  * genes: Xenium panel ∪ the atlas batch-aware HVGs (3000) ∪ curated markers;
  * cells: class-balanced subsample, at most --cap nuclei per Level-3 type, drawn from the whole atlas;
  * raw counts (layers/counts) of the subsample rows x reference genes are assembled into a small CSR.

Inputs:
    --atlas   HHCA atlas h5ad (layers/counts, obs labels, var HVG flag)
    --xenium  xenium_genes.h5ad of the cohort (build_healthy_h5ad.py / build_rejection_h5ad.py); its var_names
              define the panel

Outputs:
    --out        reference h5ad with obs['cell_type'] (Level 3), obs['cell_type_L4'], obs['atlas_row'],
                 var['gene_id_orig'], layers['counts'] and the atlas barcodes as obs_names
    --extra-out  with --extra-l3: every atlas cell of the listed Level-3 classes on the same genes (reference of the
                 Level-4 lymphoid kNN, lymphoid_level4.py); obs['in_subsample'] marks the cells that are also in
                 the subsample

Usage: python build_cellpin_reference.py --atlas ATLAS.h5ad --xenium xenium_genes.h5ad --out reference_subset.h5ad
           [--extra-l3 "T cells,NK cells" --extra-out reference_extra_lymphoid.h5ad] [--dry]
"""

import argparse
import collections
import os
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from hhca.io import decode, read_categorical

HVG_COL = "highly_variable-batch_key=donor_id-n_top_genes=3000"
DROPPED_LABELS = {"Unassigned", "nan"}
MARKERS = [
    "TNNT2",
    "MYH6",
    "MYH7",
    "NPPA",
    "NPPB",
    "ACTC1",
    "TNNI3",
    "MYL2",
    "MYL7",
    "RYR2",
    "TTN",
    "PECAM1",
    "VWF",
    "CDH5",
    "EGFL7",
    "FLT1",
    "KDR",
    "DCN",
    "PDGFRA",
    "GSN",
    "LUM",
    "COL1A1",
    "COL1A2",
    "FBLN1",
    "MYH11",
    "ACTA2",
    "TAGLN",
    "RGS5",
    "PDGFRB",
    "NOTCH3",
    "PTPRC",
    "CD68",
    "LYZ",
    "CD163",
    "MRC1",
    "CD3E",
    "NKG7",
    "NRXN1",
    "PLP1",
    "GPAM",
    "PROX1",
    "LYVE1",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--atlas", required=True, help="HHCA atlas h5ad (layers/counts, obs labels)")
    parser.add_argument("--xenium", required=True, help="query h5ad; its var_names define the panel")
    parser.add_argument("--out", required=True, help="output reference_subset.h5ad")
    parser.add_argument("--label", default="Final Level 3_harmonized", help="atlas obs column with the Level-3 labels")
    parser.add_argument(
        "--label-l4", default="Final Level 4_harmonized", help="atlas obs column with the Level-4 labels"
    )
    parser.add_argument("--cap", type=int, default=2000, help="largest number of nuclei per Level-3 type")
    parser.add_argument("--seed", type=int, default=0, help="seed of the subsample")
    parser.add_argument("--chunk", type=int, default=100_000, help="atlas rows read per chunk")
    parser.add_argument("--extra-l3", default="", help="comma-separated L3 classes written in full to --extra-out")
    parser.add_argument("--extra-out", default=None, help="output h5ad for the --extra-l3 cells")
    parser.add_argument("--dry", action="store_true", help="print the gene and subsample plan and exit (same as DRY=1)")
    return parser.parse_args()


def select_genes(atlas, xenium_path, n_var):
    """Reference genes: panel ∩ atlas ∪ atlas HVGs ∪ markers; returns their names, atlas columns and gene ids."""
    gene_symbols = np.array([decode(x) for x in atlas["var"]["_index"][:]])
    if "gene_ids" in atlas["var"]:
        gene_ids = np.array([decode(x) for x in atlas["var"]["gene_ids"][:]])
    else:
        gene_ids = gene_symbols
    hvg_mask = atlas["var"][HVG_COL][:].astype(bool) if HVG_COL in atlas["var"] else np.zeros(n_var, bool)

    # a symbol that occurs more than once in the atlas is taken from its first column
    first_column = {}
    for i, gene in enumerate(gene_symbols):
        if gene not in first_column:
            first_column[gene] = i

    panel = [str(gene) for gene in ad.read_h5ad(xenium_path).var_names]
    panel_in_atlas = [gene for gene in panel if gene in first_column]
    hvg_names = [gene_symbols[i] for i in np.where(hvg_mask)[0]]
    markers_in_atlas = [gene for gene in MARKERS if gene in first_column]
    names = sorted(set(panel_in_atlas) | set(hvg_names) | set(markers_in_atlas))
    columns = np.array([first_column[gene] for gene in names])
    print(
        f"[ref] genes: panel={len(panel_in_atlas)}/{len(panel)} hvg={len(hvg_names)} "
        f"markers={len(markers_in_atlas)} -> union={len(names)}",
        flush=True,
    )
    print(
        f"[ref] panel genes absent from the atlas: {[gene for gene in panel if gene not in first_column]}", flush=True
    )
    return names, columns, [gene_ids[i] for i in columns]


def subsample_rows(categories, codes, cap, seed):
    """Sorted atlas rows of the class-balanced subsample: at most `cap` cells per Level-3 type."""
    rng = np.random.default_rng(seed)
    rows = []
    for code, name in enumerate(categories):
        if name in DROPPED_LABELS:
            continue
        idx = np.where(codes == code)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        rows.append(idx)
    return np.sort(np.concatenate(rows))


def local_index(rows, n_obs):
    """Array that maps an atlas row to its position in `rows` (-1 for rows that are not selected)."""
    index = -np.ones(n_obs, dtype=np.int64)
    index[rows] = np.arange(len(rows))
    return index


def stream_counts(atlas, n_obs, chunk, column_index, row_indices, start):
    """Read layers/counts in row chunks; for each row selection return its (rows, columns, values) triplets."""
    counts = atlas["layers"]["counts"]
    indptr = counts["indptr"][:]
    data, indices = counts["data"], counts["indices"]
    triplets = [([], [], []) for _ in row_indices]
    for r0 in range(0, n_obs, chunk):
        r1 = min(r0 + chunk, n_obs)
        p0, p1 = int(indptr[r0]), int(indptr[r1])
        if p1 == p0:
            continue
        values = data[p0:p1]
        nnz_per_row = np.diff(indptr[r0 : r1 + 1])
        atlas_rows = r0 + np.repeat(np.arange(r1 - r0), nnz_per_row)
        local_columns = column_index[indices[p0:p1]]
        for row_index, (rows_out, columns_out, values_out) in zip(row_indices, triplets):
            local_rows = row_index[atlas_rows]
            keep = (local_rows >= 0) & (local_columns >= 0)
            if keep.any():
                rows_out.append(local_rows[keep].astype(np.int32))
                columns_out.append(local_columns[keep].astype(np.int32))
                values_out.append(values[keep].astype(np.float32))
        if (r0 // chunk) % 3 == 0:
            print(f"  streamed rows {r1}/{n_obs} ({100 * r1 / n_obs:.0f}%) t={time.time() - start:.0f}s", flush=True)
    return triplets


def assemble_csr(triplets, shape):
    """CSR matrix from the chunk-wise (rows, columns, values) lists of `stream_counts`."""
    rows, columns, values = (np.concatenate(part) for part in triplets)
    return sp.coo_matrix((values, (rows, columns)), shape=shape).tocsr()


def main():
    args = parse_args()
    start = time.time()
    print("[ref] opening the atlas with h5py:", args.atlas, flush=True)
    with h5py.File(args.atlas, "r") as atlas:
        n_obs, n_var = (int(x) for x in atlas["X"].attrs["shape"])
        print(f"[ref] atlas {n_obs} x {n_var}", flush=True)

        categories, codes = read_categorical(atlas["obs"], args.label)
        print(f"[ref] L3 types ({len(categories)}):", list(categories), flush=True)
        categories_l4, codes_l4 = read_categorical(atlas["obs"], args.label_l4)
        obs_index_key = decode(atlas["obs"].attrs.get("_index", "_index"))
        cell_names = np.array([decode(x) for x in atlas["obs"][obs_index_key][:]])  # atlas barcodes

        gene_names, gene_columns, gene_ids = select_genes(atlas, args.xenium, n_var)
        column_index = -np.ones(n_var, dtype=np.int64)
        column_index[gene_columns] = np.arange(len(gene_names))

        sub_rows = subsample_rows(categories, codes, args.cap, args.seed)
        n_sub = len(sub_rows)
        sub_index = local_index(sub_rows, n_obs)
        sub_label = np.array([categories[codes[r]] for r in sub_rows])
        print(f"[ref] subsample: {n_sub} cells across {len(set(sub_label))} L3 types (cap={args.cap})", flush=True)
        print("[ref] per-type n:", dict(collections.Counter(sub_label)), flush=True)

        # every row of the extra L3 classes (second output)
        extra_classes = [s.strip() for s in args.extra_l3.split(",") if s.strip()]
        if extra_classes and args.extra_out:
            extra_codes = [int(np.where(categories == c)[0][0]) for c in extra_classes]
            extra_rows = np.sort(np.where(np.isin(codes, extra_codes))[0])
        else:
            extra_rows = np.array([], dtype=np.int64)
        print(f"[ref] extra rows ({extra_classes}): {len(extra_rows)}", flush=True)

        if os.environ.get("DRY") or args.dry:
            print("[ref] dry run: plan only, nothing streamed", flush=True)
            return

        row_indices = [sub_index] + ([local_index(extra_rows, n_obs)] if len(extra_rows) else [])
        triplets = stream_counts(atlas, n_obs, args.chunk, column_index, row_indices, start)

    X = assemble_csr(triplets[0], (n_sub, len(gene_names)))
    print(f"[ref] assembled X {X.shape} nnz={X.nnz} t={time.time() - start:.0f}s", flush=True)

    var = pd.DataFrame({"gene_id_orig": gene_ids}, index=gene_names)
    reference = ad.AnnData(X=X, var=var)
    reference.obs["cell_type"] = pd.Categorical(sub_label)
    reference.obs["donor_id"] = "atlas"
    reference.obs["cell_type_L4"] = pd.Categorical([categories_l4[codes_l4[r]] for r in sub_rows])
    reference.obs["atlas_row"] = sub_rows
    reference.obs_names = cell_names[sub_rows]
    reference.layers["counts"] = reference.X.copy()
    print("[ref] reference:", reference.shape, "| label =", args.label, "| writing", args.out, flush=True)
    reference.write_h5ad(args.out, compression="gzip")

    if len(extra_rows):
        extra = ad.AnnData(X=assemble_csr(triplets[1], (len(extra_rows), len(gene_names))), var=var.copy())
        extra.obs["cell_type"] = pd.Categorical([categories[codes[r]] for r in extra_rows])
        extra.obs["cell_type_L4"] = pd.Categorical([categories_l4[codes_l4[r]] for r in extra_rows])
        extra.obs["atlas_row"] = extra_rows
        extra.obs["in_subsample"] = sub_index[extra_rows] >= 0
        extra.obs_names = cell_names[extra_rows]
        extra.layers["counts"] = extra.X.copy()
        print("[ref] extra reference:", extra.shape, extra.obs["cell_type_L4"].value_counts().to_dict(), flush=True)
        extra.write_h5ad(args.extra_out, compression="gzip")
    print(f"[ref] done t={time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
