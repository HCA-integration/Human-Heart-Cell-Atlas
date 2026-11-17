#!/usr/bin/env python
"""FICTURE input files from a square_002um matrix, with bounded memory (below ~1 GB).

For slides whose matrix is too large to sort by Y in memory (build_input_2um.py). The matrix is streamed in
blocks of bins and written unsorted to a temporary gzip, gene totals come from a chunked bincount, and the
transcripts are sorted by Y with an external sort on compressed temporary files.

Inputs: a Space Ranger square_002um folder.
Outputs (in --out-dir), same files and columns as build_input_2um.py: transcripts.tsv.gz (X, Y, gene, Count;
um, sorted by Y, mitochondrial genes dropped), feature.clean.tsv.gz, coordinate_minmax.tsv.

Usage: python build_input_2um_lowmem.py --slide-dir SQUARE_002UM --out-dir DIR [--sort-mem 256M]
"""

import argparse
import gzip
import os
import shutil
import subprocess
import time

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from hhca.log import get_logger
from visium_hd import is_mitochondrial, microns_per_pixel, write_coordinate_minmax, write_feature_table

log = get_logger("input")


def bin_coordinates(slide_dir, barcode_dataset, n_bins, mpp):
    """X, Y (um, float32) of every bin of the matrix.

    Matrix and positions written by regen_002um_from_feature_slice.py share one bin order, so parquet row i is
    barcode i of the h5; this is tested on 50 barcodes, and the positions are joined by barcode only if the
    order differs.
    """
    positions = pq.read_table(
        os.path.join(slide_dir, "spatial", "tissue_positions.parquet"),
        columns=["barcode", "pxl_row_in_fullres", "pxl_col_in_fullres"],
    ).to_pandas()
    assert len(positions) == n_bins, f"parquet rows {len(positions)} != n_bc {n_bins}"
    position_barcodes = positions["barcode"].values
    sampled = np.linspace(0, n_bins - 1, 50).astype(int)
    mismatches = [int(i) for i in sampled if position_barcodes[i] != barcode_dataset[int(i)].decode()]
    if mismatches:
        log(f"barcode order differs at {len(mismatches)} sampled positions: joining by barcode")
        barcodes = np.array([b.decode() for b in barcode_dataset[:]])
        order = pd.Series(np.arange(n_bins), index=position_barcodes).reindex(barcodes).values
        pxl_row = positions["pxl_row_in_fullres"].values[order]
        pxl_col = positions["pxl_col_in_fullres"].values[order]
    else:
        log("positions are in the barcode order of the matrix")
        pxl_row = positions["pxl_row_in_fullres"].values
        pxl_col = positions["pxl_col_in_fullres"].values
    x = (pxl_col.astype(float) * mpp).astype(np.float32)
    y = (pxl_row.astype(float) * mpp).astype(np.float32)
    return x, y


def sum_gene_totals(indices, data, nnz, n_genes, chunk):
    """Counts per gene over all bins, reading `chunk` non-zero entries at a time."""
    totals = np.zeros(n_genes, dtype=np.int64)
    for start in range(0, nnz, chunk):
        end = min(nnz, start + chunk)
        totals += np.bincount(indices[start:end], weights=data[start:end], minlength=n_genes).astype(np.int64)
    return totals


def write_unsorted_transcripts(path, matrix, indptr, x, y, gene_name, keep_gene, bin_block, precision):
    """Write X, Y, gene, Count of the kept genes in blocks of bins, without header; return the number of rows."""
    n_bins = x.size
    indices = matrix["indices"]
    data = matrix["data"]
    n_rows = 0
    with gzip.open(path, "wt") as gz:
        for bin0 in range(0, n_bins, bin_block):
            bin1 = min(n_bins, bin0 + bin_block)
            p0, p1 = int(indptr[bin0]), int(indptr[bin1])
            if p1 <= p0:
                continue
            gene = indices[p0:p1]
            count = data[p0:p1].astype(np.int32)
            entries_per_bin = np.diff(indptr[bin0 : bin1 + 1]).astype(np.int64)
            bin_of_entry = np.repeat(np.arange(bin0, bin1, dtype=np.int64), entries_per_bin)
            keep = keep_gene[gene]
            gene = gene[keep]
            count = count[keep]
            bin_of_entry = bin_of_entry[keep]
            if gene.size == 0:
                continue
            pd.DataFrame({"X": x[bin_of_entry], "Y": y[bin_of_entry], "gene": gene_name[gene], "Count": count}).to_csv(
                gz, sep="\t", index=False, header=False, float_format=f"%.{precision}f"
            )
            n_rows += gene.size
            if (bin0 // bin_block) % 4 == 0:
                log(f"  bins {bin1:,}/{n_bins:,}  rows={n_rows:,}")
    return n_rows


def sort_by_y(unsorted, out, out_dir, sort_mem):
    """External sort of the unsorted transcripts by Y into `out`; the unsorted file is removed."""
    sort_dir = os.path.join(out_dir, "_sorttmp")
    os.makedirs(sort_dir, exist_ok=True)
    # the unsorted file has no header: write the header, then append the sorted rows
    with gzip.open(out, "wt") as handle:
        handle.write("X\tY\tgene\tCount\n")
    sort_command = (
        f"gzip -cd {unsorted} | LC_ALL=C sort -T {sort_dir} -S {sort_mem} "
        f"--compress-program=gzip -t$'\\t' -k2,2g | gzip -c >> {out}"
    )
    returncode = subprocess.call(["bash", "-c", sort_command])
    shutil.rmtree(sort_dir, ignore_errors=True)
    if returncode != 0 or not os.path.exists(out) or os.path.getsize(out) < 100:
        raise SystemExit(f"external sort failed (rc={returncode})")
    os.remove(unsorted)


def parse_args():
    """Command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slide-dir", required=True, help="Space Ranger square_002um folder")
    parser.add_argument("--out-dir", required=True, help="folder for the FICTURE input files")
    parser.add_argument("--precision", type=int, default=2, help="decimals of the written coordinates")
    parser.add_argument("--bin-block", type=int, default=100_000, help="bins per written block")
    parser.add_argument(
        "--idx-chunk", type=int, default=10_000_000, help="non-zero entries per chunk of the gene totals"
    )
    parser.add_argument("--sort-mem", default="256M", help="memory of the external sort (sort -S)")
    return parser.parse_args()


def main():
    """Write the three FICTURE input files of a whole slide."""
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    mpp = microns_per_pixel(args.slide_dir)
    log(f"microns_per_pixel = {mpp:.6f}")

    start = time.time()
    unsorted = os.path.join(args.out_dir, "_transcripts.unsorted.tsv.gz")
    with h5py.File(os.path.join(args.slide_dir, "filtered_feature_bc_matrix.h5"), "r") as f:
        matrix = f["matrix"]
        n_genes, n_bins = [int(x) for x in matrix["shape"][:]]
        indptr = matrix["indptr"][:]
        gene_name = np.array([b.decode() for b in matrix["features"]["name"][:]])
        gene_id = np.array([b.decode() for b in matrix["features"]["id"][:]])
        nnz = int(indptr[-1])
        log(f"matrix {n_genes} genes x {n_bins:,} bins; nnz={nnz:,}")

        # the barcodes are read on demand (millions of strings)
        x, y = bin_coordinates(args.slide_dir, matrix["barcodes"], n_bins, mpp)

        # gene totals, before the mitochondrial genes are dropped from the rows
        gene_totals = sum_gene_totals(matrix["indices"], matrix["data"], nnz, n_genes, args.idx_chunk)
        log("feature totals done")

        n_rows = write_unsorted_transcripts(
            unsorted, matrix, indptr, x, y, gene_name, ~is_mitochondrial(gene_name), args.bin_block, args.precision
        )
        log(f"unsorted transcripts: {n_rows:,} rows -> {unsorted}")

    out_minmax = os.path.join(args.out_dir, "coordinate_minmax.tsv")
    write_coordinate_minmax(out_minmax, x, y, args.precision)
    log(f"wrote {out_minmax}: X[{x.min():.0f},{x.max():.0f}] Y[{y.min():.0f},{y.max():.0f}] um")

    n_features = write_feature_table(
        os.path.join(args.out_dir, "feature.clean.tsv.gz"), gene_name, gene_id, gene_totals
    )
    log(f"wrote feature.clean.tsv.gz: {n_features} genes (MT- dropped)")

    out = os.path.join(args.out_dir, "transcripts.tsv.gz")
    log("external sort by Y ...")
    sort_by_y(unsorted, out, args.out_dir, args.sort_mem)
    log(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB) in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
