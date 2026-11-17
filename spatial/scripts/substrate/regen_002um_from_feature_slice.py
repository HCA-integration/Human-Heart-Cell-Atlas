#!/usr/bin/env python
"""Write the Space Ranger square_002um layout of a Visium HD slide from its feature_slice.h5.

Space Ranger runs in cell-segmentation mode come without binned_outputs/. feature_slice.h5 holds, per feature
with at least one UMI, a sparse slice feature_slices/<feature_index>/{row,col,data} on the 2 um bin grid;
masks/filtered marks the in-tissue bins; the metadata_json attribute gives nrows, ncols, spot_pitch and the
affine spot_colrow_to_microscope_colrow from a (col, row) bin to a full-resolution image pixel.
microns_per_pixel comes from the scale of the affine: one bin step of spot_pitch um spans the mean column norm
of the affine in pixels, so mpp = spot_pitch / mean(column norms).

Inputs: feature_slice.h5 of the Space Ranger run.
Outputs (the three files the FICTURE input builders read, in --out-dir):
  filtered_feature_bc_matrix.h5       10x CSC matrix, genes x in-tissue bins
  spatial/tissue_positions.parquet    barcode, array_row / array_col, in_tissue, pxl_*_in_fullres
  spatial/scalefactors_json.json      microns_per_pixel, bin_size_um, spot_diameter_fullres

Usage: python regen_002um_from_feature_slice.py --feature-slice feature_slice.h5 --out-dir SQUARE_002UM
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from hhca.log import get_logger

log = get_logger("regen")


def read_features(f):
    """Name, id, feature type and genome of all features, as in the 10x matrix."""
    return {
        key: np.array([b.decode() for b in f["features"][key][:]]) for key in ("name", "id", "feature_type", "genome")
    }


def tissue_bins(f, nrows, ncols):
    """Grid row and column of the in-tissue bins ordered by (row, col), and the grid position -> bin lookup."""
    bin_row = f["masks/filtered/row"][:].astype(np.int64)
    bin_col = f["masks/filtered/col"][:].astype(np.int64)
    in_tissue = f["masks/filtered/data"][:] != 0
    bin_row = bin_row[in_tissue]
    bin_col = bin_col[in_tissue]
    grid_position = bin_row * ncols + bin_col
    order = np.argsort(grid_position, kind="stable")
    bin_row = bin_row[order]
    bin_col = bin_col[order]
    grid_position = grid_position[order]
    # grid position -> bin index (-1 = not in tissue)
    lookup = np.full(nrows * ncols, -1, dtype=np.int32)
    lookup[grid_position] = np.arange(bin_row.size, dtype=np.int32)
    return bin_row, bin_col, lookup


def tissue_counts(feature_slices, lookup, ncols):
    """Gene index, bin index and count of every non-zero entry that lies in an in-tissue bin."""
    keys = list(feature_slices.keys())
    total_nnz = 0
    for key in keys:
        total_nnz += feature_slices[key]["data"].shape[0]
    log(f"feature slices: {len(keys)} genes; {total_nnz:,} non-zero entries")
    gene_index = np.empty(total_nnz, dtype=np.int32)
    bin_index = np.empty(total_nnz, dtype=np.int32)
    count = np.empty(total_nnz, dtype=np.int32)
    filled = 0
    for n_done, key in enumerate(keys, start=1):
        node = feature_slices[key]
        row = node["row"][:].astype(np.int64)
        col = node["col"][:].astype(np.int64)
        data = node["data"][:]
        bin_of_entry = lookup[row * ncols + col]
        in_tissue = bin_of_entry >= 0
        n_kept = int(in_tissue.sum())
        if n_kept:
            block = slice(filled, filled + n_kept)
            gene_index[block] = int(key)
            bin_index[block] = bin_of_entry[in_tissue]
            count[block] = data[in_tissue].astype(np.int32)
            filled += n_kept
        if n_done % 3000 == 0:
            log(f"  {n_done}/{len(keys)} genes; kept nnz so far={filled:,}")
    log(f"kept nnz (in-tissue): {filled:,} / {total_nnz:,}")
    return gene_index[:filled], bin_index[:filled], count[:filled]


def write_matrix_h5(path, counts, barcodes, features, sample_id, gzip_level):
    """Write filtered_feature_bc_matrix.h5 in the 10x layout."""
    compression = {"compression": "gzip", "compression_opts": gzip_level}
    with h5py.File(path, "w") as out:
        out.attrs["filetype"] = "matrix"
        out.attrs["version"] = 2
        out.attrs["chemistry_description"] = np.bytes_("Visium HD v1")
        out.attrs["library_ids"] = np.array([np.bytes_(sample_id)])
        out.attrs["original_gem_groups"] = np.array([1])
        out.attrs["software_version"] = np.bytes_("regen_from_feature_slice")
        matrix = out.create_group("matrix")
        matrix.create_dataset("data", data=counts.data.astype(np.int32), **compression)
        matrix.create_dataset("indices", data=counts.indices.astype(np.int64), **compression)
        matrix.create_dataset("indptr", data=counts.indptr.astype(np.int64), **compression)
        matrix.create_dataset("shape", data=np.array(counts.shape, dtype=np.int32))
        matrix.create_dataset("barcodes", data=barcodes, **compression)
        group = matrix.create_group("features")
        group.create_dataset("_all_tag_keys", data=np.array([b"genome"]))
        for key in ("name", "id", "feature_type", "genome"):
            group.create_dataset(key, data=np.array([s.encode() for s in features[key]]), **compression)


def parse_args():
    """Command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--feature-slice", required=True, help="Space Ranger feature_slice.h5")
    parser.add_argument("--out-dir", required=True, help="square_002um folder to write")
    parser.add_argument("--gzip-level", type=int, default=4, help="gzip level of the h5 datasets")
    return parser.parse_args()


def main():
    """Rebuild the square_002um matrix, bin positions and scale factors of one slide."""
    args = parse_args()
    spatial_dir = os.path.join(args.out_dir, "spatial")
    os.makedirs(spatial_dir, exist_ok=True)

    start = time.time()
    with h5py.File(args.feature_slice, "r") as f:
        metadata = json.loads(f.attrs["metadata_json"])
        nrows = int(metadata["nrows"])
        ncols = int(metadata["ncols"])
        spot_pitch = float(metadata["spot_pitch"])
        affine = np.array(metadata["transform_matrices"]["spot_colrow_to_microscope_colrow"], float)
        log(f"sample_id={metadata.get('sample_id')} nrows={nrows} ncols={ncols} spot_pitch={spot_pitch}")

        features = read_features(f)
        n_genes = features["name"].size
        log(f"features: {n_genes}")

        bin_row, bin_col, lookup = tissue_bins(f, nrows, ncols)
        n_bins = bin_row.size
        log(f"filtered in-tissue bins: {n_bins:,}")
        barcodes = np.array([f"s_002um_{r:05d}_{c:05d}-1" for r, c in zip(bin_row, bin_col)], dtype="S21")

        # full-resolution pixel of each bin: [pxl_col, pxl_row, 1]^T = affine @ [bin_col, bin_row, 1]^T
        pixels = np.column_stack([bin_col.astype(float), bin_row.astype(float), np.ones(n_bins)]) @ affine.T
        pxl_col = pixels[:, 0]
        pxl_row = pixels[:, 1]
        px_per_bin = float(np.mean([np.hypot(affine[0, 0], affine[1, 0]), np.hypot(affine[0, 1], affine[1, 1])]))
        mpp = spot_pitch / px_per_bin
        log(f"px_per_bin={px_per_bin:.4f}  microns_per_pixel={mpp:.6f}")

        gene_index, bin_index, count = tissue_counts(f["feature_slices"], lookup, ncols)
        del lookup

    log("building CSC (genes x bins) ...")
    counts = sp.coo_matrix((count, (gene_index, bin_index)), shape=(n_genes, n_bins)).tocsc()
    del gene_index, bin_index, count
    counts.sum_duplicates()
    log(f"CSC: shape={counts.shape} nnz={counts.nnz:,}")

    out_h5 = os.path.join(args.out_dir, "filtered_feature_bc_matrix.h5")
    log(f"writing {out_h5} (gzip={args.gzip_level}) ...")
    write_matrix_h5(out_h5, counts, barcodes, features, metadata.get("sample_id", "regen"), args.gzip_level)
    log(f"wrote h5 ({os.path.getsize(out_h5) / 1e6:.1f} MB)")

    out_positions = os.path.join(spatial_dir, "tissue_positions.parquet")
    positions = pd.DataFrame(
        {
            "barcode": [b.decode() for b in barcodes],
            "in_tissue": np.ones(n_bins, dtype=np.int64),
            "array_row": bin_row.astype(np.int64),
            "array_col": bin_col.astype(np.int64),
            "pxl_row_in_fullres": pxl_row,
            "pxl_col_in_fullres": pxl_col,
        }
    )
    positions.to_parquet(out_positions, index=False)
    log(f"wrote {out_positions} ({os.path.getsize(out_positions) / 1e6:.1f} MB)")

    scalefactors = {
        "spot_diameter_fullres": spot_pitch / mpp,
        "bin_size_um": spot_pitch,
        "microns_per_pixel": mpp,
        "regen_from_feature_slice": True,
        "px_per_bin": px_per_bin,
        "nrows": nrows,
        "ncols": ncols,
        "spot_colrow_to_microscope_colrow": affine.tolist(),
    }
    with open(os.path.join(spatial_dir, "scalefactors_json.json"), "w") as handle:
        json.dump(scalefactors, handle, indent=2)
    log("wrote scalefactors_json.json")
    log(f"done in {time.time() - start:.0f}s: {n_genes} genes x {n_bins:,} bins, mpp={mpp:.6f}")


if __name__ == "__main__":
    main()
