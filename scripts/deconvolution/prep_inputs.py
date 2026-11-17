#!/usr/bin/env python
"""Write the inputs every deconvolution method reads, so that all methods see the same spots and genes.

Per slide: Space Ranger filtered_feature_bc_matrix.h5 and spatial/tissue_positions, in-tissue spots,
filter_genes(min_cells=3), filter_cells(min_counts=200) (the same filters as the NNLS run). Each method
intersects genes with the reference inside its own runner.

Inputs:  <raw>/<slide>/ (Space Ranger files), the shared reference h5ad
Outputs: <outdir>/<slide>/slide.h5ad         raw counts in X, obsm['spatial'] (Python methods)
         <outdir>/<slide>/matrix.mtx.gz      genes x spots raw counts, genes.txt, barcodes.txt (R methods)
         <outdir>/<slide>/coords.csv         barcode, total_counts, x, y, array_row, array_col
         <ref-outdir>/ref_matrix.mtx.gz      genes x cells raw counts, ref_genes.txt, ref_meta.csv (R methods)
         <summary>                           spots and genes per slide

Usage: prep_inputs.py --raw DIR --outdir DIR --ref REF.h5ad --ref-outdir DIR --summary CSV [--slides a,b]
"""

import argparse
import os
import traceback

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from hhca.io import write_mtx_gz
from hhca.log import get_logger
from visium_io import find_positions

POSITION_COLUMNS = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Method inputs: slides and reference as h5ad and MatrixMarket")
    parser.add_argument(
        "--raw", required=True, help="directory with <slide>/filtered_feature_bc_matrix.h5 and spatial/"
    )
    parser.add_argument("--outdir", required=True, help="output directory of the slide inputs")
    parser.add_argument("--ref", default="", help="shared reference h5ad (build_reference.py)")
    parser.add_argument("--ref-outdir", default="", help="output directory of the reference MatrixMarket export")
    parser.add_argument("--summary", required=True, help="output csv with spots and genes per slide")
    parser.add_argument(
        "--slides",
        help="comma-separated slide names",
        default="HHH1-LA,HHH2-LV-endo-1,HHH4-RA2,HHH4-RV2,A17-AH1-C1,A18-AH2-D1,"
        "HHH2-D1,HHH2-LV-endo-2,HHH4-A1,HHH5-B1,HHH5-C1",
    )
    return parser.parse_args()


def prep_slide(sample, raw_dir, out_dir, log):
    """Filter one slide and write its h5ad, MatrixMarket export and coordinates; return spots, genes, has_spatial."""
    slide_out = f"{out_dir}/{sample}"
    adata = sc.read_10x_h5(f"{raw_dir}/{sample}/filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()
    spatial_dir = f"{raw_dir}/{sample}/spatial"
    pos_path = find_positions(spatial_dir) if os.path.isdir(spatial_dir) else None
    has_spatial = False
    if pos_path is not None:
        pos = pd.read_csv(pos_path)
        if list(pos.columns[:2]) != ["barcode", "in_tissue"]:
            pos = pd.read_csv(pos_path, header=None, names=POSITION_COLUMNS)
        pos = pos.set_index("barcode").reindex(adata.obs_names)
        keep = pos["in_tissue"].notna().values & (pos["in_tissue"].fillna(0).values == 1)
        adata = adata[keep].copy()
        pos = pos.loc[adata.obs_names]
        adata.obs["array_row"] = pos["array_row"].astype(int).values
        adata.obs["array_col"] = pos["array_col"].astype(int).values
        adata.obsm["spatial"] = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)
        has_spatial = True
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_counts=200)
    adata.X = sp.csr_matrix(adata.X).astype(np.float32)
    adata.obs["total_counts"] = np.asarray(adata.X.sum(1)).ravel()
    os.makedirs(slide_out, exist_ok=True)
    adata.write(f"{slide_out}/slide.h5ad")
    # genes x spots export for the R methods
    write_mtx_gz(adata.X.T, f"{slide_out}/matrix.mtx.gz")
    with open(f"{slide_out}/genes.txt", "w") as handle:
        handle.write("\n".join(map(str, adata.var_names)) + "\n")
    with open(f"{slide_out}/barcodes.txt", "w") as handle:
        handle.write("\n".join(map(str, adata.obs_names)) + "\n")
    coords = pd.DataFrame({"barcode": adata.obs_names, "total_counts": adata.obs["total_counts"].values})
    if has_spatial:
        coords["x"] = adata.obsm["spatial"][:, 0]
        coords["y"] = adata.obsm["spatial"][:, 1]
        coords["array_row"] = adata.obs["array_row"].values
        coords["array_col"] = adata.obs["array_col"].values
    coords.to_csv(f"{slide_out}/coords.csv", index=False)
    log(f"  {sample}: {adata.n_obs} spots x {adata.n_vars} genes spatial={has_spatial}")
    return adata.n_obs, adata.n_vars, has_spatial


def prep_reference(ref_h5ad, ref_out, log):
    """Write the reference as genes x cells MatrixMarket with gene list and cell metadata (R methods)."""
    adata = sc.read_h5ad(ref_h5ad)
    adata.X = sp.csr_matrix(adata.X)
    write_mtx_gz(adata.X.T, f"{ref_out}/ref_matrix.mtx.gz")
    with open(f"{ref_out}/ref_genes.txt", "w") as handle:
        handle.write("\n".join(map(str, adata.var_names)) + "\n")
    meta = adata.obs.copy()
    meta.insert(0, "barcode", adata.obs_names)
    meta.to_csv(f"{ref_out}/ref_meta.csv", index=False)
    log(f"  reference: {adata.n_obs} cells x {adata.n_vars} genes; types={adata.obs['cell_type_L3'].nunique()}")


def main():
    """Export the reference, then every slide, and write the per-slide summary."""
    args = parse_args()
    log = get_logger()
    slides = [slide for slide in args.slides.split(",") if slide]
    os.makedirs(args.ref_outdir, exist_ok=True)
    log("reference")
    prep_reference(args.ref, args.ref_outdir, log)
    rows = []
    for slide in slides:
        # a slide that cannot be read is reported and left out of the summary; the others are still written
        try:
            n_spots, n_genes, has_spatial = prep_slide(slide, args.raw, args.outdir, log)
            rows.append((slide, n_spots, n_genes, has_spatial))
        except (OSError, KeyError, ValueError) as error:
            log(f"  {slide} failed: {type(error).__name__}: {error}")
            traceback.print_exc()
    pd.DataFrame(rows, columns=["slide", "n_spots", "n_genes", "has_spatial"]).to_csv(args.summary, index=False)
    log("done:", len(rows), "slides")


if __name__ == "__main__":
    main()
