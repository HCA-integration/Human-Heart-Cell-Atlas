#!/usr/bin/env python
"""Ground-truth simulation: aggregate one Visium HD section into 55 um pseudo-spots of known composition.

The section is tiled into 55 um squares (hhca.tl.simulate_spots; settings: config gtsim). Per tile, the
pseudo-spot counts are the summed 2 um transcript counts per gene, and the known composition is the share of
decoded 2 um bins of each Level-3 type (an area fraction, not a nucleus count; the HD sections are ventricular).
K1 of the decode is the 0-based index into the factor order.

Inputs:  <slide-dir>/inputs/transcripts.tsv.gz        X, Y, gene, Count (um)
         <slide-dir>/inputs/feature.clean.tsv.gz      gene list
         <slide-dir>/inputs/coordinate_minmax.tsv     section bounds (um)
         <slide-dir>/output/<MODEL>.pixel.tsv.gz      FICTURE 2 um decode (X, Y, K1)
Outputs: <out>/<slide>/slide.h5ad, matrix.mtx.gz, genes.txt, barcodes.txt, coords.csv (the layout of
         prep_inputs.py), truth_proportions.csv.gz (rows sum to 1), truth_pixelcounts.csv.gz, sim_meta.json

Usage: build_sim.py --slide-dir DIR --factor-order JSON --out DIR [--config CONFIG] SLIDE
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import deconv_config
import hhca
from hhca.io import write_mtx_gz
from hhca.log import get_logger

MODEL = "atlas_L3.decode.prj_12.r_4.0_5.0"


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slide-dir", required=True, help="decode directory of the slide (inputs/ and output/)")
    parser.add_argument("--factor-order", required=True, help="json list of the Level-3 factor names")
    parser.add_argument("--out", required=True, help="output root; files go to <out>/<slide>/")
    deconv_config.add_argument(parser)
    parser.add_argument("slide", help="slide name")
    return parser.parse_args()


def main():
    """Tile one section, keep the tiles with enough counts and decoded bins, and write the simulated slide."""
    args = parse_args()
    log = get_logger()
    settings = deconv_config.load(args.config)["gtsim"]
    tile_um, min_counts, min_pixels = settings["tile_um"], settings["min_counts"], settings["min_pixels"]
    with open(args.factor_order) as handle:
        factor_order = json.load(handle)
    slide = args.slide
    slide_dir = args.slide_dir
    transcript_path = f"{slide_dir}/inputs/transcripts.tsv.gz"
    feature_path = f"{slide_dir}/inputs/feature.clean.tsv.gz"
    pixel_path = f"{slide_dir}/output/{MODEL}.pixel.tsv.gz"
    bounds_path = f"{slide_dir}/inputs/coordinate_minmax.tsv"
    for path in (transcript_path, feature_path, pixel_path, bounds_path):
        assert os.path.exists(path), f"missing {path}"

    bounds = pd.read_csv(bounds_path, sep="\t", header=None, index_col=0)[1].to_dict()
    genes = pd.read_csv(feature_path, sep="\t")["gene"].astype(str).tolist()
    log(f"{slide}: genes={len(genes)}, tiles of {tile_um} um")
    adata = hhca.tl.simulate_spots(
        transcript_path,
        pixel_path,
        genes,
        bounds,
        factor_order,
        tile_um,
        min_counts,
        min_pixels,
        settings["min_cells"],
        chunksize=settings["chunk"],
        log=log,
    )
    truth_df, pixcount_df = adata.obsm.pop("truth"), adata.obsm.pop("truth_pixels")
    truth_df.index.name = "barcode"
    pixcount_df.index.name = "barcode"
    grid = adata.uns.pop("grid")
    log(f"tiles passing (counts>={min_counts} & pix>={min_pixels}): {adata.n_obs} / {grid['n_tiles']}")
    log(f"genes passing (min_cells>={settings['min_cells']}): {adata.n_vars} / {grid['n_genes']}")

    # write the slide in the layout of the method inputs
    out = f"{args.out}/{slide}"
    os.makedirs(out, exist_ok=True)
    adata.uns["sim"] = {
        "slide": slide,
        "tile_um": tile_um,
        "min_counts": min_counts,
        "min_pixels": min_pixels,
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "source": "FICTURE 2um L3 decode",
    }
    adata.write(f"{out}/slide.h5ad")

    write_mtx_gz(adata.X.T, f"{out}/matrix.mtx.gz")
    with open(f"{out}/genes.txt", "w") as handle:
        handle.write("\n".join(map(str, adata.var_names)) + "\n")
    with open(f"{out}/barcodes.txt", "w") as handle:
        handle.write("\n".join(map(str, adata.obs_names)) + "\n")
    coords = pd.DataFrame(
        {
            "barcode": adata.obs_names,
            "total_counts": adata.obs["total_counts"].values,
            "x": adata.obsm["spatial"][:, 0],
            "y": adata.obsm["spatial"][:, 1],
            "array_row": adata.obs["array_row"].values,
            "array_col": adata.obs["array_col"].values,
        }
    )
    coords.to_csv(f"{out}/coords.csv", index=False)
    truth_df.to_csv(f"{out}/truth_proportions.csv.gz", compression="gzip")
    pixcount_df.to_csv(f"{out}/truth_pixelcounts.csv.gz", compression="gzip")
    meta = {
        "slide": slide,
        "tile_um": tile_um,
        "min_counts_per_spot": min_counts,
        "min_bins_per_spot": min_pixels,
        "min_cells_per_gene": settings["min_cells"],
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "grid": [grid["nx"], grid["ny"]],
        "n_transcripts": grid["n_transcripts"],
        "n_decoded_pixels": grid["n_decoded_pixels"],
        "bounds_um": {key: float(bounds[key]) for key in ("xmin", "xmax", "ymin", "ymax")},
        "median_counts_per_spot": float(np.median(adata.obs["total_counts"])),
        "median_bins_per_spot": float(np.median(adata.obs["n_bins"])),
        "mean_truth_top": truth_df.mean().sort_values(ascending=False).head(8).round(4).to_dict(),
    }
    with open(f"{out}/sim_meta.json", "w") as handle:
        json.dump(meta, handle, indent=2)
    log("wrote", out)
    log(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
