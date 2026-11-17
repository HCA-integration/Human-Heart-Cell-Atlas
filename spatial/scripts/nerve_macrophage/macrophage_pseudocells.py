#!/usr/bin/env python
"""Build macrophage pseudo-cells on the 11 Visium HD slides (data behind Fig 5 k-l).

Per slide, the 2 um decode pixels whose top factor (K1) is Macrophages are joined to the nearest in-tissue
SpaceRanger 2 um bin and clustered with DBSCAN (eps 10 um, min_samples 3); clusters of more than 50 pixels are
split into 12 um grid tiles (tiles with >= 3 pixels kept). A pseudo-cell's counts are the sum of the 2 um counts
of its unique member bins; pseudo-cells with >= 100 counts are kept (hhca.tl.pseudocells; config
`macrophage.pseudocells`). Each pseudo-cell gets the nearest vessel structure (by structure centroid) of the calibre
table of pipeline `vascular`: distance, equivalent diameter and calibre class (hhca.tl.nearest_structure); with
--caliber none these columns are NaN. Per-slide h5ads in <outdir>/cache are reused when present.

Inputs: FICTURE 2 um decodes, SpaceRanger square_002um outputs, the Level-3 factor order, the calibre table.
Outputs: <outdir>/cache/Macrophage__<slide>.h5ad, <outdir>/cache/Macrophage_pseudocells_allslides.h5ad,
<outdir>/tables/macrophage_pseudocell_counts.csv and <outdir>/logs/build_summary.json.

Usage: python macrophage_pseudocells.py --decode DECODE_TEMPLATE --spaceranger SQUARE_002UM_TEMPLATE
           --factor-order atlas_L3_factor_order.json --caliber CALIBER.csv --outdir DIR
"""

import argparse
import gc
import json
import os

from hhca import thread_env

os.environ.update(thread_env(1))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import anndata as ad
import numpy as np
import pandas as pd

import hhca
from hhca.config import load_config
from hhca.log import get_logger

NAME = "Macrophage"  # name of the pseudo-cells in the file names and in obs["cell_type"]

log = get_logger()


def build_slide(slide, structures, decode, spaceranger, factor, settings):
    """Pseudo-cells of one slide with their nearest vessel structure.

    decode: 2 um decode pixel table of the slide; spaceranger: its square_002um folder; structures: calibre
    table of all slides, or None.
    """
    bins = hhca.io.read_visium_hd(spaceranger)
    bins.var_names_make_unique()
    pixels = hhca.io.read_decode(decode)
    if structures is not None:
        structures = structures[structures.slide == slide]
        assert len(structures) > 0, f"no vessel structures for {slide} in caliber table"
    log(
        f"{slide}: decode {len(pixels):,}px, bins {bins.n_obs:,}x{bins.n_vars}, "
        f"vessels {len(structures) if structures is not None else 'NA'}"
    )
    cells = hhca.tl.pseudocells(bins, pixels, factor, **settings)
    cells.obs.insert(0, "slide", slide)
    cells.obs.insert(1, "cell_type", NAME)
    hhca.tl.nearest_structure(cells.obs, structures)
    cells.obs_names = [f"{NAME}__{slide}__{i}" for i in range(cells.n_obs)]
    log(f"  kept {cells.n_obs} (>= {settings['min_counts']} counts), median {np.median(cells.obs.total_counts):.0f}")
    return cells


def pool_slides(paths, cache_dir):
    """Concatenate the per-slide pseudo-cells, write the pooled h5ad and return its summary."""
    pooled = ad.concat([ad.read_h5ad(path) for path in paths], join="inner", index_unique=None)
    pooled_path = f"{cache_dir}/{NAME}_pseudocells_allslides.h5ad"
    pooled.write_h5ad(pooled_path)
    classes = pooled.obs.class_near.value_counts().to_dict()
    log(f"pooled: {pooled.n_obs} pseudo-cells, {pooled.n_vars} genes -> {pooled_path}")
    log(f"  classes: {classes}")
    return {
        "cell_type": NAME,
        "n_pseudocells": int(pooled.n_obs),
        "n_genes": int(pooled.n_vars),
        "n_slides": len(paths),
        "n_small": int(classes.get("small_resistance", 0)),
        "n_intermediate": int(classes.get("intermediate", 0)),
        "n_large": int(classes.get("large_conductance", 0)),
        "median_counts": float(np.median(pooled.obs.total_counts)),
        "median_bins": float(np.median(pooled.obs.n_bins)),
        "median_dist_vessel_um": float(np.median(pooled.obs.dist_vessel_struct_um)),
    }


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--decode", required=True, help="2 um decode pixel.tsv.gz, path template with {s} for the slide"
    )
    parser.add_argument("--spaceranger", required=True, help="square_002um folder, path template with {s}")
    parser.add_argument(
        "--spaceranger-ah", default=None, help="square_002um template of the VS_AH* slides (default: --spaceranger)"
    )
    parser.add_argument("--factor-order", required=True, help="Level-3 factor order json of the FICTURE model")
    parser.add_argument("--caliber", required=True, help="caliber_per_structure.csv, or 'none'")
    parser.add_argument("--outdir", required=True, help="output folder; cache/, logs/ and tables/ are created in it")
    parser.add_argument("--slides", nargs="+", default=None, help="slide ids (default: all 11)")
    return parser.parse_args()


def main():
    """Build the pseudo-cells of every slide (reusing cached slides), pool them and write the summary."""
    args = parse_args()
    cfg = load_config("nerve_macrophage")
    settings = cfg["macrophage"]["pseudocells"]
    spaceranger_ah = args.spaceranger_ah or args.spaceranger
    # the slides are pooled with the Berlin slides in alphabetical order, followed by the AH slides
    slides = list(args.slides) if args.slides else sorted(cfg["slides"]["berlin"]) + cfg["slides"]["ah"]
    cache_dir, log_dir, table_dir = f"{args.outdir}/cache", f"{args.outdir}/logs", f"{args.outdir}/tables"
    with open(args.factor_order) as handle:
        factor = json.load(handle).index(cfg["macrophage"]["cell_type"])
    for folder in (cache_dir, log_dir, table_dir):
        os.makedirs(folder, exist_ok=True)
    structures = None if args.caliber.lower() == "none" else pd.read_csv(args.caliber)
    paths = []
    for slide in slides:
        path = f"{cache_dir}/{NAME}__{slide}.h5ad"
        if not os.path.exists(path):
            spaceranger = (spaceranger_ah if slide.startswith("VS_AH") else args.spaceranger).format(s=slide)
            build_slide(slide, structures, args.decode.format(s=slide), spaceranger, factor, settings).write_h5ad(path)
            gc.collect()
        paths.append(path)

    summary = [pool_slides(paths, cache_dir)]
    pd.DataFrame(summary).to_csv(f"{table_dir}/macrophage_pseudocell_counts.csv", index=False)
    with open(f"{log_dir}/build_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    log("done")


if __name__ == "__main__":
    main()
