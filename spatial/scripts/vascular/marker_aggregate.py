#!/usr/bin/env python
"""Fig 4 h, per slide: pooled marker counts per vessel-wall compartment from the SpaceRanger 2 um matrix.

Genes: the vessel-wall panel, every connexin and the tight-junction genes of `markers` in config/vascular.yaml;
the per-slide tables are pooled by marker_pool.py (Fig 4 h matrix).

Compartments (hhca.tl.wall_compartments): each 2 um key gets the L3 type of its highest-P1 decode pixel; the
seven vessel types are kept and capillary EC is split by a pericyte bin within `pericyte_radius_um`. The extra
column "Venous EC (muscular vein)" is every SpaceRanger 2 um bin, of any type, whose lattice cell belongs to a
Venous(muscular) structure (vessel_structures.py). Per compartment (hhca.tl.marker_counts): summed marker counts,
bins with >= 1 count, bin count and summed total counts (all features of the slide's matrix).

With --flags (CMFLAG-clean), 2 um bins carrying any of the four spillover flags are removed first: their decode
pixels do not enter the argmax or the pericyte split, and their SpaceRanger counts do not enter any compartment,
the muscular-vein column included.

Inputs: the FICTURE pixel decode and Level-3 factor order, the SpaceRanger square_002um matrix, the tables of
        vessel_structures.py and, with --flags, the CMFLAG flags of the slide.
Outputs: <out-dir>/<slide>.csv (slide, type, gene, marker_sum, n_expr, n_bins, total_counts), <slide>.meta.json

Usage: python marker_aggregate.py --slide S --decode DECODE.tsv.gz --factor-order L3_ORDER.json
           --spaceranger SQUARE_002UM_DIR --structures-dir DIR --out-dir DIR [--flags FLAGS.parquet]
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

from hhca import thread_env

os.environ.update(thread_env(1))

import pandas as pd

import hhca
from hhca.config import load_config
from hhca.grid import spillover_keys
from hhca.log import get_logger

warnings.filterwarnings("ignore")

log = get_logger()


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slide", required=True, help="slide id")
    parser.add_argument("--decode", required=True, help="FICTURE pixel decode of the slide")
    parser.add_argument("--factor-order", required=True, help="JSON list of the Level-3 types in factor order")
    parser.add_argument("--spaceranger", required=True, help="binned_outputs/square_002um directory")
    parser.add_argument("--structures-dir", required=True, help="directory with the vessel-structure tables")
    parser.add_argument("--out-dir", required=True, help="output directory (one table per slide)")
    parser.add_argument("--flags", default=None, help="CMFLAG flags of the slide (<slide>_clean_bins.parquet)")
    return parser.parse_args()


def main():
    """Assign the 2 um bins of one slide to compartments and sum the marker counts per compartment."""
    args = parse_args()
    slide = args.slide
    os.makedirs(args.out_dir, exist_ok=True)
    table_path = f"{args.out_dir}/{slide}.csv"
    if os.path.exists(table_path):
        log(f"{slide}: output exists, skipped")
        return
    cfg = load_config("vascular")
    structures, markers = cfg["structures"], cfg["markers"]
    genes = sorted(set(markers["panel"] + markers["connexins"] + markers["tight_junctions"]))
    with open(args.factor_order) as handle:
        factor_order = json.load(handle)
    spillover = spillover_keys(args.flags) if args.flags else None

    log(f"{slide}: reading decode ...")
    pixels = hhca.io.read_decode(args.decode, columns=("X", "Y", "K1", "P1"))
    vessels = hhca.io.read_vessels(
        f"{args.structures_dir}/{slide}_cells.parquet", structures["vascular_types"], factor_order
    )
    structs = pd.read_csv(f"{args.structures_dir}/{slide}_structs.csv")
    adata = hhca.io.read_visium_hd(args.spaceranger)
    hhca.tl.wall_compartments(
        adata,
        pixels,
        factor_order,
        vessels,
        structs,
        markers["vessel_types"],
        spillover,
        markers["pericyte_radius_um"],
        structures["lattice_um"],
    )
    del pixels
    counts = adata.uns["wall_compartments"]
    log(
        f"{slide}: spaceranger bins={adata.n_obs:,}; vessel-type mapped={counts['n_mapped']:,}; "
        f"muscular-vein structs={counts['n_mv_structs']} member 2um bins={counts['n_mv_bins2um']:,}"
    )

    out = hhca.tl.marker_counts(adata, genes)
    out.insert(0, "slide", slide)
    out.to_csv(table_path + ".part", index=False)
    os.replace(table_path + ".part", table_path)
    with open(f"{args.out_dir}/{slide}.meta.json", "w") as handle:
        json.dump(
            {
                "slide": slide,
                "genes_present": out.attrs["genes_present"],
                "mpp": adata.uns["microns_per_pixel"],
                "n_sr_bins": int(adata.n_obs),
                **counts,
                "cmflag": args.flags,
            },
            handle,
            indent=1,
        )
    log(f"{slide}: wrote {table_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
