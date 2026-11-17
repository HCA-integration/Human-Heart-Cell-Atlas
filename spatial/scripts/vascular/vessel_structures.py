#!/usr/bin/env python
"""Vessel structures of one Visium HD slide from its FICTURE 2 um L3 decode.

A lattice cell is vascular when enough of its 2 um decode pixels carry a vascular Level-3 type; the structures
are the connected components (8-connectivity) of vascular lattice cells (hhca.tl.vascular_lattice,
hhca.tl.vessel_structures). Parameters: `structures` in config/vascular.yaml.

Inputs: the FICTURE pixel decode of the slide and the Level-3 factor order.
Outputs: DIR/<slide>_cells.parquet (one row per vascular 8 um lattice cell: key, gx, gy, votes, n_votes,
         dom_subtype, x_um, y_um, struct) and DIR/<slide>_structs.csv (one row per structure: size, votes,
         vote fractions, vessel_type, slide).

Usage: python vessel_structures.py --slide S --decode DECODE.tsv.gz --factor-order L3_ORDER.json --out-dir DIR
"""

import argparse
import json
import os

from hhca import thread_env

os.environ.update(thread_env(1, numba=False))

import hhca
from hhca.config import load_config
from hhca.log import get_logger

log = get_logger()


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slide", required=True, help="slide id")
    parser.add_argument("--decode", required=True, help="FICTURE pixel decode of the slide (.pixel.tsv.gz)")
    parser.add_argument("--factor-order", required=True, help="JSON list of the Level-3 types in factor order")
    parser.add_argument("--out-dir", required=True, help="output directory")
    parser.add_argument("--force", action="store_true", help="rebuild even if both outputs exist")
    return parser.parse_args()


def main():
    """Build and write the cell and structure tables of one slide."""
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cells_out = f"{args.out_dir}/{args.slide}_cells.parquet"
    structs_out = f"{args.out_dir}/{args.slide}_structs.csv"
    if os.path.exists(cells_out) and os.path.exists(structs_out) and not args.force:
        log(f"  {args.slide}: outputs exist, skipped")
        return
    params = load_config("vascular")["structures"]
    with open(args.factor_order) as handle:
        factor_order = json.load(handle)
    pixels = hhca.io.read_decode(args.decode)
    vessels = hhca.tl.vascular_lattice(
        pixels, factor_order, params["vascular_types"], params["lattice_um"], params["min_votes"]
    )
    hhca.tl.vessel_structures(vessels)
    structs = hhca.tl.structure_table(vessels, params["muscular_vsmc_fraction"], params["lattice_um"])
    log(f"  {args.slide}: {vessels.n_obs} vascular lattice cells -> {len(structs)} vessel structures")
    hhca.io.write_vessels(vessels, cells_out)
    structs.assign(slide=args.slide).to_csv(structs_out + ".part", index=False)
    os.replace(structs_out + ".part", structs_out)


if __name__ == "__main__":
    main()
