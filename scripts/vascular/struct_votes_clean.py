#!/usr/bin/env python
"""Fig 4 g: CMFLAG-clean vascular-subtype votes per vessel structure.

For each slide the votes of every lattice cell are recounted without the decode pixels whose 2 um bin carries
any of the four spillover flags (hhca.tl.clean_votes), and summed per structure (hhca.tl.structure_votes). The
mask changes neither the structures, nor their membership, nor their calibre.

Inputs: decode and <flags-dir>/<slide>_clean_bins.parquet per slide, the tables of vessel_structures.py.
Outputs: vessel_structs_cmflag_clean.csv (--out): slide, struct, n_bins, vote_<subtype>, vote_<subtype>_raw
         (one row per structure)

Usage: python struct_votes_clean.py --slides S ... --decodes DECODE ... --structures-dir DIR
           --flags-dir DIR --factor-order L3_ORDER.json --out vessel_structs_cmflag_clean.csv
"""

from __future__ import annotations

import argparse
import json
import os

from hhca import thread_env

os.environ.update(thread_env(1, numba=False, keep_existing=True))

import pandas as pd

import hhca
from hhca.config import load_config
from hhca.grid import spillover_keys


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slides", nargs="+", required=True, help="slide ids")
    parser.add_argument("--structures-dir", required=True, help="directory with the vessel-structure tables")
    parser.add_argument("--decodes", nargs="+", required=True, help="one decode per slide, in the order of --slides")
    parser.add_argument("--flags-dir", required=True, help="directory with <slide>_clean_bins.parquet")
    parser.add_argument("--factor-order", required=True, help="JSON list of the Level-3 types in factor order")
    parser.add_argument("--out", required=True, help="output CSV")
    return parser.parse_args()


def main():
    """Recount the votes of every slide and write one table of all structures."""
    args = parse_args()
    params = load_config("vascular")["structures"]
    with open(args.factor_order) as handle:
        factor_order = json.load(handle)
    assert len(args.decodes) == len(args.slides), "--decodes must list one decode per slide"
    tables = []
    for slide, decode in zip(args.slides, args.decodes):
        vessels = hhca.io.read_vessels(
            f"{args.structures_dir}/{slide}_cells.parquet", params["vascular_types"], factor_order
        )
        spillover = spillover_keys(f"{args.flags_dir}/{slide}_clean_bins.parquet")
        hhca.tl.clean_votes(vessels, hhca.io.read_decode(decode), spillover, params["lattice_um"])
        per_struct = hhca.tl.structure_votes(vessels)
        per_struct.insert(0, "slide", slide)
        tables.append(per_struct)
        kept = vessels.layers["clean"].sum() / vessels.X.sum()
        print(
            f"{slide:30s} {len(per_struct):6d} structures, {100 * kept:5.1f}% of vascular pixels survive CMFLAG",
            flush=True,
        )

    votes = pd.concat(tables, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    votes.to_csv(args.out, index=False)
    clean_columns = [f"vote_{name}" for name in params["vascular_types"]]
    print(
        f"\n{len(votes):,} structures, "
        f"{int((votes[clean_columns].sum(1) == 0).sum()):,} lose every vascular pixel to the spillover mask"
    )


if __name__ == "__main__":
    main()
