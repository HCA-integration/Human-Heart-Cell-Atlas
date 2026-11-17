#!/usr/bin/env python
"""Lumen-inclusive vessel calibre and calibre class of every vessel structure on the 11 slides.

Calibre per structure = 2 * sqrt(area / pi), area = max(convex-hull area of the member lattice-cell centres,
n_bins * cell area), so a ring of wall cells counts with its lumen (hhca.tl.vessel_calibre). The cut points are
quantiles of the diameters on the --berlin slides; the --ah slides are classified with the same cut points
(hhca.tl.calibre_classes). Parameters: `calibre` in config/vascular.yaml.

Inputs: the vessel-structure tables of vessel_structures.py.
Outputs: DIR/caliber_per_structure_berlin.csv (Berlin slides), DIR/ah/<slide>_vessel_caliber.csv (AH slides),
         DIR/caliber_per_structure.csv (all slides), DIR/caliber_cuts.json

Usage: python caliber.py --structures-dir DIR --factor-order L3_ORDER.json --berlin SLIDE ... --ah SLIDE ...
           --out-dir DIR
"""

import argparse
import json
import os

from hhca import thread_env

os.environ.update(thread_env(1))

import pandas as pd

import hhca
from hhca.config import load_config

COMMON = [
    "slide",
    "struct",
    "n_bins",
    "x_um",
    "y_um",
    "mural_VSMC_frac",
    "hull_area_um2",
    "lumen_equiv_diam_um",
    "vessel_class",
]


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--structures-dir", required=True, help="directory with <slide>_cells.parquet")
    parser.add_argument("--factor-order", required=True, help="JSON list of the Level-3 types in factor order")
    parser.add_argument("--berlin", nargs="+", required=True, help="slides that define the cut points")
    parser.add_argument("--ah", nargs="*", default=[], help="slides classified with the Berlin cut points")
    parser.add_argument("--out-dir", required=True, help="output directory")
    return parser.parse_args()


def main():
    """Calibre of the Berlin structures, cut points, AH structures, and the table of all slides."""
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = load_config("vascular")
    structures, large, small = cfg["structures"], cfg["calibre"]["large"], cfg["calibre"]["small"]
    with open(args.factor_order) as handle:
        factor_order = json.load(handle)

    def calibre_of(slide, filled_diameter):
        vessels = hhca.io.read_vessels(
            f"{args.structures_dir}/{slide}_cells.parquet", structures["vascular_types"], factor_order
        )
        table = hhca.tl.vessel_calibre(vessels, structures["lattice_um"], filled_diameter)
        table.insert(0, "slide", slide)
        return table

    def classify(table, d_large, d_small):
        return hhca.tl.calibre_classes(table, d_large, d_small, large["min_vsmc_fraction"], small["max_vsmc_fraction"])

    structs = pd.concat([calibre_of(slide, True) for slide in args.berlin], ignore_index=True)
    d90 = float(structs["lumen_equiv_diam_um"].quantile(large["quantile"]))
    d33 = float(structs["lumen_equiv_diam_um"].quantile(small["quantile"]))
    structs = classify(structs, d90, d33)
    structs.to_csv(f"{args.out_dir}/caliber_per_structure_berlin.csv", index=False)
    print(
        "calibre classes:",
        structs.vessel_class.value_counts().to_dict(),
        f"| cuts d90={d90:.1f} d33={d33:.1f} um",
        flush=True,
    )

    # The AH slides are classified, and the table of all slides is built, from the Berlin table as read back from
    # its CSV, so every value in the table of all slides has the precision of the written files.
    berlin = pd.read_csv(f"{args.out_dir}/caliber_per_structure_berlin.csv")
    d90_csv = float(berlin["lumen_equiv_diam_um"].quantile(large["quantile"]))
    d33_csv = float(berlin["lumen_equiv_diam_um"].quantile(small["quantile"]))
    parts = [berlin[COMMON]]
    for slide in args.ah:
        ah_structs = classify(calibre_of(slide, False), d90_csv, d33_csv)
        os.makedirs(f"{args.out_dir}/ah", exist_ok=True)
        ah_structs.to_csv(f"{args.out_dir}/ah/{slide}_vessel_caliber.csv", index=False)
        ah_structs = pd.read_csv(f"{args.out_dir}/ah/{slide}_vessel_caliber.csv")
        print(f"AH {slide}: {len(ah_structs):,} structures, classes {ah_structs.vessel_class.value_counts().to_dict()}")
        parts.append(ah_structs[COMMON])
    all_slides = pd.concat(parts, ignore_index=True)
    assert all_slides[COMMON].notna().all().all(), "NA in the calibre table of all slides"
    all_slides.to_csv(f"{args.out_dir}/caliber_per_structure.csv", index=False)
    with open(f"{args.out_dir}/caliber_cuts.json", "w") as handle:
        json.dump(
            {"D90_um": d90, "D33_um": d33, "D90_um_csv": d90_csv, "D33_um_csv": d33_csv, "cut_slides": args.berlin},
            handle,
            indent=1,
        )
    print(f"all slides: {len(all_slides):,} structures on {all_slides.slide.nunique()} slides")
    print("  classes:", all_slides.vessel_class.value_counts().to_dict())


if __name__ == "__main__":
    main()
