#!/usr/bin/env python
"""Build resources/vascular/common_panel_symbols.json, the gene panel shared by all Visium HD slides.

The panel is the sorted intersection of the feature names of the Space Ranger square_002um matrices
(18,082 names: the panel of the 8 Berlin slides, a subset of the AH panel).

Inputs: filtered_feature_bc_matrix.h5 of each square_002um directory.
Outputs: the JSON list of the common feature names (--out).

Usage: python common_panel.py --out common_panel_symbols.json SQUARE_002UM_DIR ...
"""

import argparse
import json

import h5py


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="output JSON")
    parser.add_argument("dirs", nargs="+", help="square_002um directories, one per slide")
    return parser.parse_args()


def main():
    """Intersect the feature names of the matrices and write the sorted list."""
    args = parse_args()
    name_sets = []
    for directory in args.dirs:
        with h5py.File(f"{directory}/filtered_feature_bc_matrix.h5", "r") as matrix:
            name_sets.append(set(matrix["matrix/features/name"][:].astype(str)))
    common = sorted(set.intersection(*name_sets))
    with open(args.out, "w") as handle:
        json.dump(common, handle)
    print(f"{len(common)} common feature names from {len(args.dirs)} matrices -> {args.out}")


if __name__ == "__main__":
    main()
