#!/usr/bin/env python
"""Fig 4 h, pooling: compartment x gene matrix over all slides plus connexin / tight-junction detection tables.

Pooling (hhca.tl.pool_markers): per compartment, cp10k = 1e4 * sum(marker counts) / sum(total counts) over
slides, pct = bins with >= 1 count / bins, logcp = log1p(cp10k). Detected (hhca.tl.marker_detection) = enough
counts over the compartments and a high enough pct in at least one (`markers.detected` in config/vascular.yaml).

Inputs: the per-slide tables of marker_aggregate.py.
Outputs: connexin_tj_pooled_matrix.csv (Fig 4 h), connexin_detection.csv,
         tight_junction_detection.csv, add_lists.json

Usage: python marker_pool.py --ckpt-dir DIR --out-dir DIR
"""

import argparse
import glob
import json
import os

import pandas as pd

import hhca
from hhca.config import load_config

# members of the two families drawn in Fig 4 h
IN_FIGURE_CONNEXIN = ["GJA5", "GJA4", "GJA1", "GJC1"]
IN_FIGURE_TJ = ["CLDN5", "CDH5", "TJP1", "OCLN", "F11R"]


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ckpt-dir", required=True, help="directory with the per-slide tables of marker_aggregate.py")
    parser.add_argument("--out-dir", required=True, help="output directory")
    return parser.parse_args()


def main():
    """Pool the per-slide tables and write the matrix, the detection tables and the gene lists."""
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    markers = load_config("vascular")["markers"]
    files = sorted(glob.glob(f"{args.ckpt_dir}/*.csv"))
    files = [path for path in files if not os.path.basename(path).startswith("._")]
    print(f"pooling {len(files)} slide tables")
    per_slide = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)

    pooled = hhca.tl.pool_markers(per_slide)
    pooled.to_csv(f"{args.out_dir}/connexin_tj_pooled_matrix.csv", index=False)

    pd.set_option("display.width", 200)
    detection = {}
    for family in ("connexins", "tight_junctions"):
        detection[family] = hhca.tl.marker_detection(pooled, markers[family], **markers["detected"])
        print(f"\n{family}")
        print(detection[family].to_string(index=False))
    connexins, tight_junctions = detection["connexins"], detection["tight_junctions"]
    connexins.to_csv(f"{args.out_dir}/connexin_detection.csv", index=False)
    tight_junctions.to_csv(f"{args.out_dir}/tight_junction_detection.csv", index=False)
    detected_connexins = connexins[connexins.detected].gene.tolist()
    detected_tj = tight_junctions[tight_junctions.detected].gene.tolist()
    add_connexins = [gene for gene in detected_connexins if gene not in IN_FIGURE_CONNEXIN]
    add_tj = [gene for gene in detected_tj if gene not in IN_FIGURE_TJ]
    print("\nconnexins detected but not in the figure:", add_connexins)
    print("tight-junction genes detected but not in the figure:", add_tj)
    with open(f"{args.out_dir}/add_lists.json", "w") as handle:
        json.dump(
            {
                "add_connexins": add_connexins,
                "add_tj": add_tj,
                "det_connexins": detected_connexins,
                "det_tj": detected_tj,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
