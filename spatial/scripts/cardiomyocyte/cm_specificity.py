"""Cardiomyocyte specificity of every gene from the atlas Level-3 pseudobulk (mean normcounts per type).

  cm_expressed = ventricular-CM mean normcount >= specificity.min_ventricular_mean
  cm_specific  = max(VentCM, AtrCM) / max(other Level-3 type) >= specificity.min_ratio
  cm_safe      = cm_expressed and cm_specific
(hhca.tl.cardiomyocyte_specificity; thresholds in config/cardiomyocyte.yaml)

Inputs: mean_normcounts_L3.parquet of atlas_pseudobulk.py.
Outputs: one row per gene with the two means, the top non-cardiomyocyte type, the ratio, the three flags and a
text call.

Usage: cm_specificity.py --pseudobulk mean_normcounts_L3.parquet --out snrna_cm_specificity_allgenes.csv
"""

import argparse

import pandas as pd

import hhca
from hhca.config import load_config


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="snRNA cardiomyocyte-specificity call per gene")
    parser.add_argument("--pseudobulk", required=True, help="genes x Level-3 types, mean normcounts (parquet)")
    parser.add_argument("--out", required=True, help="output csv")
    return parser.parse_args()


def main():
    """Read the pseudobulk, call the specificity and write the table."""
    args = parse_args()
    table = hhca.tl.cardiomyocyte_specificity(
        pd.read_parquet(args.pseudobulk), **load_config("cardiomyocyte")["specificity"]
    )
    table.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(table)} genes, cm_safe {int(table.cm_safe.sum())}")


if __name__ == "__main__":
    main()
