#!/usr/bin/env python
"""Fig 4 k: drug-class calibre selectivity on the CMFLAG-clean per-structure pseudobulk.

--variant clean (Fig 4 k) uses the CMFLAG-clean sums of struct_pb_h5ad.py; --variant raw the unmasked sums
(sensitivity analysis without the spillover mask).

The drug classes are hand-curated target-gene sets (ChEMBL targets as listed in drug2cell 0.1.2's bundled
dictionary; `drug.classes` in config/vascular.yaml). Class score per structure = mean over the class's target
genes of log1p(CP10k), library over the 11-slide common gene set (hhca.tl.drug_class_scores); structures with
library >= `drug.min_library`; OLS score ~ is_large + z(log lib) + C(slide) over large + small structures,
BH-FDR over the classes, plus the same model with z(mural_VSMC_frac) and a Mann-Whitney test
(hhca.tl.calibre_selectivity).

Inputs: struct_pb_cmflag_clean.parquet (struct_pb_h5ad.py --pool), caliber_per_structure.csv (caliber.py).
Outputs: <out-dir>/tables/drug_class_by_caliber_summary.csv, per_slide_class_caliber.csv,
         caliber_composition_context.csv, per_structure_drug_caliber.parquet

Usage: python drug_caliber.py --struct-pb struct_pb_cmflag_clean.parquet
           --caliber caliber_per_structure.csv --variant clean --out-dir DIR
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

import hhca
from hhca.config import load_config


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--variant",
        choices=["clean", "raw"],
        default="clean",
        help="use the CMFLAG-clean sums (default) or the unmasked ones",
    )
    parser.add_argument("--struct-pb", required=True, help="pooled per-structure pseudobulk of struct_pb_h5ad.py")
    parser.add_argument("--caliber", required=True, help="calibre table of all slides (caliber.py)")
    parser.add_argument("--out-dir", required=True, help="output directory")
    return parser.parse_args()


def main():
    """Score the drug classes per structure and test large against small vessels."""
    args = parse_args()
    variant = args.variant
    tables = f"{args.out_dir}/tables"
    os.makedirs(tables, exist_ok=True)
    drug = load_config("vascular")["drug"]
    classes = drug["classes"]

    structs = hhca.tl.drug_class_scores(pd.read_parquet(args.struct_pb), pd.read_csv(args.caliber), classes, variant)
    print(f"[{variant}] structures joined to the calibre table: {len(structs):,}")
    structs.to_parquet(f"{tables}/per_structure_drug_caliber.parquet")

    tested = hhca.tl.tested_structures(structs, drug["min_library"])
    print(f"[{variant}] structures with lib_common>={drug['min_library']}: {len(tested):,}")
    print(tested["caliber"].value_counts().to_string())

    summary = hhca.tl.calibre_selectivity(tested, classes)
    summary.to_csv(f"{tables}/drug_class_by_caliber_summary.csv", index=False)
    hhca.tl.drug_scores_by_section(tested, classes).to_csv(f"{tables}/per_slide_class_caliber.csv", index=False)
    hhca.tl.calibre_context(tested).to_csv(f"{tables}/caliber_composition_context.csv", index=False)

    columns = ["drug_class", "coef_large_minus_small_adj", "p_adj_LvsS", "coef_LvsS_composition_adj"]
    print(summary[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
