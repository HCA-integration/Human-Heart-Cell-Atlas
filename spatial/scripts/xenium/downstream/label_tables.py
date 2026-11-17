"""CellPin rejection labels -> the two label tables the downstream scripts read.

From results/xenium/rejection/xenium_cellpin_labels.csv this writes
  xenium_atlas_labels.csv          (cell_id, Sample, author metadata, atlas_L3, atlas_immune)
  xenium_lymphoid8_annotation.csv  (cell_id, Sample, atlas_L3, lineage, label8, confidence, is_prolif)
with (hhca.tl.atlas_labels)
  atlas_L3     = cellpin_L3, atrial + ventricular cardiomyocytes pooled into "Cardiomyocytes"
  atlas_immune = `immune` (10 immune L3 classes without ILC; ILC held-out F1 0.33 < 0.5)
  label8       = lymphoid8 (6 classes: CD4 T, CD8 T, NK, each +/- proliferating; no Treg)
Author metadata columns are taken from the extraction of GSE290577_heart_spatial.rds (Aug 2025; `lineage` reads
Stromal).

Inputs:
    results/xenium/rejection/xenium_cellpin_labels.csv (assemble_labels.py; config inputs.rejection.cellpin_labels)
    obs.csv of the extraction folder (config inputs.extracted)

Outputs (<out>/):
    xenium_atlas_labels.csv, xenium_lymphoid8_annotation.csv

Usage: python scripts/xenium/downstream/label_tables.py [--cellpin PATH] [--out DIR]
"""

import argparse

import anndata as ad
import pandas as pd

import downstream_common as common
import hhca

META_COLUMNS = [
    "cell_id",
    "Sample",
    "patient_id",
    "patient_biopsy",
    "ct_second_pass",
    "lineage",
    "biopsy_rejection_type",
    "patient_rejection_type",
    "biopsy_cellular_grading",
    "biopsy_antibody_grading",
    "biopsy_timing",
    "slide",
    "age_at_biopsy",
    "sex",
    "total_counts",
    "nCount_Xenium",
    "nFeature_Xenium",
    "barcode",
]


def parse_args(cfg):
    """Command-line arguments; the defaults come from config/xenium_downstream.yaml."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--cellpin",
        default=common.repo_path(cfg["inputs"]["rejection"]["cellpin_labels"]),
        help="CellPin label table of the rejection cohort",
    )
    parser.add_argument(
        "--obs",
        default=common.extracted(cfg) + "/obs.csv",
        help="author metadata of the GEO object (Aug 2025: lineage reads Stromal)",
    )
    parser.add_argument("--out", default=common.out_dir(cfg, "labels"), help="output folder")
    return parser.parse_args()


def main():
    cfg = common.load_config()
    settings = cfg["labels"]
    args = parse_args(cfg)

    cellpin = pd.read_csv(args.cellpin)
    obs = pd.read_csv(args.obs).set_index("barcode", drop=False).reindex(cellpin.barcode)
    assert obs.index.notna().all() and len(obs) == len(cellpin)
    obs.index.name = None
    adata = ad.AnnData(obs=obs)
    hhca.tl.atlas_labels(
        adata,
        cellpin,
        pool=settings.get("pool_cardiomyocytes", []),
        exclude=settings.get("exclude_l3", []),
        lymphoid=settings["lymphoid"],
    )

    labels = adata.obs[[*META_COLUMNS, "atlas_L3", "atlas_immune"]]
    labels.to_csv(f"{args.out}/xenium_atlas_labels.csv", index=False)
    level4 = adata.obs.loc[
        adata.obs.label8.notna(), ["cell_id", "Sample", "atlas_L3", "lymphoid_lineage", "label8", "label8_confidence"]
    ]
    level4 = level4.rename(columns={"lymphoid_lineage": "lineage", "label8_confidence": "confidence"})
    level4["is_prolif"] = level4.label8.str.startswith("Proliferating")
    level4.to_csv(f"{args.out}/xenium_lymphoid8_annotation.csv", index=False)

    print(
        f"cells {len(labels):,} | immune {int(labels.atlas_immune.sum()):,} | lymphoid L4 {len(level4):,} | "
        f"ILC (excluded) {int((labels.atlas_L3 == 'ILC').sum())} | "
        f"cardiomyocytes pooled {int((labels.atlas_L3 == 'Cardiomyocytes').sum()):,}"
    )
    print(level4.label8.value_counts().to_string())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
