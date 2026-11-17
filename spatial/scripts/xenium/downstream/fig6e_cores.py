"""Fig 6 e at TMA-core level (drawn by figures/Fig6/panel_e.py).

Unit = TMA core; grade = per-core biopsy_cellular_grading; foci = fraction of immune cells inside DBSCAN foci
(hhca.tl.core_grades); Spearman correlation.

Inputs:
    obs.csv, cells_geom.csv of the extraction folder (config inputs.extracted)
    labels/xenium_atlas_labels.csv, labels/xenium_lymphoid8_annotation.csv (label_tables.py)
    tables/geom_features_L4.csv (core_geometry.py)

Outputs (<out>/):
    fig6e_cores.csv, fig6e_cores_stats.json

Usage: python scripts/xenium/downstream/fig6e_cores.py [--geom CSV]
"""

import argparse
import json

import pandas as pd
from scipy import stats

import downstream_common as common
import hhca


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common.add_label_arguments(parser)
    parser.add_argument("--geom", help="tables/geom_features_L4.csv")
    parser.add_argument("--out", help="output folder (default: under config outputs.root)")
    return parser.parse_args()


def main():
    cfg = common.load_config()
    args = parse_args()
    tables = common.out_dir(cfg, "tables")
    out = args.out or common.out_dir(cfg, "stats")

    adata = common.load_cohort(cfg, args)
    geometry = pd.read_csv(args.geom or f"{tables}/geom_features_L4.csv").set_index("Sample")
    cores = hhca.tl.core_grades(adata, geometry)
    rho, p = stats.spearmanr(cores.grade, cores.ff)
    print(f"cores n={len(cores)} | Spearman rho={rho:.3f} p={p:.3g}")
    print("by rejection type:", cores.rej.value_counts(dropna=False).to_dict())
    cores.to_csv(f"{out}/fig6e_cores.csv")
    summary = {
        "n": len(cores),
        "rho": rho,
        "p": p,
        "medians": cores.groupby("grade").ff.median().to_dict(),
        "by_rej": cores.rej.value_counts(dropna=False).to_dict(),
    }
    with open(f"{out}/fig6e_cores_stats.json", "w") as handle:
        json.dump(summary, handle, indent=1, default=str)


if __name__ == "__main__":
    main()
