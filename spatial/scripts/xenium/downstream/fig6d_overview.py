"""Fig 6 d (pre-treatment) and ED9 a (post-treatment) core overview.

Unit = TMA core; cores of patients with a rejection episode at the given timing, >= 25 immune cells, sorted by
rejection type, per-core grade, immune fraction (hhca.tl.core_overview). Composition classes: lymphoid Level 4 (no
Treg) + Level 3 (config classes.composition; ILC not "Other"). "Resolution" strip = GEO resolution_broad (the
patient outcome, as Fig 6 f-g). Drawn by figures/Fig6/panel_d.py and figures/ED9/panel_a.py.

Inputs:
    tables/amr_axis_features_ATLAS.csv (core_features.py)
    labels/xenium_atlas_labels.csv, labels/xenium_lymphoid8_annotation.csv (label_tables.py)
    obs.csv, cells_geom.csv of the extraction folder (config inputs.extracted)
    GEO clinical metadata table (config inputs.metadata)

Outputs:
    <out>/fig6d_overview_<timing>.csv (one row per drawn core, in drawing order)

Usage: python scripts/xenium/downstream/fig6d_overview.py --timing pre|post
"""

import argparse

import pandas as pd

import downstream_common as common
import hhca


def parse_args(cfg):
    """Command-line arguments; the defaults come from config/xenium_downstream.yaml."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--timing", default="pre", choices=["pre", "post"], help="biopsy timing relative to treatment")
    common.add_label_arguments(parser)
    parser.add_argument("--features", help="tables/amr_axis_features_ATLAS.csv")
    parser.add_argument("--meta", default=cfg["inputs"]["metadata"], help="GEO clinical metadata table")
    parser.add_argument("--out", help="output folder (default: under config outputs.root)")
    return parser.parse_args()


def main():
    cfg = common.load_config()
    args = parse_args(cfg)
    tables = common.out_dir(cfg, "tables")
    out = args.out or common.out_dir(cfg, "stats")
    timing = args.timing

    adata = common.load_cohort(cfg, args)
    common.classify(adata, cfg, "composition", key_added="cat")
    features = pd.read_csv(args.features or f"{tables}/amr_axis_features_ATLAS.csv")
    resolution = pd.read_csv(args.meta, sep="\t").set_index("Sample").resolution_broad.dropna().to_dict()
    overview = hhca.tl.core_overview(
        adata,
        features,
        resolution,
        timing,
        cfg["classes"]["composition_order"],
        min_immune=cfg["params"]["fig6d_min_immune"],
    )
    print(
        f"{timing}: cores {len(overview)} | per patient-type: {overview.prt.value_counts().to_dict()} | resolution "
        f"(resolution_broad) {overview.res.value_counts().to_dict()}"
    )
    overview.to_csv(f"{out}/fig6d_overview_{timing}.csv", index=False)
    print("wrote", f"{out}/fig6d_overview_{timing}.csv")


if __name__ == "__main__":
    main()
