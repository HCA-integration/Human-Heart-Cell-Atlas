"""Per-core geometry features on the Level-4 lymphoid labels (no Treg classes, ILC not immune).

Immune  = atlas_immune: the lymphoid compartment + Macrophages, Monocytes, Dendritic cells, B cells, Plasma cells,
          Mast cells
T cells = the lymphoid Level-4 T classes (config labels.tcell; NK excluded)
Foci    = DBSCAN(eps 30 um, min_samples 5) on immune-cell centroids, per core (hhca.tl.immune_foci).

Inputs:
    labels/xenium_atlas_labels.csv, labels/xenium_lymphoid8_annotation.csv (label_tables.py)
    obs.csv, cells_geom.csv of the extraction folder (config inputs.extracted)

Outputs:
    <out>/geom_features_L4.csv (one row per core) -> Fig 6 e, f, g, l

Usage: python scripts/xenium/downstream/core_geometry.py
"""

import argparse
import os

import downstream_common as common
import hhca


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common.add_label_arguments(parser)
    parser.add_argument("--out", help="output folder (default: under config outputs.root)")
    return parser.parse_args()


def main():
    cfg = common.load_config()
    args = parse_args()
    out = args.out or common.out_dir(cfg, "tables")
    dbscan = cfg["params"]["dbscan"]

    print("loading ...")
    adata = common.load_cohort(cfg, args)
    common.classify(adata, cfg, "immune")
    obs = adata.obs
    is_t = obs.ct.isin(cfg["labels"]["tcell"])
    print(
        f"cells {adata.n_obs:,} | immune {int(obs.atlas_immune.sum()):,} | T {int(is_t.sum()):,} "
        f"| cores {obs.Sample.nunique()}"
    )
    print(obs.loc[obs.atlas_immune, "ct"].value_counts().to_string())

    features = hhca.tl.immune_foci(adata, "atlas_immune", eps=dbscan["eps_um"], min_samples=dbscan["min_samples"])
    frac_tcell = is_t.groupby(obs.Sample).sum().loc[features.index] / features.n_total
    features.insert(3, "frac_Tcell", frac_tcell)
    features.reset_index().to_csv(os.path.join(out, "geom_features_L4.csv"), index=False)
    print(f"\nwrote {out}/geom_features_L4.csv  ({len(features)} cores)")


if __name__ == "__main__":
    main()
