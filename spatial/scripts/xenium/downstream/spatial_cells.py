"""Per-cell display classes with centroids for the rejection-cohort spatial panels (Fig 6 c, ED6 j).

Fig 6 c: lymphoid Level-4 classes + six other immune L3 classes (config classes.slide_immune; the slide with more
cores); ED6 j: immune lineages (config classes.lineage) on two cores, per-core grade from the GEO metadata. The
classes are assigned by hhca.tl.spatial_classes; drawn by figures/Fig6/panel_c.py and figures/ED6/panel_j.py.

Inputs:
    labels/xenium_atlas_labels.csv, labels/xenium_lymphoid8_annotation.csv (label_tables.py)
    obs.csv, cells_geom.csv of the extraction folder (config inputs.extracted)

Outputs (<out>/):
    spatial_cells.csv.gz (cell_id, Sample, slide, x, y, core_grade, fig6c, ed6j), spatial_counts.json

Usage: python scripts/xenium/downstream/spatial_cells.py
"""

import argparse
import json

import pandas as pd

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

    adata = common.load_cohort(cfg, args)
    cells = hhca.tl.spatial_classes(
        adata,
        cfg["classes"]["slide_immune"],
        cfg["labels"]["lymphoid"],
        cfg["classes"]["lineage"],
        exclude=cfg["labels"].get("exclude_l3", []),
    )
    cells.to_csv(f"{out}/spatial_cells.csv.gz", index=False)
    slide = cells.groupby("slide").Sample.nunique().idxmax()
    slide_cells = cells[cells.slide == slide]
    candidates = pd.DataFrame(
        {
            "grade": slide_cells.groupby("Sample").core_grade.first(),
            "nimm": slide_cells.groupby("Sample").fig6c.apply(lambda s: s.notna().sum()),
        }
    ).dropna()
    zoom_by_rule = candidates.sort_values(["grade", "nimm"], ascending=False).index[0]
    summary = {
        "slide": slide,
        "zoom_by_rule": zoom_by_rule,
        "zoom_config": cfg["figures"]["fig6c_zoom_core"],
        "fig6c_counts_slide": {k: int(v) for k, v in slide_cells.fig6c.value_counts().items()},
        "fig6c_counts_all": {k: int(v) for k, v in cells.fig6c.value_counts().items()},
        "ed6j": {
            core: {
                "n": int((cells.Sample == core).sum()),
                "immune_pct": float(100 * (cells.loc[cells.Sample == core, "ed6j"] != "structural / other").mean()),
                "grade": float(cells.loc[cells.Sample == core, "core_grade"].iloc[0]),
            }
            for core in cfg["figures"]["ed6j_cores"]
        },
    }
    with open(f"{out}/spatial_counts.json", "w") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
