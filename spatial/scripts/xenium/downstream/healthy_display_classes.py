"""ED6 i display classes for the healthy Xenium section (CellPin + HHCA labels; drawn by figures/ED6/panel_i.py).

  * cardiomyocytes pooled into one class (hMulti has no chamber markers),
  * endothelial subtypes shown only for EC-marker-positive cells. Gate (config ed6i; hhca.tl.endothelial_gate): a
    cell labelled with an EC subtype passes when its raw transcripts summed over the pan-EC genes (PECAM1, VWF,
    EGFL7, ERG, RAMP2; plus PROX1, LYVE1, MMRN1 for lymphatic EC) are >= ec_min_counts AND make up >=
    ec_min_fraction of the cell's panel transcripts; otherwise it is drawn as "EC label, marker-negative".
The same marker count is reported for non-EC cells (fraction of cardiomyocytes / fibroblasts that would pass) as the
specificity of the gate.

Inputs:
    results/xenium/healthy/xenium_cellpin_labels.csv (assemble_labels.py; config inputs.healthy.cellpin_labels)
    10x cell_feature_matrix.h5 of the healthy section (config inputs.healthy.counts_h5)

Outputs:
    results/xenium/downstream/healthy/ed6i_cells.csv, ed6i_counts.json

Usage: python scripts/xenium/downstream/healthy_display_classes.py [--labels CSV] [--counts-h5 H5] [--out DIR]
"""

import argparse
import json

import pandas as pd
import scanpy as sc

import downstream_common as common
import hhca

# non-endothelial classes for which the share of cells passing the gate is reported
SPECIFICITY_CLASSES = [
    "Atrial Cardiomyocytes",
    "Ventricular Cardiomyocytes",
    "Fibroblasts",
    "Vascular Smooth Muscle cells",
    "Pericytes",
]


def parse_args(cfg):
    """Command-line arguments; the defaults come from config/xenium_downstream.yaml."""
    settings = cfg["ed6i"]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--labels",
        default=common.repo_path(cfg["inputs"]["healthy"]["cellpin_labels"]),
        help="CellPin label table of the healthy section",
    )
    parser.add_argument("--counts-h5", default=cfg["inputs"]["healthy"]["counts_h5"], help="10x cell_feature_matrix.h5")
    parser.add_argument(
        "--min-counts", type=int, default=settings["ec_min_counts"], help="EC gate: fewest marker transcripts"
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=settings["ec_min_fraction"],
        help="EC gate: smallest marker share of the cell's transcripts",
    )
    parser.add_argument("--out", default=common.out_dir(cfg, "healthy"), help="output folder")
    return parser.parse_args()


def main():
    cfg = common.load_config()
    settings = cfg["ed6i"]
    args = parse_args(cfg)

    labels = pd.read_csv(args.labels)
    adata = sc.read_10x_h5(args.counts_h5)
    adata.var_names_make_unique()
    assert (adata.obs_names == labels.cell_id.values).all(), "cell order differs from the label table"
    l3 = labels.cellpin_L3.astype(str)
    hhca.tl.endothelial_gate(
        adata,
        l3.values,
        settings["ec_classes"],
        settings["ec_markers"],
        settings["lymphatic_markers"],
        min_counts=args.min_counts,
        min_fraction=args.min_fraction,
        pool=settings["pool_cardiomyocytes"],
        negative_label=settings["ec_negative_label"],
    )
    obs = adata.obs
    marker, gate, display = obs.ec_marker_counts.values, obs.ec_gate.values, obs.display.values
    is_ec = l3.isin(settings["ec_classes"]).values

    cells = pd.DataFrame(
        {
            "cell_id": labels.cell_id,
            "x": labels.x_centroid,
            "y": labels.y_centroid,
            "cellpin_L3": l3,
            "display": display,
            "ec_marker_counts": marker,
            "total_counts": obs.total_counts.values,
            "ec_gate_pass": obs.ec_gate_pass.values,
        }
    )
    cells.to_csv(f"{args.out}/ed6i_cells.csv", index=False)

    per_ec_class = {}
    for name in settings["ec_classes"]:
        in_class = (l3 == name).values
        per_ec_class[name] = {
            "labelled": int(in_class.sum()),
            "pass": int((in_class & gate).sum()),
            "fail": int((in_class & ~gate).sum()),
        }
    specificity = {}
    for name in SPECIFICITY_CLASSES:
        in_class = (l3 == name).values
        specificity[name] = {
            "n": int(in_class.sum()),
            "frac_would_pass": float(gate[in_class].mean()),
            "frac_would_pass_counts_only": float((marker[in_class] >= args.min_counts).mean()),
        }
    summary = {
        "n_cells": len(cells),
        "markers": adata.uns["endothelial_gate"]["markers"],
        "lymphatic_markers": adata.uns["endothelial_gate"]["lymphatic_markers"],
        "min_counts": args.min_counts,
        "min_fraction": args.min_fraction,
        "ec_labelled": int(is_ec.sum()),
        "ec_pass": int((is_ec & gate).sum()),
        "ec_fail": int((is_ec & ~gate).sum()),
        "per_ec_class": per_ec_class,
        "non_ec_frac_would_pass": specificity,
        "display_counts": {k: int(v) for k, v in pd.Series(display).value_counts().items()},
    }
    with open(f"{args.out}/ed6i_counts.json", "w") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
