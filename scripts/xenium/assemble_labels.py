#!/usr/bin/env python
"""Assemble the CellPin label table for one Xenium cohort (the input of every downstream step).

rejection: one row per cell of the GSE290577 object (162,638), author obs columns (extract_mtx.R obs.csv),
           CellPin L3 + confidence, lineage roll-up, Level-4 lymphoid lineage / label8 (lymphoid_level4.py),
           immune flags. ILC rule (config ilc_rule): ILC leaves the immune set when its held-out F1 is below
           annotate.min_f1_reliable; `immune` applies the rule, `immune_incl_ILC` does not.
healthy:   one row per cell (26,366), centroids, CellPin L3 / lineage / L4 (+ per-class L4 reliability).

Inputs:
    --annotation  output folder of transfer_labels.py (per_cell_predictions.csv, annotation_summary.json,
                  healthy: heldout_L4_within_L3_perclass.csv)
    --config      config/xenium.yaml (annotate.min_f1_reliable, immune_L3)
    --obs, --lymphoid   rejection: obs.csv of extract_mtx.R, xenium_lymphoid8_cellpin.csv of lymphoid_level4.py
    --xenium      healthy: xenium_genes.h5ad (centroids)

Outputs:
    --out         label table (csv) and <out>_meta.json next to it

Usage: python assemble_labels.py rejection|healthy --annotation DIR --config config/xenium.yaml --out LABELS.csv
           [--obs obs.csv --lymphoid xenium_lymphoid8_cellpin.csv | --xenium xenium_genes.h5ad]
"""

import argparse
import json

import anndata as ad
import pandas as pd

from hhca.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cohort", choices=["rejection", "healthy"])
    parser.add_argument("--annotation", required=True, help="output directory of transfer_labels.py")
    parser.add_argument("--config", required=True, help="config/xenium.yaml")
    parser.add_argument("--obs", default=None, help="rejection: extract_mtx.R obs.csv")
    parser.add_argument(
        "--lymphoid", default=None, help="rejection: xenium_lymphoid8_cellpin.csv of lymphoid_level4.py"
    )
    parser.add_argument("--xenium", default=None, help="healthy: xenium_genes.h5ad (centroids)")
    parser.add_argument("--out", required=True, help="output label table (csv)")
    return parser.parse_args()


def rejection_table(labels, obs_path, lymphoid_path, immune_l3, summary, min_f1):
    """Author obs columns + CellPin labels + lymphoid Level 4 + immune flags; returns the table and its counts."""
    obs = pd.read_csv(obs_path, dtype={"biopsy_antibody_grading": str})
    assert list(obs.barcode.astype(str)) == list(labels.cell_id), "row order differs from the query object"
    table = obs.copy()  # author columns incl. cell_id and barcode
    for column in labels.columns[1:]:
        table[column] = labels[column].values
    lymphoid = pd.read_csv(lymphoid_path).set_index("cell_id")  # keyed on the query obs_names (= barcode)
    table["lymphoid_lineage"] = table.barcode.astype(str).map(lymphoid["lineage"])
    table["lymphoid8"] = table.barcode.astype(str).map(lymphoid["label8"])
    table["lymphoid8_conf"] = table.barcode.astype(str).map(lymphoid["confidence"])
    cell_id_equals_barcode = bool((table.cell_id.astype(str) == table.barcode.astype(str)).all())
    immune = set(immune_l3)
    f1_ilc = summary["per_class_ref_f1"].get("ILC", float("nan"))
    drop_ilc = bool(f1_ilc < min_f1)
    table["immune_incl_ILC"] = table.cellpin_L3.isin(immune)
    table["immune"] = table.cellpin_L3.isin(immune - ({"ILC"} if drop_ilc else set()))
    counts = {
        "cell_id_equals_barcode": cell_id_equals_barcode,
        "ILC_heldout_f1": f1_ilc,
        "ILC_support": summary["per_class_ref_support"].get("ILC"),
        "ILC_excluded_from_immune": drop_ilc,
        "n_immune": int(table.immune.sum()),
        "n_immune_incl_ILC": int(table.immune_incl_ILC.sum()),
        "n_ILC": int((table.cellpin_L3 == "ILC").sum()),
        "n_lymphoid_compartment": int(table.lymphoid8.notna().sum()),
    }
    return table, counts


def healthy_table(labels, predictions, xenium_path, annotation):
    """Centroids + CellPin labels + Level 4 with its per-class reliability; returns the table and its counts."""
    xenium = ad.read_h5ad(xenium_path, backed="r")
    assert list(xenium.obs_names.astype(str)) == list(labels.cell_id)
    table = labels.copy()
    table.insert(1, "x_centroid", xenium.obs["x_centroid"].values)
    table.insert(2, "y_centroid", xenium.obs["y_centroid"].values)
    counts = {}
    if "L4" in predictions:
        table["cellpin_L4"] = predictions["L4"].values
        table["cellpin_L4_conf"] = predictions["L4_confidence"].round(4).values
        per_class = pd.read_csv(f"{annotation}/heldout_L4_within_L3_perclass.csv")
        # an L4 name under two L3 classes: reliable only if both are
        reliable = per_class.groupby("class")["reliable"].min()
        table["cellpin_L4_reliable"] = table.cellpin_L4.map(reliable).fillna(False).astype(bool)
        counts["n_L4_reliable_cells"] = int(table.cellpin_L4_reliable.sum())
    return table, counts


def main():
    args = parse_args()
    cfg = load_config(args.config)
    with open(f"{args.annotation}/annotation_summary.json") as handle:
        summary = json.load(handle)
    predictions = pd.read_csv(f"{args.annotation}/per_cell_predictions.csv")
    min_f1 = cfg["annotate"]["min_f1_reliable"]

    labels = pd.DataFrame(
        {
            "cell_id": predictions.cell_id.astype(str),
            "cellpin_L3": predictions["fine"],
            "cellpin_L3_conf": predictions["fine_confidence"].round(4),
            "cellpin_lineage": predictions["coarse"],
            "cellpin_lineage_conf": predictions["coarse_confidence"].round(4),
        }
    )
    meta = {
        "cohort": args.cohort,
        "n_cells": int(len(labels)),
        "min_f1_reliable": min_f1,
        "heldout_accuracy": summary["reference_internal_heldout_accuracy"],
        "heldout_balanced_accuracy": summary.get("reference_internal_heldout_balanced_accuracy"),
    }
    if args.cohort == "rejection":
        table, counts = rejection_table(labels, args.obs, args.lymphoid, cfg["immune_L3"], summary, min_f1)
    else:
        table, counts = healthy_table(labels, predictions, args.xenium, args.annotation)
    meta.update(counts)

    table.to_csv(args.out, index=False)
    with open(args.out.replace(".csv", "_meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
