#!/usr/bin/env python
"""Cell-type composition of the endocardial left ventricle across modalities.

Compares the composition of LV-endo tissue from
  * standard Visium (HHH2-LV-endo-1 and -2): mean per-spot proportion of every deconvolution method and NNLS
  * Visium HD: area fraction of the 2 um FICTURE decode of the two LV-endo sections (areafraction_L3.py)
  * Visium HD: nucleus counts, pooled over the ventricular sections (no endo-specific count)
  * snRNA-seq: ventricular composition of the single-nucleus reference
and ranks the methods by their agreement with the HD area fraction (method_agreement.csv).

Inputs:  per-spot proportions <outputs>/<method>/<slide>.csv.gz and <nnls-dir>/<slide>_L3_deconv.csv.gz, the HD
         area-fraction table, the nucleus-count and snRNA composition tables, the leaderboard and region table
         of the ground-truth simulation (score_gt.py), the Level-3 factor order
Outputs: <out>/tables/endo_LV_composition.csv (compartments, %), endo_LV_composition_L3.csv,
         method_agreement.csv, summary.json

Usage: build_endo_LV_composition.py --outputs DIR --nnls-dir DIR --areafraction CSV --units CSV --sn CSV
           --leaderboard CSV --region-effect CSV --factor-order JSON --out DIR
"""

import os

from hhca import thread_env

os.environ.update(thread_env(1, numba=False))

import argparse
import json

import numpy as np
import pandas as pd

SV_ENDO_SLIDES = ["HHH2-LV-endo-1", "HHH2-LV-endo-2"]  # standard Visium LV-endo
HD_ENDO_SLIDES = ["VS_CA_H0054LV_henricke_endo", "VS_CA_H0063LV_henricke_endo"]
BENCH_METHODS = ["spotiphy", "cell2location", "RCTD", "SPOTlight", "stereoscope", "ucaspatial"]
PRIMARY = "spotiphy"  # most accurate on the ground-truth simulation
IMMUNE_CLEAN_ALT = "cell2location"

# Level-3 type -> compartment (the snRNA reference is annotated at this coarser level)
L3_TO_COMP = {
    "Ventricular Cardiomyocytes": "Cardiomyocytes",
    "Atrial Cardiomyocytes": "Cardiomyocytes",
    "Fibroblasts": "Fibroblasts",
    "Capillary Endothelial cells": "Vascular EC",
    "Arterial Endothelial cells": "Vascular EC",
    "Venous Endothelial cells": "Vascular EC",
    "Endocardial Endothelial Cells": "Endocardial EC",
    "Lymphatic Endothelial Cells": "Lymphatic EC",
    "Pericytes": "Pericytes",
    "Vascular Smooth Muscle cells": "VSMC",
    "Macrophages": "Myeloid",
    "Monocytes": "Myeloid",
    "Dendritic cells": "Myeloid",
    "T cells": "Lymphoid",
    "B cells": "Lymphoid",
    "NK cells": "Lymphoid",
    "ILC": "Lymphoid",
    "Plasma Cells": "Lymphoid",
    "Lymphoid Cells (Proliferating)": "Lymphoid",
    "Mast cells": "Mast",
    "Neural Cells": "Neural",
    "Adipocytes": "Adipocytes",
    "Epicardium": "Epicardium",
}
# snRNA reference label -> compartment
SN_TO_COMP = {
    "VentricularCardiomyocytes": "Cardiomyocytes",
    "CardiacConductionSystem": "Cardiomyocytes",  # conduction cells are specialised cardiomyocytes (0.06 %)
    "Fibroblasts": "Fibroblasts",
    "ValveFibroblasts": "Fibroblasts",
    "MyeloidCells": "Myeloid",
    "EndothelialCells": "Vascular EC",
    "Pericytes": "Pericytes",
    "EndocardialCells": "Endocardial EC",
    "LymphoidCellsNonB": "Lymphoid",
    "LymphoidCellsB": "Lymphoid",
    "NeuralCells": "Neural",
    "MastCells": "Mast",
    "SmoothMuscleCells": "VSMC",
    "Adipocytes": "Adipocytes",
    "LymphaticEndothelialCells": "Lymphatic EC",
    "EpicardialCells": "Epicardium",
}
# compartments by descending abundance in the myocardium
COMP_ORDER = [
    "Cardiomyocytes",
    "Fibroblasts",
    "Vascular EC",
    "Pericytes",
    "VSMC",
    "Myeloid",
    "Lymphoid",
    "Lymphatic EC",
    "Endocardial EC",
    "Neural",
    "Adipocytes",
    "Mast",
    "Epicardium",
]


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outputs", required=True, help="directory with <method>/<slide>.csv.gz")
    parser.add_argument("--nnls-dir", required=True, help="directory with the NNLS tables <slide>_L3_deconv.csv.gz")
    parser.add_argument(
        "--areafraction", required=True, help="HD area fraction per slide (lv_areafraction_L3_perslide.csv)"
    )
    parser.add_argument("--units", required=True, help="HD nucleus counts (lv_composition_units_pooled.csv)")
    parser.add_argument(
        "--sn", required=True, help="snRNA ventricular composition (hca_ref_composition_ventricular.csv)"
    )
    parser.add_argument(
        "--leaderboard", required=True, help="leaderboard of the ground-truth simulation (leaderboard_hhca.csv)"
    )
    parser.add_argument(
        "--region-effect", required=True, help="accuracy by region of the simulation (region_effect_hhca.csv)"
    )
    parser.add_argument("--factor-order", required=True, help="json list of the Level-3 cell types")
    parser.add_argument("--out", required=True, help="output directory; tables go to <out>/tables/")
    return parser.parse_args()


def to_comp(series_l3):
    """Sum a Level-3 fraction series into compartments."""
    out = dict.fromkeys(COMP_ORDER, 0.0)
    for cell_type, value in series_l3.items():
        compartment = L3_TO_COMP.get(cell_type)
        if compartment is not None:
            out[compartment] += float(value)
    return pd.Series(out)[COMP_ORDER]


def load_method_endo(method, l3_types, outputs_dir, nnls_dir):
    """Mean per-spot proportion of one method over the spots of both LV-endo slides."""
    frames = []
    for slide in SV_ENDO_SLIDES:
        if method == "NNLS":
            path = f"{nnls_dir}/{slide}_L3_deconv.csv.gz"
        else:
            path = f"{outputs_dir}/{method}/{slide}.csv.gz"
        if not os.path.exists(path):
            print(f"no output of {method} for {slide}")
            continue
        table = pd.read_csv(path)
        props = table[[cell_type for cell_type in l3_types if cell_type in table.columns]].copy()
        for cell_type in l3_types:
            if cell_type not in props.columns:
                props[cell_type] = 0.0
        props = props[l3_types].clip(lower=0)
        row_sums = props.sum(axis=1).replace(0, np.nan)
        props = props.div(row_sums, axis=0).dropna(how="all")
        frames.append(props)
    if not frames:
        return None
    pooled = pd.concat(frames, axis=0, ignore_index=True)
    return pooled.mean(axis=0)


def load_visium_methods(l3_types, outputs_dir, nnls_dir):
    """Standard Visium: Level-3 fractions of the LV-endo slides per method (method -> series)."""
    fractions = {}
    for method in ["NNLS"] + BENCH_METHODS:
        mean_prop = load_method_endo(method, l3_types, outputs_dir, nnls_dir)
        if mean_prop is not None:
            fractions[method] = mean_prop
            print(
                f"standard Visium {method:>14s}  CM={mean_prop['Ventricular Cardiomyocytes']:.3f}  "
                f"n_types={(mean_prop > 0).sum()}"
            )
    return fractions


def load_hd_area_fraction(path, l3_types):
    """Visium HD: area fraction of the LV-endo sections, as fractions that sum to 1."""
    area = pd.read_csv(path).set_index("L3_type")
    fractions = area[HD_ENDO_SLIDES].mean(axis=1) / 100.0
    fractions = fractions.reindex(l3_types).fillna(0.0)
    return fractions / fractions.sum()


def load_hd_nucleus_counts(path, l3_types):
    """Visium HD: nucleus counts pooled over the ventricular sections, as fractions that sum to 1."""
    units = pd.read_csv(path).set_index("L3_type")
    fractions = (units["nucleus_count_pct_pooled"] / 100.0).reindex(l3_types).fillna(0.0)
    return fractions / fractions.sum()


def load_snrna_compartments(path):
    """snRNA-seq: ventricular composition by compartment, as fractions that sum to 1."""
    table = pd.read_csv(path)
    fractions = dict.fromkeys(COMP_ORDER, 0.0)
    for _, row in table.iterrows():
        compartment = SN_TO_COMP.get(row["cell_type"])
        if compartment is not None:
            fractions[compartment] += float(row["pct"]) / 100.0
    fractions = pd.Series(fractions)[COMP_ORDER]
    return fractions / fractions.sum()


def jsd(p, q):
    """Jensen-Shannon divergence (log base 2) of two composition vectors."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = p / p.sum()
    q = q / q.sum()
    mixture = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, mixture) + 0.5 * kl(q, mixture)


def rmse(p, q):
    """Root mean squared difference of two vectors."""
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(q, float)) ** 2)))


def method_agreement(comp_tbl, methods, leaderboard):
    """Agreement of each method with the HD area fraction and with snRNA-seq, sorted by RMSE against the former."""
    rows = []
    hd_area = comp_tbl["HD_area_fraction"].values
    sn_lv = comp_tbl["sn_LV"].values
    for method in methods:
        composition = comp_tbl[f"stdVisium_{method}"].values
        sim_rmse = leaderboard.loc[method, "rmse"] if method in leaderboard.index else np.nan
        sim_jsd = leaderboard.loc[method, "jsd"] if method in leaderboard.index else np.nan
        rows.append(
            {
                "method": method,
                "rmse_vs_HDarea": rmse(composition, hd_area),
                "jsd_vs_HDarea": jsd(composition, hd_area),
                "pearson_vs_HDarea": float(np.corrcoef(composition, hd_area)[0, 1]),
                "rmse_vs_snLV": rmse(composition, sn_lv),
                "jsd_vs_snLV": jsd(composition, sn_lv),
                "pearson_vs_snLV": float(np.corrcoef(composition, sn_lv)[0, 1]),
                "groundtruth_sim_rmse_HHCA": float(sim_rmse) if sim_rmse == sim_rmse else np.nan,
                "groundtruth_sim_jsd_HHCA": float(sim_jsd) if sim_jsd == sim_jsd else np.nan,
            }
        )
    agreement = pd.DataFrame(rows).set_index("method")
    return agreement.sort_values("rmse_vs_HDarea")


def main():
    """Collect the LV-endo composition of every source and write the composition and agreement tables."""
    args = parse_args()
    table_dir = f"{args.out}/tables"
    os.makedirs(table_dir, exist_ok=True)
    with open(args.factor_order) as handle:
        l3_types = list(json.load(handle))  # the 23 Level-3 types, in factor order

    visium_l3 = load_visium_methods(l3_types, args.outputs, args.nnls_dir)
    hd_area = load_hd_area_fraction(args.areafraction, l3_types)
    print(f"HD area fraction, endo CM={hd_area['Ventricular Cardiomyocytes']:.3f}")
    hd_nuclei = load_hd_nucleus_counts(args.units, l3_types)
    print(f"HD nucleus count CM={hd_nuclei['Ventricular Cardiomyocytes']:.3f}")
    sn_comp = load_snrna_compartments(args.sn)
    print(f"snRNA-seq CM={sn_comp['Cardiomyocytes']:.3f}")

    # compartment table of all sources
    comp_tbl = pd.DataFrame(index=COMP_ORDER)
    for method in visium_l3:
        comp_tbl[f"stdVisium_{method}"] = to_comp(visium_l3[method])
    comp_tbl["HD_area_fraction"] = to_comp(hd_area)
    comp_tbl["HD_nucleus_count"] = to_comp(hd_nuclei)
    comp_tbl["sn_LV"] = sn_comp
    comp_tbl = comp_tbl.fillna(0.0)
    comp_pct = (comp_tbl * 100).round(3)
    comp_pct.to_csv(f"{table_dir}/endo_LV_composition.csv", index_label="compartment")

    # Level-3 table (the snRNA reference has no Level-3 labels)
    l3_tbl = pd.DataFrame(index=l3_types)
    for method in visium_l3:
        l3_tbl[f"stdVisium_{method}"] = visium_l3[method]
    l3_tbl["HD_area_fraction"] = hd_area
    l3_tbl["HD_nucleus_count"] = hd_nuclei
    (l3_tbl.fillna(0.0) * 100).round(4).to_csv(f"{table_dir}/endo_LV_composition_L3.csv", index_label="L3_type")

    # accuracy on the ground-truth simulation
    leaderboard = pd.read_csv(args.leaderboard).set_index("method")
    agreement = method_agreement(comp_tbl, visium_l3, leaderboard)
    agreement.round(4).to_csv(f"{table_dir}/method_agreement.csv", index_label="method")

    best_hd = agreement["rmse_vs_HDarea"].idxmin()
    best_sn = agreement["rmse_vs_snLV"].idxmin()
    best_sim = leaderboard["rmse"].idxmin()

    # RMSE on the simulated endo section
    region = pd.read_csv(args.region_effect)
    endo_rmse = region[region["region"] == "endo"].set_index("method")["rmse"].round(4).to_dict()
    summary = {
        "sv_endo_slides": SV_ENDO_SLIDES,
        "hd_endo_slides": HD_ENDO_SLIDES,
        "primary_method": PRIMARY,
        "immune_clean_alt": IMMUNE_CLEAN_ALT,
        "closest_to_HDarea_method": best_hd,
        "closest_to_snLV_method": best_sn,
        "groundtruth_winner": best_sim,
        "groundtruth_leaderboard_rmse": leaderboard["rmse"].round(4).to_dict(),
        "groundtruth_endo_rmse": endo_rmse,
        "CM_fraction_by_source": {
            **{
                f"stdVisium_{method}": round(float(comp_tbl.loc["Cardiomyocytes", f"stdVisium_{method}"]), 3)
                for method in visium_l3
            },
            "HD_area_fraction": round(float(comp_tbl.loc["Cardiomyocytes", "HD_area_fraction"]), 3),
            "HD_nucleus_count": round(float(comp_tbl.loc["Cardiomyocytes", "HD_nucleus_count"]), 3),
            "sn_LV": round(float(comp_tbl.loc["Cardiomyocytes", "sn_LV"]), 3),
        },
    }
    with open(f"{table_dir}/summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\ncompartment composition (%):")
    print(comp_pct.to_string())
    print("\nmethod agreement (sorted by RMSE against the HD area fraction):")
    print(agreement.round(4).to_string())
    print(
        f"\nclosest to the HD area fraction: {best_hd} | closest to snRNA-seq: {best_sn} | "
        f"best on the simulation: {best_sim}"
    )
    print("\ntables written to", table_dir)


if __name__ == "__main__":
    main()
