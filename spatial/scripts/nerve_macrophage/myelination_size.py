#!/usr/bin/env python
"""Fig 5 e data: per-nerve myelination score vs nerve size, pooled over the per-nerve pseudobulks of all slides.

Nerves with n_bins >= 3, myelinating = myelin_total >= 3, size classes 3-5 / 6-9 / 10-19 / 20-39 / >= 40 bins
(left-closed), score = log1p(myelin_total) (hhca.tl.nerve_myelination; config `nerve`), Kruskal-Wallis over the
classes, Spearman(n_bins, myelin_total). Drawn by figures/Fig5/panel_e.py.

Inputs: the per-nerve pseudobulk h5ads of nerves.py.
Outputs: per_nerve.csv (one row per nerve: n_bins, myelin_total, slide, myelinating, szbin, logmy) and
per_nerve_summary.json.

Usage: python myelination_size.py --pseudobulk P1.h5ad P2.h5ad ... --slides S1 S2 ... --out per_nerve.csv
"""

import argparse
import json
import os
import warnings

import anndata as ad
import pandas as pd
from scipy import stats

import hhca
from hhca.config import load_config

warnings.filterwarnings("ignore")


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pseudobulk", nargs="+", required=True, help="per-nerve pseudobulk h5ads, one per slide")
    parser.add_argument("--slides", nargs="+", required=True, help="slide ids, in the order of --pseudobulk")
    parser.add_argument("--out", required=True, help="output csv; the summary json is written next to it")
    return parser.parse_args()


def main():
    """Pool the nerves, test the size dependence of the myelination score and write table and summary."""
    args = parse_args()
    assert len(args.pseudobulk) == len(args.slides)
    nerve_cfg = load_config("nerve_macrophage")["nerve"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pooled = pd.concat(
        [
            ad.read_h5ad(path).obs[["n_bins", "myelin_total"]].assign(slide=slide)
            for slide, path in zip(args.slides, args.pseudobulk)
        ]
    )
    nerves = hhca.tl.nerve_myelination(pooled, nerve_cfg["myelinated_min"], nerve_cfg["size_classes"])
    labels = hhca.tl.size_class_labels(nerve_cfg["size_classes"])
    kruskal_h, kruskal_p = stats.kruskal(*[nerves[nerves.szbin == label].logmy.values for label in labels])
    rho, rho_p = stats.spearmanr(nerves.n_bins, nerves.myelin_total)
    nerves.to_csv(args.out, index=False)
    print(
        f"{nerves.slide.nunique()} slides: {len(nerves)} nerves; Kruskal p={kruskal_p:.2e}; "
        f"Spearman rho={rho:.3f} p={rho_p:.1e}"
    )
    classes = []
    for label in labels:
        members = nerves[nerves.szbin == label]
        classes.append(
            {
                "size_class": label,
                "n": int(len(members)),
                "pct_myelinating": float(100 * members.myelinating.mean()),
                "median_logmy": float(members.logmy.median()),
            }
        )
        print(f"  {label:6s} n={len(members):5d}  %myel={100 * members.myelinating.mean():.1f}")
    per_slide = (
        nerves.groupby("slide")
        .agg(n=("n_bins", "size"), pct_myelinating=("myelinating", "mean"))
        .assign(pct_myelinating=lambda t: 100 * t.pct_myelinating)
        .reset_index()
        .to_dict("records")
    )
    summary = {
        "n_nerves": int(len(nerves)),
        "n_slides": int(nerves.slide.nunique()),
        "kruskal_H": float(kruskal_h),
        "kruskal_p": float(kruskal_p),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "classes": classes,
        "per_slide": per_slide,
    }
    with open(args.out.replace(".csv", "_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
