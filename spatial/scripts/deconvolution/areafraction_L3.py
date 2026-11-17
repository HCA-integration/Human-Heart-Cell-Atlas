#!/usr/bin/env python
"""Area fraction of each Level-3 cell type on the Visium HD sections, from the FICTURE 2 um decode.

Area fraction = share of decoded 2 um bins whose top factor (K1) is the type. It follows tissue area and mRNA
footprint, so large transcript-rich cells (cardiomyocytes) weigh more than in nucleus counts. Only the K1 column
is streamed.

Inputs:  the pixel-level decode of each slide, the Level-3 factor order
Outputs: <out>/lv_areafraction_L3_perslide.csv   cell type x slide, % of decoded bins
         <out>/lv_areafraction_L3_summary.csv    mean, sd, min, max, median across slides
         <out>/lv_areafraction_L3_meta.json      slides and bins per slide

Usage: areafraction_L3.py --decode-template '.../{S}/output/...pixel.tsv.gz' --factor-order JSON --out DIR
           [--slides a,b,...]
"""

import os

from hhca import thread_env

os.environ.update(thread_env(1, numba=False))

import argparse
import json

import numpy as np
import pandas as pd

from hhca.log import get_logger

# LV Visium HD sections used when --slides is not given
SLIDES = [
    "VS_BO_2024BH-008_myo",
    "VS_BO_HDZ150319_34_myo",
    "VS_CA_H0054LV_henricke_endo",
    "VS_CA_H0054LV_henricke_epi",
    "VS_CA_H0054LV_henricke_myo",
    "VS_CA_H0063LV_henricke_endo",
    "VS_CA_H0063LV_henricke_epi",
    "VS_CA_H0063LV_henricke_myo",
    "VS_AH9_LV_FW_epi",
]


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--decode-template", required=True, help="path of the pixel decode, with {S} for the slide name"
    )
    parser.add_argument("--factor-order", required=True, help="json list of the Level-3 factor names")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--slides", default="", help="comma-separated slide names (default: the nine LV sections below)"
    )
    return parser.parse_args()


def main():
    """Count the decoded bins per type on every slide and write the per-slide and summary tables."""
    args = parse_args()
    log = get_logger()
    os.makedirs(args.out, exist_ok=True)
    with open(args.factor_order) as handle:
        factor_order = json.load(handle)
    n_factors = len(factor_order)
    slides = [slide for slide in args.slides.split(",") if slide] if args.slides else SLIDES

    percent = {}
    n_bins = {}
    for slide in slides:
        path = args.decode_template.format(S=slide)
        if not os.path.exists(path):
            log(f"skip {slide} (no decode)")
            continue
        total = np.zeros(n_factors, np.int64)
        n = 0
        for chunk in pd.read_csv(path, sep="\t", usecols=["K1"], dtype={"K1": np.int16}, chunksize=10_000_000):
            total += np.bincount(chunk["K1"].values.astype(np.int64), minlength=n_factors)
            n += len(chunk)
        percent[slide] = 100.0 * total / max(n, 1)
        n_bins[slide] = n
        log(f"{slide}: {n:,} decoded 2um bins")

    per_slide = pd.DataFrame(percent, index=factor_order)
    per_slide.index.name = "L3_type"
    per_slide = per_slide.round(4)
    per_slide.to_csv(f"{args.out}/lv_areafraction_L3_perslide.csv")

    values = per_slide.values
    summary = pd.DataFrame(
        {
            "mean_pct": values.mean(1),
            "sd_pct": values.std(1, ddof=1),
            "min_pct": values.min(1),
            "max_pct": values.max(1),
            "median_pct": np.median(values, 1),
        },
        index=per_slide.index,
    ).round(4)
    summary = summary.sort_values("mean_pct", ascending=False)
    summary.to_csv(f"{args.out}/lv_areafraction_L3_summary.csv")

    with open(f"{args.out}/lv_areafraction_L3_meta.json", "w") as handle:
        json.dump({"slides": list(percent.keys()), "n_bins": n_bins, "n_slides": len(percent)}, handle, indent=2)
    log("wrote the per-slide and summary tables")
    log("\n" + summary.to_string())


if __name__ == "__main__":
    main()
