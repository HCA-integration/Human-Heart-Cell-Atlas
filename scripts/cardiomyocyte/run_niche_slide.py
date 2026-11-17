"""Fig 3 f-h, step 1: 10 um niche bins of one Visium HD slide and the NPPA / NPPB high-vs-low comparison.

Decode pixels and FICTURE input transcripts are counted on a 10 um grid (hhca.tl.niche_bins); every bin gets the
composition of its neighbourhood and the distance to the nearest fibroblast-rich, endocardial and vessel-rich bin
(hhca.tl.niche_metrics); gene-high cardiomyocyte bins are compared with the other cardiomyocyte bins
(hhca.tl.compare_high_low). Parameters: `niche` in config/cardiomyocyte.yaml.

Inputs: the slide's FICTURE pixel decode and input transcripts, the Level-3 factor order and, with --flags, the
CMFLAG <slide>_clean_bins.parquet (pixels and transcripts of flagged 2 um bins are left out).
Outputs: in --out-dir, niche_bins_<slide>.csv.gz (one row per 10 um bin) and niche_summary_<slide>.json.

Usage: run_niche_slide.py SLIDE --decode PIXEL.tsv.gz --transcripts transcripts.tsv.gz
                          --factor-order atlas_L3_factor_order.json --out-dir DIR [--flags PARQUET]
"""

import os

from hhca import thread_env

os.environ.update(thread_env(1, keep_existing=True))

import argparse
import json

import hhca
from hhca.config import load_config
from hhca.grid import spillover_keys
from hhca.log import get_logger


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="10 um niche bins of one Visium HD slide (Fig 3 f-h)")
    parser.add_argument("slide", help="slide id (names the output files)")
    parser.add_argument("--decode", required=True, help="FICTURE pixel decode of the slide (tsv.gz)")
    parser.add_argument("--transcripts", required=True, help="FICTURE input transcripts of the slide (tsv.gz)")
    parser.add_argument("--factor-order", required=True, help="JSON list of the Level-3 factor names")
    parser.add_argument("--out-dir", required=True, help="output folder")
    parser.add_argument("--flags", default="", help="CMFLAG <slide>_clean_bins.parquet; flagged 2 um bins are left out")
    return parser.parse_args()


def main():
    """Build the bin table of the slide, write it and the high-vs-low summary."""
    args = parse_args()
    niche = load_config("cardiomyocyte")["niche"]
    os.makedirs(args.out_dir, exist_ok=True)
    log = get_logger(args.slide)
    with open(args.factor_order) as handle:
        factor_order = json.load(handle)
    bins = hhca.tl.niche_bins(
        hhca.io.read_decode(args.decode),
        args.transcripts,
        factor_order,
        niche["groups"],
        genes=niche["genes"],
        spillover=spillover_keys(args.flags) if args.flags else None,
        grid_um=niche["grid_um"],
        min_pixels=niche["min_pixels"],
        chunksize=niche["transcript_chunk"],
    )
    log(f"{len(bins):,} bins")
    bins = hhca.tl.niche_metrics(bins, niche["neighbourhoods"], niche["rich"], niche["radius_um"])
    columns = ["bin", "x_um", "y_um", "npix", "total_counts", *niche["genes"]]
    columns += [f"{gene}_cp1k" for gene in niche["genes"]] + [f"frac_{group}" for group in niche["groups"]]
    columns += [f"nbhd_{group}" for group in niche["neighbourhoods"]] + [f"dist_{group}_um" for group in niche["rich"]]
    bins_path = f"{args.out_dir}/niche_bins_{args.slide}.csv.gz"
    bins[columns].to_csv(bins_path, index=False, compression="gzip")
    log(f"saved bins -> {bins_path}")

    summary = {
        "slide": args.slide,
        "n_bins": int(len(bins)),
        "grid_um": niche["grid_um"],
        "nbhd_um": niche["radius_um"],
        "n_endocardial_pixels_bins": int((bins["frac_endocardial"] > 0).sum()),
    }
    for gene in niche["genes"]:
        summary[f"frac_bins_with_{gene}"] = float((bins[gene] > 0).mean())
    for gene in niche["genes"]:
        summary[gene] = hhca.tl.compare_high_low(
            bins, gene, niche["metrics"], niche["cm_fraction"], niche["high_quantile"]
        )
        log(f"{gene}: n_high={summary[gene].get('n_high')} metrics={list(summary[gene].get('metrics', {}))}")
    with open(f"{args.out_dir}/niche_summary_{args.slide}.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    log("saved summary, done")


if __name__ == "__main__":
    main()
