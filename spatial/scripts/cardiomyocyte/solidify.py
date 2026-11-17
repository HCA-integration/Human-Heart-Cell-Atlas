"""Fig 3 f-h, step 2: NPPA / NPPB-high vs the other cardiomyocyte bins, tested per slide and pooled over slides.

Metrics per 10 um cardiomyocyte bin: (a) fibroblast / endocardial / vessel / immune fraction within 30 um,
(b) distance to the nearest fibroblast-rich, endocardial or vessel-rich bin. For every slide, gene and metric:
one-sided Mann-Whitney U and a label-permutation null (2000 shuffles of the high / low labels among cardiomyocyte
bins; empirical p and z); BH q values across all tests (hhca.tl.niche_tests). Parameters: `niche` in
config/cardiomyocyte.yaml.

Inputs: niche_bins_<slide>.csv.gz of run_niche_slide.py.
Outputs: in --out-dir, niche_solidify_pooled.csv (one row per slide, gene and metric) and
niche_solidify_consistency.json (number of slides with a significant effect in the expected direction, per metric
and gene).

Usage: solidify.py --bins-dir DIR --out-dir DIR
"""

import os

from hhca import thread_env

os.environ.update(thread_env(1, keep_existing=True))

import argparse
import json

import pandas as pd

import hhca
from hhca.config import load_config
from hhca.log import get_logger


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Pooled NPPA / NPPB niche statistics (Fig 3 f-h)")
    parser.add_argument("--bins-dir", required=True, help="folder with niche_bins_<slide>.csv.gz")
    parser.add_argument("--out-dir", required=True, help="output folder")
    return parser.parse_args()


def main():
    """Test every slide, adjust the p values over all tests and write the pooled table and the consistency."""
    args = parse_args()
    cfg = load_config("cardiomyocyte")
    niche = cfg["niche"]
    slides = cfg["slides"]["berlin"] + cfg["slides"]["ah"]
    endo_slides = cfg["slides"]["endo_sections"]
    os.makedirs(args.out_dir, exist_ok=True)
    log = get_logger("solidify")

    pool = hhca.tl.niche_tests(
        {slide: pd.read_csv(f"{args.bins_dir}/niche_bins_{slide}.csv.gz") for slide in slides},
        niche["genes"],
        niche["metrics"],
        endo_slides,
        labels={
            slide: {"short": f"{section['donor']} {section['layer']}", "donor": section["donor"]}
            for slide, section in cfg["sections"].items()
        },
        cm_fraction=niche["cm_fraction"],
        quantile=niche["high_quantile"],
        n_perm=niche["n_perm"],
        seed=niche["seed"],
        alpha=niche["alpha"],
        log=log,
    )
    pool.to_csv(f"{args.out_dir}/niche_solidify_pooled.csv", index=False)
    log(f"wrote pooled stats: {len(pool)} rows -> {args.out_dir}/niche_solidify_pooled.csv")

    summaries = [
        ("fibroblast_nbhd_allslides", "nbhd_fibroblast", slides),
        ("fibroblast_dist_allslides", "dist_fibroblast_um", slides),
        ("endocardial_nbhd_endosections", "nbhd_endocardial", endo_slides),
        ("endocardial_dist_endosections", "dist_endocardial_um", endo_slides),
        ("vessel_nbhd_allslides", "nbhd_vessel", slides),
        ("vessel_dist_allslides", "dist_vessel_um", slides),
        ("immune_nbhd_allslides", "nbhd_immune", slides),
    ]
    consistency = {
        key: hhca.tl.niche_consistency(pool, metric, on_slides, niche["alpha"]) for key, metric, on_slides in summaries
    }
    consistency["n_perm"] = niche["n_perm"]
    with open(f"{args.out_dir}/niche_solidify_consistency.json", "w") as handle:
        json.dump(consistency, handle, indent=2)
    log("across-slide consistency:")
    print(json.dumps(consistency, indent=2))
    log("done")


if __name__ == "__main__":
    main()
