#!/usr/bin/env python
"""Fig 5 d data, per nerve: shortest distance from a nerve's 2 um bins to the nearest vascular bin.

The distances are compared with those of the same nerve shapes placed at random tissue positions
(hhca.tl.nerve_vessel_distance; config `proximity`).

  nerve       nerve of nerves.py (>= 3 neural bins linked at 6 um, CMFLAG-clean 2 um h5ad)
  distance    min over the nerve's bins of the distance to the nearest vascular bin (5 types, config proximity)
  background  each nerve's bin set translated so that its medoid bin lands on a random non-neural, non-vascular
              bin of the same slide (--placements per nerve, numpy default_rng(--seed), slides in config order);
              a placement is kept when >= 90% of the moved bins lie within 1.5 um of a tissue bin (up to 50 tries).
              Same shape and size as the nerve, so the size dependence of a minimum distance cancels.
  test        one-sided Mann-Whitney, nerves closer than placements; per size class the fraction within 8 um

Inputs: the CMFLAG-clean 2 um h5ads and the neural-bin tables of nerves.py.
Outputs: per_nerve.npz (d_nerve, n_bins, slide index, d_background, background nerve index) and
per_nerve_summary.json.

Usage: python nerve_vessel_distance_pernerve.py --h5ad '<dir>/{s}.h5ad' --nerves-dir results/nerve_macrophage/nerves
           --slides S1 S2 ... --out results/nerve_macrophage/d/per_nerve.npz
"""

import argparse
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import hhca
from hhca import config_path
from hhca.config import load_config


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--h5ad", required=True, help="CMFLAG-clean 2 um h5ad, path template with {s} for the slide")
    parser.add_argument("--nerves-dir", required=True, help="folder with the <slide>_neural_bins.csv.gz of nerves.py")
    parser.add_argument("--slides", nargs="+", required=True, help="slide ids, in the order they are processed")
    parser.add_argument("--out", required=True, help="output npz; the summary json is written next to it")
    parser.add_argument("--placements", type=int, default=5, help="random placements per nerve")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random placements")
    parser.add_argument(
        "--config",
        help="pipeline config with the vascular and neural types",
        default=config_path("nerve_macrophage"),
    )
    return parser.parse_args()


def main():
    """Measure the nerve-vessel distances and their random-placement background on all slides."""
    args = parse_args()
    cfg = load_config(args.config)
    proximity = cfg["proximity"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sections = {slide: ad.read_h5ad(args.h5ad.replace("{s}", slide), backed="r") for slide in args.slides}
    neural_bins = {slide: pd.read_csv(f"{args.nerves_dir}/{slide}_neural_bins.csv.gz") for slide in args.slides}
    distances = hhca.tl.nerve_vessel_distance(
        sections,
        neural_bins,
        proximity["vascular_types"],
        cfg["nerve"]["neural_type"],
        placements=args.placements,
        seed=args.seed,
        max_tries=proximity["max_tries"],
        min_on_tissue=proximity["min_on_tissue"],
        on_tissue_um=proximity["on_tissue_um"],
    )
    nerves, placements = distances["nerves"], distances["placements"]
    print(distances["sections"].to_string(index=False), flush=True)
    d_nerve, d_background = nerves["d_vessel_um"].to_numpy(), placements["d_vessel_um"].to_numpy()
    U, p = mannwhitneyu(d_nerve, d_background, alternative="less")
    summary = {
        "n_nerves": int(len(d_nerve)),
        "n_background": int(len(d_background)),
        "median_nerve_um": float(np.median(d_nerve)),
        "median_background_um": float(np.median(d_background)),
    }
    for near in proximity["near_um"]:
        summary[f"pct_within_{near:g}um_nerve"] = float(100 * (d_nerve <= near).mean())
        summary[f"pct_within_{near:g}um_background"] = float(100 * (d_background <= near).mean())
    summary.update(
        {
            "mwu_U": float(U),
            "mwu_p_less": float(p),
            "placements_per_nerve": args.placements,
            "seed": args.seed,
            "size_classes": hhca.tl.proximity_by_size(
                nerves, placements, cfg["nerve"]["size_classes"], proximity["near_um"][0]
            ).to_dict("records"),
            "per_slide": distances["sections"].to_dict("records"),
        }
    )
    print(json.dumps({k: v for k, v in summary.items() if k not in ("size_classes", "per_slide")}, indent=1))
    np.savez_compressed(
        args.out,
        d_nerve=d_nerve,
        n_bins=nerves["n_bins"].to_numpy(),
        slide=nerves["slide"].map({slide: i for i, slide in enumerate(args.slides)}).to_numpy(),
        slides=np.array(args.slides),
        d_background=d_background,
        background_nerve=placements["nerve_index"].to_numpy(),
    )
    with open(args.out.replace(".npz", "_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
