#!/usr/bin/env python3
"""Certainty score per cell type, and candidate per-type thresholds.

Reads the feature checkpoints of build_certainty_score.py (per decode row: k1 = top factor, margin = P1 - P2,
depth_raw, marker_raw, spatial_raw, marker_support, cross_stable_bin, refit_changed), pools up to 250,000 rows
per type and slide (hhca.celltyping.pool_type_samples) and derives the thresholds
(hhca.celltyping.fit_type_thresholds).

Inputs: --checkpoints (<slide>.features.npz), --cert-tables (normalization_quantiles.json,
marker_sets_from_pseudobulk.json), --atlas-dir (Level-3 model and factor order), --weights and --params (the
`certainty.weights` and `tau` blocks of config/substrate.yaml as JSON).
Outputs in <outdir>/tables:
  pertype_score_distributions.csv   quantiles of the score and medians of its components per type
  signature_density.csv             marker enrichment in the model and observed marker density per type
  pertype_thresholds.csv            candidate thresholds per type and the coverage each one keeps
  global_normalization.json         pooled 5-95 % quantiles, weights and per-type quantiles
and in --cache: pooled_samples.npz, the pooled samples read by assign_with_thresholds.py.

Usage: python fit_type_thresholds.py --checkpoints DIR --cert-tables DIR --atlas-dir DIR --outdir DIR --cache DIR \
           --weights JSON --params JSON
"""

import argparse
import json
import os
from pathlib import Path

from hhca import thread_env

os.environ.update(thread_env(1, keep_existing=True))

import numpy as np

from certainty import checkpoint_slides, load_factor_order, load_model
from hhca import celltyping

CACHED_FIELDS = ["cert_global", "marker_raw", "depth_raw", "marker_support", "cross_stable_bin", "refit_changed"]


def parse_args():
    """Command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--checkpoints", required=True, help="folder with the certainty checkpoints <slide>.features.npz"
    )
    parser.add_argument(
        "--cert-tables", required=True, help="certainty tables folder (normalization_quantiles.json, marker sets)"
    )
    parser.add_argument("--atlas-dir", required=True, help="folder with atlas_L3_pseudobulk.tsv and the factor order")
    parser.add_argument("--outdir", required=True, help="tables are written to <outdir>/tables")
    parser.add_argument(
        "--cache", required=True, help="folder for the pooled samples read by assign_with_thresholds.py"
    )
    parser.add_argument("--weights", required=True, help="`certainty.weights` of config/substrate.yaml as JSON")
    parser.add_argument("--params", required=True, help="the `tau` block of config/substrate.yaml as JSON")
    return parser.parse_args()


def main():
    """Pool the checkpoints per type and write the score distributions and candidate thresholds."""
    args = parse_args()
    weights, params = json.loads(args.weights), json.loads(args.params)
    atlas_dir = Path(args.atlas_dir)
    checkpoint_dir = Path(args.checkpoints)
    cert_table_dir = Path(args.cert_tables)
    table_dir = Path(args.outdir) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    order = load_factor_order(atlas_dir)
    n_types = len(order)
    # pooled 5-95 % quantiles of the score components
    quantiles = json.loads((cert_table_dir / "normalization_quantiles.json").read_text())
    markers = json.loads((cert_table_dir / "marker_sets_from_pseudobulk.json").read_text())

    slides = checkpoint_slides(checkpoint_dir)
    print("slides:", slides)
    checkpoints = {slide: np.load(checkpoint_dir / f"{slide}.features.npz") for slide in slides}
    pooled, n_bins = celltyping.pool_type_samples(
        checkpoints, n_types, quantiles, weights, params["sample_cap"], params["sampling_seed"]
    )
    for checkpoint in checkpoints.values():
        checkpoint.close()
    fit = celltyping.fit_type_thresholds(
        pooled,
        n_bins,
        load_model(atlas_dir, order),
        markers,
        quantiles,
        weights,
        params["global_tau"],
        params["rare_types"],
    )

    fit["distributions"].to_csv(table_dir / "pertype_score_distributions.csv", index=False)
    fit["density"].to_csv(table_dir / "signature_density.csv", index=False)
    fit["thresholds"].to_csv(table_dir / "pertype_thresholds.csv", index=False)
    with open(table_dir / "global_normalization.json", "w") as handle:
        json.dump(fit["normalization"], handle, indent=2)

    np.savez_compressed(
        cache_dir / "pooled_samples.npz",
        **{
            f"{t}_{field}": pooled[t][field] for t in range(n_types) if pooled[t] is not None for field in CACHED_FIELDS
        },
        order=np.array(order, dtype=object),
        tot_bins=n_bins,
    )
    print("tables ->", table_dir)
    print(
        fit["thresholds"][
            [
                "cell_type",
                "n_argmax_bins",
                "tau_faith_calibrated",
                "cov_global",
                "cov_corrected",
                "cov_faith",
                "faith_global",
                "faith_corrected",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
