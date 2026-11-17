#!/usr/bin/env python3
"""Per-type thresholds on the certainty score, and the cell-type composition they give per slide.

The posterior of the decode is close to 0 or 1 for most bins, so the confidence of a bin is its certainty
score, not P1. The threshold of a type (pertype_tau_recommended.json) is the score that keeps the same share
of the type's bins as the density-corrected score of fit_type_thresholds.py keeps at 0.37.
Assignments compared:
  decode_argmax         top factor of every bin, no threshold
  global_037            top factor if the score is >= 0.37
  pertype               top factor if the score is >= the threshold of its type
  pertype_rare_first, pertype_abund_first
                        the first of the top-5 factors that passes its threshold, rare or abundant types
                        taking priority

The thresholds and the assignments are computed by hhca.celltyping (type_thresholds, assignment_schemes).

Inputs: --checkpoints (<slide>.features.npz), --cert-tables (normalization_quantiles.json), --atlas-dir (factor
order), --decode-root (<slide>/output/<decode>), <outdir>/tables/pertype_thresholds.csv and
<cache>/pooled_samples.npz of fit_type_thresholds.py, --weights and --params (the `certainty.weights` and `tau`
blocks of config/substrate.yaml as JSON).
Outputs in <outdir>/tables: pertype_tau_recommended.json, composition_<assignment>.csv (% per slide),
pooled_counts_by_scheme.csv, the share of each type kept (rare_type_recovery_qc.csv), the effect of the priority
order (layering_order_comparison.csv), the comparison with the refit decode (decode_vs_refit.csv) and
assignment_summary.json; in --cache: refit_changed_certainty.npz.

Usage: python assign_with_thresholds.py --checkpoints DIR --cert-tables DIR --atlas-dir DIR --outdir DIR \
           --cache DIR --decode-root DIR --weights JSON --params JSON
"""

import argparse
import json
import os
from pathlib import Path

from hhca import thread_env

os.environ.update(thread_env(1, keep_existing=True))

import numpy as np
import pandas as pd

from certainty import DECODE_NAME, checkpoint_slides, load_factor_order, read_decode
from hhca import celltyping
from hhca.celltyping import SCHEMES

VENTRICULAR_CM = "Ventricular Cardiomyocytes"
FACTOR_COLUMNS = ["K1", "K2", "K3", "K4", "K5"]
POSTERIOR_COLUMNS = ["P1", "P2", "P3", "P4", "P5"]
# scores kept per slide and refit type for the mean certainty of the bins the refit decode calls differently
REFIT_SAMPLE_CAP = 50000
REFIT_SAMPLING_SEED = 1


def write_compositions(composition, unassigned, order, slides, table_dir):
    """Write the composition per slide (% of the assigned bins) of each assignment and the counts over all slides."""
    for scheme in SCHEMES:
        counts = composition[scheme]
        assigned = counts.sum(1, keepdims=True)
        assigned[assigned == 0] = 1
        percent = 100.0 * counts / assigned
        pd.DataFrame(percent.T, index=order, columns=slides).to_csv(table_dir / f"composition_{scheme}.csv")
    pooled_counts = pd.DataFrame({scheme: composition[scheme].sum(0) for scheme in SCHEMES}, index=order)
    pooled_counts.loc["__UNASSIGNED__"] = [unassigned[scheme].sum() for scheme in SCHEMES]
    pooled_counts.to_csv(table_dir / "pooled_counts_by_scheme.csv")


def recovery_table(kept_bins, order, rare_types):
    """Share of each type kept by the single and by the per-type threshold, rarest type first."""
    rows = []
    for t, name in enumerate(order):
        kept = kept_bins[t]
        if kept["n_argmax"] == 0:
            continue
        rows.append(
            {
                "cell_type": name,
                "is_rare": name in rare_types,
                "n_argmax": kept["n_argmax"],
                "kept_global": kept["keep_global"],
                "kept_pertype": kept["keep_pertype"],
                "cov_global": round(kept["keep_global"] / kept["n_argmax"], 4),
                "cov_pertype": round(kept["keep_pertype"] / kept["n_argmax"], 4),
                "recovery_x": round((kept["keep_pertype"] / max(1, kept["keep_global"])), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("n_argmax")


def percent_other(counts, type_index):
    """Percent of bins that are not of type `type_index`."""
    total = counts.sum()
    return 100.0 * (total - counts[type_index]) / total if total else np.nan


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
    parser.add_argument("--cache", required=True, help="folder with the pooled samples of fit_type_thresholds.py")
    parser.add_argument("--decode-root", required=True, help="folder with <slide>/output/<decode>")
    parser.add_argument("--weights", required=True, help="`certainty.weights` of config/substrate.yaml as JSON")
    parser.add_argument("--params", required=True, help="the `tau` block of config/substrate.yaml as JSON")
    return parser.parse_args()


def main():
    """Set the threshold per type, assign the bins of every slide and write the comparison tables."""
    args = parse_args()
    weights, params = json.loads(args.weights), json.loads(args.params)
    decode_root = Path(args.decode_root)
    checkpoint_dir = Path(args.checkpoints)
    table_dir = Path(args.outdir) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    order = load_factor_order(Path(args.atlas_dir))
    n_types = len(order)
    ventricular_cm = order.index(VENTRICULAR_CM)
    quantiles = json.loads((Path(args.cert_tables) / "normalization_quantiles.json").read_text())

    pooled_samples = np.load(cache_dir / "pooled_samples.npz", allow_pickle=True)
    pooled_scores = {
        t: pooled_samples[f"{t}_cert_global"] for t in range(n_types) if f"{t}_cert_global" in pooled_samples.files
    }
    tau = celltyping.type_thresholds(
        pd.read_csv(table_dir / "pertype_thresholds.csv"), pooled_scores, order, params["global_tau"]
    ).to_numpy()
    with open(table_dir / "pertype_tau_recommended.json", "w") as handle:
        json.dump({order[t]: round(float(tau[t]), 4) for t in range(n_types)}, handle, indent=2)
    print("threshold per type:")
    for t in np.argsort(tau):
        print(f"  {order[t]:34s} tau={tau[t]:.4f}")

    slides = checkpoint_slides(checkpoint_dir)
    # priority orders from the number of bins per type over all slides
    bins_per_type = pooled_samples["tot_bins"]

    composition = {scheme: np.zeros((len(slides), n_types), dtype=np.int64) for scheme in SCHEMES}
    unassigned = {scheme: np.zeros(len(slides), dtype=np.int64) for scheme in SCHEMES}
    refit_argmax = np.zeros((len(slides), n_types), dtype=np.int64)
    # bins kept per type, over all slides
    kept_bins = {t: {"n_argmax": 0, "keep_global": 0, "keep_pertype": 0} for t in range(n_types)}
    order_diff_bins = np.zeros(len(slides), dtype=np.int64)
    # certainty of the bins that the refit decode calls a different type: refit type -> list of scores
    refit_changed_scores = {}
    refit_changed_n = np.zeros(n_types, dtype=np.int64)

    for si, slide in enumerate(slides):
        checkpoint = np.load(checkpoint_dir / f"{slide}.features.npz")
        k1 = checkpoint["k1"].astype(np.int64)
        score = celltyping.certainty_score(checkpoint, quantiles, weights)
        if "refit_k1" in checkpoint.files:
            refit = checkpoint["refit_k1"].astype(np.int64)
        else:
            refit = np.full(k1.size, -1, np.int64)
        checkpoint.close()
        n = k1.size
        # top-5 factors of the decode, same row order as the checkpoint
        decode = read_decode(
            f"{decode_root}/{slide}/output/{DECODE_NAME}",
            FACTOR_COLUMNS + POSTERIOR_COLUMNS,
            dict.fromkeys(FACTOR_COLUMNS, np.int16),
        )
        assignments = celltyping.assignment_schemes(
            decode, score, tau, bins_per_type, params["global_tau"], params["p_floor"]
        )
        del decode
        keep_global = assignments["global_037"] >= 0
        keep_pertype = assignments["pertype"] >= 0
        order_diff_bins[si] = int((assignments["pertype_rare_first"] != assignments["pertype_abund_first"]).sum())
        for scheme, assigned in assignments.items():
            is_assigned = assigned >= 0
            composition[scheme][si] = np.bincount(assigned[is_assigned], minlength=n_types)
            unassigned[scheme][si] = int((~is_assigned).sum())
        has_refit = refit >= 0
        refit_argmax[si] = np.bincount(refit[has_refit], minlength=n_types)

        for t in range(n_types):
            of_type = k1 == t
            if of_type.any():
                kept_bins[t]["n_argmax"] += int(of_type.sum())
                kept_bins[t]["keep_global"] += int(keep_global[of_type].sum())
                kept_bins[t]["keep_pertype"] += int(keep_pertype[of_type].sum())
        # certainty of the bins whose call changes in the refit decode
        if has_refit.any():
            changed = has_refit & (refit != k1)
            for refit_type in np.unique(refit[changed]):
                refit_changed_n[refit_type] += int((refit[changed] == refit_type).sum())
                scores = score[changed][refit[changed] == refit_type]
                if scores.size > REFIT_SAMPLE_CAP:
                    scores = np.random.default_rng(REFIT_SAMPLING_SEED).choice(scores, REFIT_SAMPLE_CAP, replace=False)
                refit_changed_scores.setdefault(int(refit_type), []).append(scores)
        print(
            f"  {slide}: n={n} order_diff={order_diff_bins[si]} pertype_unassigned={unassigned['pertype'][si] / n:.3f}"
        )

    write_compositions(composition, unassigned, order, slides, table_dir)
    recovery_table(kept_bins, order, params["rare_types"]).to_csv(table_dir / "rare_type_recovery_qc.csv", index=False)

    # effect of the priority order
    layering = pd.DataFrame(
        {
            "slide": slides,
            "order_diff_bins": order_diff_bins,
            "n_bins": [
                composition["decode_argmax"][i].sum() + unassigned["decode_argmax"][i] for i in range(len(slides))
            ],
        }
    )
    layering["order_diff_frac"] = layering.order_diff_bins / layering.n_bins
    layering.to_csv(table_dir / "layering_order_comparison.csv", index=False)

    # decode vs refit decode
    decode_bins = composition["decode_argmax"].sum(0)
    refit_bins = refit_argmax.sum(0)
    pd.DataFrame(
        {
            "cell_type": order,
            "decode_argmax_bins": decode_bins,
            "refit_argmax_bins": refit_bins,
            "decode_pct": 100.0 * decode_bins / decode_bins.sum(),
            "refit_pct": 100.0 * refit_bins / max(1, refit_bins.sum()),
            "refit_changed_to_bins": refit_changed_n,
            "mean_decode_cert_of_refit_changed": [
                (float(np.concatenate(refit_changed_scores[t]).mean()) if t in refit_changed_scores else np.nan)
                for t in range(n_types)
            ],
        }
    ).to_csv(table_dir / "decode_vs_refit.csv", index=False)
    summary = {
        "n_slides": len(slides),
        "slides": slides,
        "decode_nonCM_pct": round(percent_other(decode_bins, ventricular_cm), 4),
        "refit_nonCM_pct": round(percent_other(refit_bins, ventricular_cm), 4),
        "decode_VCM_pct": round(100.0 * decode_bins[ventricular_cm] / decode_bins.sum(), 4),
        "refit_VCM_pct": round(100.0 * refit_bins[ventricular_cm] / max(1, refit_bins.sum()), 4),
        "mean_order_diff_frac": float(layering.order_diff_frac.mean()),
        "total_order_diff_bins": int(order_diff_bins.sum()),
        "total_bins": int(layering.n_bins.sum()),
    }
    with open(table_dir / "assignment_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    np.savez_compressed(
        cache_dir / "refit_changed_certainty.npz",
        **{f"{t}": np.concatenate(scores) for t, scores in refit_changed_scores.items()},
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
