#!/usr/bin/env python3
"""Certainty score for every 2 um bin of the FICTURE Level-3 decodes, and its operating threshold.

The score combines four components per decode row (hhca.celltyping.certainty_components), each scaled to [0, 1]
between the pooled 5th and 95th percentile (weights: `certainty.weights` in config/substrate.yaml):
  0.15  posterior margin        P1 - P2
  0.35  local depth             UMI and gene richness in the 3 x 3 neighbourhood
  0.30  marker faithfulness     share of the 3 x 3 UMI that are markers of the called type
  0.20  spatial agreement       share of the 8 neighbours with the same call
The threshold is chosen on a precision / coverage curve (hhca.celltyping.precision_coverage, select_threshold),
where a bin counts as positive when it has marker support and the same call in the secondary decodes (midground,
adaptive_res). Bins whose call changes in the adaptive (refit) decode are flagged; without secondary decodes the
threshold is --threshold.

Inputs: --decode-root (<slide>/output/<decode>, <slide>/inputs/{transcripts.tsv.gz,coordinate_minmax.tsv}),
--alt-root (<slide>/{midground,adaptive,adaptive_res}/output/<decode>), --atlas-dir (Level-3 model and factor
order), --params (the `certainty` block of config/substrate.yaml as JSON).
Outputs in --outdir: checkpoints/<slide>.features.npz (components per row), tables/ (certainty columns per
slide, row-aligned with the decode; quantiles; precision / coverage curve; threshold report).

Usage: python build_certainty_score.py --decode-root DIR --atlas-dir DIR --outdir DIR --slides A,B,... --params JSON
           [--alt-root DIR] [--checkpoint-only] [--threshold 0.37]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from hhca import thread_env

os.environ.update(thread_env(1, numba=False, keep_existing=True))

import numpy as np
import pandas as pd

from certainty import DECODE_NAME, load_factor_order, load_model, read_decode
from hhca import celltyping
from hhca.log import get_logger

# stands in for --alt-root when there are no secondary decodes: none of the paths below it exists
NO_ALT_ROOT = Path("/nonexistent")
log = get_logger("certainty")


def secondary_decode(folder: Path) -> Path:
    """Pixel table of a secondary decode; these are staged inputs and keep the model name they were written with."""
    found = sorted(folder.glob("*" + DECODE_NAME.split(".", 1)[1]))
    return found[0] if found else folder / DECODE_NAME


def read_calls(path: Path):
    """Row keys and top factor of a secondary decode."""
    decode = read_decode(path, ["X", "Y", "K1"], {"K1": np.int16})
    x, y = decode["X"].to_numpy(np.float32, copy=True), decode["Y"].to_numpy(np.float32, copy=True)
    return x, y, celltyping.decode_keys(x, y), decode["K1"].to_numpy(np.int16, copy=True)


def secondary_calls(slide, alt: Path, keys, order, table_dir: Path):
    """K1 of the midground, adaptive_res and refit decodes per row (-1 = absent), and the number of refit-only bins.

    The bins that only the refit decode assigns are written to <slide>_refit_only_bins.tsv.gz.
    """
    paths = {
        "refit": secondary_decode(alt / "adaptive" / "output"),
        "midground": secondary_decode(alt / "midground" / "output"),
        "adaptive": secondary_decode(alt / "adaptive_res" / "output"),
    }
    calls = {name: np.full(keys.shape[0], -1, dtype=np.int16) for name in paths}
    refit_only = 0
    for name in ("midground", "adaptive"):
        if paths[name].exists():
            _, _, other_keys, other_k1 = read_calls(paths[name])
            calls[name] = celltyping.match_calls(keys, other_keys, other_k1)
    if paths["refit"].exists():
        x, y, other_keys, other_k1 = read_calls(paths["refit"])
        calls["refit"] = celltyping.match_calls(keys, other_keys, other_k1)
        missing = celltyping.unmatched_rows(other_keys, keys)
        refit_only = int(missing.sum())
        if refit_only > 0:
            pd.DataFrame(
                {
                    "X": x[missing].astype(np.float32),
                    "Y": y[missing].astype(np.float32),
                    "K1_refit": other_k1[missing].astype(np.int16),
                    "label_refit": [order[i] for i in other_k1[missing]],
                }
            ).to_csv(table_dir / f"{slide}_refit_only_bins.tsv.gz", sep="\t", index=False, compression="gzip")
    return calls, refit_only, {name: bool(path.exists()) for name, path in paths.items()}


def build_slide_checkpoint(slide, args, order, markers, table_dir: Path, checkpoint_dir: Path) -> bool:
    """Write the feature checkpoint of a slide unless it exists; False if the slide has no decode."""
    run = Path(args.decode_root) / slide
    decode_path = run / "output" / DECODE_NAME
    chk_npz = checkpoint_dir / f"{slide}.features.npz"
    chk_json = checkpoint_dir / f"{slide}.meta.json"

    if not decode_path.exists():
        log(f"skip {slide}: no decode")
        return False
    if chk_npz.exists() and chk_json.exists() and not args.force:
        log(f"{slide}: checkpoint exists, skip")
        return True

    log(f"{slide}: reading the decode")
    decode = read_decode(decode_path, ["X", "Y", "K1", "P1", "P2"], {"K1": np.int16})
    log(f"{slide}: reading transcripts (local depth and marker UMI)")
    components = celltyping.certainty_components(decode, run / "inputs", markers, chunksize=args.chunksize)
    x, y, k1 = decode["X"].to_numpy(np.float32), decode["Y"].to_numpy(np.float32), decode["K1"].to_numpy(np.int16)

    log(f"{slide}: matching the calls of the secondary decodes")
    alt_root = Path(args.alt_root) if args.alt_root else NO_ALT_ROOT
    calls, refit_only, available = secondary_calls(
        slide, alt_root / slide, celltyping.decode_keys(x, y), order, table_dir
    )
    cross_stability, cross_stable_bin = celltyping.call_stability(k1, (calls["midground"], calls["adaptive"]))
    refit_changed = ((calls["refit"] >= 0) & (calls["refit"] != k1)).astype(np.uint8)
    nbins = int(k1.shape[0])

    vals = {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "k1": k1.astype(np.int16),
        "p1": decode["P1"].to_numpy(np.float32),
        "p2": decode["P2"].to_numpy(np.float32),
        "margin": components["margin"].to_numpy(np.float32),
        "local_umi": components["local_umi"].to_numpy(np.float32),
        "local_gene_est": components["local_gene_est"].to_numpy(np.float32),
        "marker_raw": components["marker_raw"].to_numpy(np.float32),
        "spatial_raw": components["spatial_raw"].to_numpy(np.float32),
        "depth_raw": components["depth_raw"].to_numpy(np.float32),
        "marker_support": components["marker_support"].to_numpy(np.uint8),
        "mid_k1": calls["midground"].astype(np.int16),
        "adaptive_k1": calls["adaptive"].astype(np.int16),
        "refit_k1": calls["refit"].astype(np.int16),
        "cross_stability": cross_stability.astype(np.float32),
        "cross_stable_bin": cross_stable_bin.astype(np.uint8),
        "refit_changed": refit_changed.astype(np.uint8),
        "refit_recall_flag": refit_changed.astype(np.uint8),
    }
    # np.savez_compressed appends ".npz" unless the path ends in it, so the partial file ends in ".npz" too
    partial_npz = chk_npz.parent / (chk_npz.stem + ".tmp.npz")
    np.savez_compressed(partial_npz, **vals)
    os.replace(partial_npz, chk_npz)
    meta = {
        "slide": slide,
        "n_bins": nbins,
        "refit_only_bins": refit_only,
        "has_midground": available["midground"],
        "has_refit": available["refit"],
        "has_adaptive": available["adaptive"],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    chk_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"{slide}: checkpoint written ({nbins} bins)")
    return True


def write_slide_tables(
    slides: list[str], checkpoint_dir: Path, table_dir: Path, quantiles: dict[str, list], weights, threshold: float
) -> tuple[list[dict], dict[str, int]]:
    """Write the certainty columns of every slide; return the summary row per slide and counts over all bins."""
    summary_rows = []
    counts = {"total": 0, "low": 0, "refit": 0, "low_and_refit": 0}
    for slide in slides:
        arr = np.load(checkpoint_dir / f"{slide}.features.npz")
        vals = {k: arr[k] for k in arr.files}
        arr.close()
        certainty = celltyping.certainty_score(vals, quantiles, weights)

        # certainty columns, one row per decode row in the same order; the components of the score stay in
        # the slide checkpoint
        columns = {
            "certainty_score": np.round(certainty, 4).astype(np.float32),
            "certainty_pass_recommended": (certainty >= threshold).astype(np.uint8),
            "refit_recall_flag": vals["refit_recall_flag"].astype(np.uint8),
            "refit_changed_call": vals["refit_changed"].astype(np.uint8),
        }
        log(f"writing certainty columns: {slide}")
        pd.DataFrame(columns).to_csv(
            table_dir / f"{slide}_certainty_columns.tsv.gz", sep="\t", index=False, compression="gzip"
        )

        n = certainty.shape[0]
        low = certainty < threshold
        rflag = vals["refit_recall_flag"].astype(bool)
        counts["total"] += n
        counts["low"] += int(low.sum())
        counts["refit"] += int(rflag.sum())
        counts["low_and_refit"] += int((low & rflag).sum())
        meta = json.loads((checkpoint_dir / f"{slide}.meta.json").read_text())
        summary_rows.append(
            {
                "slide": slide,
                "n_bins": int(n),
                "mean_certainty": float(certainty.mean()),
                "frac_pass_recommended": float((certainty >= threshold).mean()),
                "frac_marker_supported": float(vals["marker_support"].mean()),
                "frac_cross_stable": float(np.nanmean(vals["cross_stability"])),
                "frac_refit_changed": float(vals["refit_changed"].mean()),
                "refit_only_bins": int(meta["refit_only_bins"]),
            }
        )
        del vals, certainty
    return summary_rows, counts


def write_json(value, path: Path) -> None:
    """Write `value` as indented JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def parse_args() -> argparse.Namespace:
    """Command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slides", required=True, help="comma-separated slide ids, in the order they are pooled")
    parser.add_argument("--force", action="store_true", help="rebuild slide checkpoints that exist")
    parser.add_argument("--params", required=True, help="the `certainty` block of config/substrate.yaml as JSON")
    parser.add_argument("--chunksize", type=int, default=1_200_000, help="transcript rows read per chunk")
    parser.add_argument(
        "--checkpoint-only", action="store_true", help="build the slide checkpoints and stop before pooling"
    )
    parser.add_argument("--decode-root", required=True, help="folder with <slide>/output and <slide>/inputs")
    parser.add_argument(
        "--alt-root",
        default=None,
        help="folder with the secondary decodes <slide>/{midground,adaptive,adaptive_res}/output",
    )
    parser.add_argument("--atlas-dir", required=True, help="folder with atlas_L3_pseudobulk.tsv and the factor order")
    parser.add_argument("--outdir", required=True, help="output folder (checkpoints/ and tables/)")
    parser.add_argument(
        "--threshold", type=float, default=None, help="threshold to use when there are no secondary decodes"
    )
    return parser.parse_args()


def main() -> None:
    """Build the slide checkpoints, choose the threshold on the pooled sample and write the tables."""
    args = parse_args()
    params = json.loads(args.params)
    weights = params["weights"]
    atlas_dir = Path(args.atlas_dir)
    out_dir = Path(args.outdir)
    table_dir = out_dir / "tables"
    checkpoint_dir = out_dir / "checkpoints"
    for folder in (out_dir, table_dir, checkpoint_dir):
        folder.mkdir(parents=True, exist_ok=True)
    slides = [s.strip() for s in args.slides.split(",") if s.strip()]

    order = load_factor_order(atlas_dir)
    markers = celltyping.model_markers(load_model(atlas_dir, order), **params["markers"])
    for name, genes in markers.items():
        log(f"marker set {name}: {len(genes)} genes")
    write_json({name: sorted(genes) for name, genes in markers.items()}, table_dir / "marker_sets_from_pseudobulk.json")

    # feature checkpoint per slide
    available = []
    for slide in slides:
        log(f"=== build {slide} ===")
        if build_slide_checkpoint(slide, args, order, markers, table_dir, checkpoint_dir):
            available.append(slide)

    if args.checkpoint_only:
        log(f"checkpoints written: {available}")
        return

    if not available:
        raise SystemExit("no slide with a decode")

    # pooled sample: quantiles of the components, precision / coverage curve, threshold
    checkpoints = {slide: np.load(checkpoint_dir / f"{slide}.features.npz") for slide in available}
    sample = celltyping.sample_rows(checkpoints, params["sample_per_slide"], params["sampling_seed"])
    for checkpoint in checkpoints.values():
        checkpoint.close()
    quantiles = celltyping.component_quantiles(sample, *params["quantiles"])
    write_json(quantiles, table_dir / "normalization_quantiles.json")

    curve = celltyping.precision_coverage(sample, quantiles, weights)
    curve.to_csv(table_dir / "precision_coverage_curve.csv", index=False)
    curve_ok = curve.dropna()
    if curve_ok.empty and args.threshold is not None:
        log(f"no secondary decodes: threshold set to --threshold {args.threshold}")
        threshold = float(args.threshold)
    else:
        threshold = celltyping.select_threshold(curve_ok, params["target_precision"], params["min_coverage"])
    chosen = curve.iloc[(curve["threshold"] - threshold).abs().argmin()]
    if curve_ok.empty:  # no secondary decodes
        baseline_precision, operating_points = float("nan"), []
    else:
        baseline_precision = float(curve_ok.sort_values("threshold").iloc[0]["precision"])
        operating_points = [
            celltyping.operating_point(curve_ok, p, params["min_coverage"]) for p in params["operating_points"]
        ]
    write_json(operating_points, table_dir / "operating_points.json")

    # certainty columns per slide and counts over all bins
    summary_rows, counts = write_slide_tables(available, checkpoint_dir, table_dir, quantiles, weights, threshold)
    pd.DataFrame(summary_rows).to_csv(table_dir / "slide_feature_summary.csv", index=False)
    n_total = counts["total"]
    report = {
        "slides": available,
        "n_eval_sample_bins": curve.attrs["n_rows"],
        "n_total_bins": int(n_total),
        "baseline_keepall_precision": baseline_precision,
        "recommended_threshold": float(threshold),
        "recommended_precision": float(chosen["precision"]),
        "recommended_coverage": float(chosen["coverage"]),
        "recommended_marker_rate": float(chosen["marker_rate"]),
        "recommended_stability_rate": float(chosen["stability_rate"]),
        "operating_points": operating_points,
        "low_cert_fraction": float(counts["low"] / n_total) if n_total else float("nan"),
        "refit_recall_fraction": float(counts["refit"] / n_total) if n_total else float("nan"),
        "overlap_lowcert_refit": float(counts["low_and_refit"] / n_total) if n_total else float("nan"),
        "positive_label": (
            "marker_supported (called-type marker UMI in 3x3) AND cross-decode stable (same call in midground)"
        ),
        "quantiles": quantiles,
        "weights": weights,
    }
    write_json(report, table_dir / "certainty_threshold_report.json")

    log(
        f"recommended threshold={threshold:.2f} "
        f"(precision={chosen['precision']:.3f}, coverage={chosen['coverage']:.3f})"
    )
    log(f"outputs: {out_dir}")


if __name__ == "__main__":
    main()
