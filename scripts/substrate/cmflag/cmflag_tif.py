#!/usr/bin/env python
"""CMFLAG for one Visium HD slide with a microscopy H&E TIFF (the 8 Berlin slides).

Flags decoded 2 um bins whose transcripts spilled from a neighbouring cell into empty space: threshold-gated
type assignment, spillover flags from the decay of marker density into whitespace, and the lumen of large
vessels. Whitespace, eosin and haematoxylin are sampled per bin from the microscopy H&E (SIFT + RANSAC + ECC
registration plus an offset field), not from the coarse CytAssist image.

Steps (hhca.celltyping; thresholds and marker lists in config/substrate_cmflag.yaml):
  registration   register_he         SIFT + RANSAC affine, ECC refinement (microscopy -> CytAssist image);
                                     um -> microscopy pixel map M0, t0 saved in <slide>_align_grids.npz
  H&E per bin    sample_he           offset field (phase correlation on a 300 um lattice, bilinear per bin), then
                                     grey, eosin and haematoxylin OD (CytAssist grey outside the TIFF)
  whitespace     grey > 235, or inside the TIFF with eosin OD < 0.008
  assignment     assign_labels       certainty score against per-type thresholds (top factor without a
                                     certainty checkpoint)
  nucleus gate   nucleus_gate        bins with a nucleus (haematoxylin OD) are never flagged
  spillover      spillover_flags     bin of the type, expected spillover share >= 0.80, whitespace, no marker
                                     of a competing type
  lumen          vessel_lumen        whitespace (grey > 220) inside the filled outline of a large vessel

Inputs: the pixel decode, certainty checkpoint, Space Ranger outs (binned_outputs/square_002um and square_008um),
microscopy TIFF, factor order, certainty quantiles, per-type thresholds, decay fits, vessel tables.
Outputs: <outdir>/<slide>_clean_bins.parquet (flags per decode row), <slide>_counts.json, and the registration
(--reg-cache, read if present unless --force-reg).

Usage: python cmflag_tif.py SLIDE --decode PIXEL.tsv.gz [--gcert-checkpoint NPZ] --spaceranger OUTS --tif HE.tif \
           --factor-order JSON --quantiles JSON --tau JSON --cm-fit JSON --vsmc-fit JSON \
           [--structs CSV --cells PARQUET] --reg-cache NPZ --outdir DIR [--force] [--force-reg]
"""

import argparse
import json
import logging
import os
import time

from hhca import thread_env

# BLAS on four threads unless the caller sets these
os.environ.update(thread_env(4, keep_existing=True))

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from hhca import celltyping
from hhca.config import load_config
from hhca.log import get_logger

Image.MAX_IMAGE_PIXELS = None

log = get_logger("cmflag")

FACTOR_COLUMNS = ["K1", "K2", "K3", "K4", "K5"]
POSTERIOR_COLUMNS = ["P1", "P2", "P3", "P4", "P5"]


def parse_args(argv=None):
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("slide", help="slide id")
    parser.add_argument("--decode", required=True, help="atlas_L3.decode.prj_12.r_4.0_5.0.pixel.tsv.gz of the slide")
    parser.add_argument(
        "--gcert-checkpoint",
        default="",
        help="certainty checkpoint <slide>.features.npz (without it the top factor is used)",
    )
    parser.add_argument(
        "--spaceranger", required=True, help="Space Ranger <slide>/outs (binned_outputs/square_002um and square_008um)"
    )
    parser.add_argument("--tif", required=True, help="microscopy H&E TIFF")
    parser.add_argument("--factor-order", required=True, help="atlas_L3_factor_order.json")
    parser.add_argument("--quantiles", required=True, help="normalization_quantiles.json of the certainty score")
    parser.add_argument("--tau", required=True, help="pertype_tau_recommended.json")
    parser.add_argument(
        "--cm-fit", required=True, help="decay fit of cardiomyocyte markers into whitespace (A, lam, b)"
    )
    parser.add_argument("--vsmc-fit", required=True, help="decay fit of VSMC markers into whitespace (A, lam, b)")
    parser.add_argument("--structs", default="", help="vessel structures <slide>_structs.csv (for the lumen)")
    parser.add_argument("--cells", default="", help="member cells of the vessel structures <slide>_cells.parquet")
    parser.add_argument("--reg-cache", required=True, help="registration .npz (read if present, else written)")
    parser.add_argument("--outdir", required=True, help="folder for <slide>_clean_bins.parquet and <slide>_counts.json")
    parser.add_argument("--force", action="store_true", help="overwrite existing flags")
    parser.add_argument("--force-reg", action="store_true", help="recompute the registration")
    return parser.parse_args(argv)


def read_json(path):
    """Content of a JSON file."""
    with open(path) as handle:
        return json.load(handle)


def load_decode_assign(decode_path, checkpoint, order, quantiles, tau, weights, p_floor):
    """X, Y and assigned type of every decode row.

    With a certainty checkpoint the type is assigned against the per-type thresholds, otherwise it is the top
    factor.
    """
    if os.path.exists(checkpoint):
        decode = pd.read_csv(
            decode_path,
            sep="\t",
            usecols=["X", "Y", *FACTOR_COLUMNS, *POSTERIOR_COLUMNS],
            dtype=dict.fromkeys(FACTOR_COLUMNS, np.int16) | dict.fromkeys(POSTERIOR_COLUMNS, np.float32),
        )
        ck = np.load(checkpoint)
        assert len(ck["k1"]) == len(decode) and (ck["k1"].astype(np.int64) == decode["K1"].to_numpy(np.int64)).all(), (
            "checkpoint and decode rows differ"
        )
        score = celltyping.certainty_score(ck, quantiles, weights)
        assign = celltyping.assign_labels(decode, score, tau, p_floor=p_floor).astype(np.int64)
        log(f"assignment with thresholds n={len(decode):,} unassigned%={100 * (assign < 0).mean():.2f}")
    else:
        decode = pd.read_csv(decode_path, sep="\t", usecols=["X", "Y", "K1"], dtype={"K1": np.int16})
        assign = decode["K1"].to_numpy(np.int64)
        log(f"assignment = top factor (no certainty checkpoint) n={len(decode):,}")
    return decode["X"].to_numpy(np.float64), decode["Y"].to_numpy(np.float64), assign


def main():
    """Register the H&E of one slide, flag its decoded bins and write the flag table and counts."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[cmflag] %(message)s")
    cfg = load_config("substrate_cmflag")
    weights = load_config("substrate")["certainty"]["weights"]
    cuts = cfg["whitespace"]
    order = read_json(args.factor_order)
    tau = celltyping.flag_thresholds(
        read_json(args.tau), order, cfg["assignment"]["unscaled"], cfg["assignment"]["rare_tau_scale"]
    )
    os.makedirs(args.outdir, exist_ok=True)
    slide = args.slide
    flags_path = f"{args.outdir}/{slide}_clean_bins.parquet"
    if os.path.exists(flags_path) and not args.force:
        log(f"skip {slide}: flags exist")
        return
    start = time.time()
    square_002um = f"{args.spaceranger}/binned_outputs/square_002um"
    square_008um = f"{args.spaceranger}/binned_outputs/square_008um"
    cyt = f"{square_008um}/spatial/tissue_hires_image.png"
    scalefactors_8um = f"{square_008um}/spatial/scalefactors_json.json"
    inputs = {
        "decode": args.decode,
        "tif": args.tif,
        "cyt": cyt,
        "sf8": scalefactors_8um,
        "h5": f"{square_002um}/filtered_feature_bc_matrix.h5",
        "tp": f"{square_002um}/spatial/tissue_positions.parquet",
        "sf2": f"{square_002um}/spatial/scalefactors_json.json",
    }
    for name, path in inputs.items():
        if not os.path.exists(path):
            log(f"missing input {name}: {path}")
    has_checkpoint = os.path.exists(args.gcert_checkpoint)
    CMPP, mpp_full = celltyping.cytassist_scale(scalefactors_8um)
    log(f"=== {slide}  hires um/px={CMPP:.4f}  certainty checkpoint={has_checkpoint} ===")
    M0, t0, reg = celltyping.register_he(slide, args.tif, cyt, CMPP, args.reg_cache, args.force_reg)
    X, Y, assign = load_decode_assign(
        args.decode, args.gcert_checkpoint, order, read_json(args.quantiles), tau, weights, cfg["assignment"]["p_floor"]
    )
    mic = celltyping.Microscopy(
        tifffile.memmap(args.tif, mode="r"), np.asarray(Image.open(cyt).convert("RGB")), (M0, t0), CMPP
    )
    he = celltyping.sample_he(mic, X, Y)
    eod, hema, cov = he["eosin_od"].to_numpy(), he["hema_od"].to_numpy(), he["covered"].to_numpy()
    grey_eff = he["grey"].to_numpy()
    white_bin = (grey_eff > cuts["grey"]) | (cov & (eod < cuts["eosin_od"]))
    resident, gate = celltyping.nucleus_gate(hema, cov, white_bin, cfg["nucleus_fraction"])
    marker = celltyping.marker_depth(celltyping.read_bins(square_002um), X, Y, white_bin, cfg["markers"])
    fits = {"cm": read_json(args.cm_fit), "vsmc": read_json(args.vsmc_fit)}
    flags = celltyping.spillover_flags(
        assign, order, marker, white_bin, resident, cfg["flags"], fits, cfg["min_spillover_share"]
    )
    lumen = celltyping.vessel_lumen(X, Y, grey_eff > cuts["lumen_grey"], args.structs, args.cells)
    table = celltyping.flag_table(X, Y, assign, flags, lumen, resident, grey_eff, eod, hema, cov)
    table.to_parquet(flags_path, index=False)
    n = len(X)
    counts = {
        "slide": slide,
        "tier": "full_real_HE",
        "n_bins": int(n),
        "assign_source": ("tau_gated" if has_checkpoint else "K1_argmax"),
        "whitespace_source": "microscopy_HE_registered",
        **gate,
        "registration": reg,
        "cmpp_hires": CMPP,
        "mpp_full": mpp_full,
        "micro_coverage": float(cov.mean()),
        "real_white_frac": float(white_bin.mean()),
        "field_kept": he.attrs["field"]["nkeep"],
        "field_cells": he.attrs["field"]["ncells"],
        "pct_excluded_total": celltyping.percent(table["exclude_spillover"].to_numpy().sum(), n),
        **celltyping.flag_counts(flags, lumen),
        "cm_lambda": fits["cm"]["lam"],
        "vsmc_lambda": fits["vsmc"]["lam"],
        "F_HI": cfg["min_spillover_share"],
        "WHITE_CUT": cuts["grey"],
        "EOD_MIN": cuts["eosin_od"],
    }
    with open(f"{args.outdir}/{slide}_counts.json", "w") as handle:
        json.dump(counts, handle, indent=2, default=float)
    log(f"wrote {flags_path} and counts.json  ({time.time() - start:.0f}s)  NCC={reg.get('ncc')}")


if __name__ == "__main__":
    main()
