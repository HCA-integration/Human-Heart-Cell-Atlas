#!/usr/bin/env python
"""Per-bin 2 um AnnData: the FICTURE Level-3 decode joined to the Space Ranger square_002um counts.

The join is hhca.celltyping.bins_2um:
  * one obs per unique 2 um grid key  key = round(X/2)*1e7 + round(Y/2)  of the decode (X, Y in um);
    the decode row with the highest P1 represents the bin; among tied P1 the first decode row wins;
    obs are ordered by key and named <slide>__<round(X/2)>_<round(Y/2)>
  * the same key computed from square_002um  pxl_(col,row)_in_fullres * microns_per_pixel  for the
    filtered barcodes maps a key to a SpaceRanger row (first barcode wins on a key collision);
    unmatched decode bins are dropped
  * X = raw square_002um counts of the matched row (float32 CSR), var = SpaceRanger features with
    var_names_make_unique
  * obs: L3_label (argmax K1 over the 23-type factor order), gcert (certainty_score of the chosen decode row,
    NaN without a certainty table), posterior_max (P1), x_um / y_um (decode coordinates), slide

Inputs: the pixel decode table of the slide, its square_002um folder, the Level-3 factor order and, if present,
the certainty columns of build_certainty_score.py.
Outputs: --out, one h5ad with the join statistics in uns["build"].

Usage: python build_2um_h5ad.py --slide SLIDE --decode PIXEL.tsv.gz --square-002um DIR --factor-order JSON \
           --region REGION [--certainty TSV] --out SLIDE.h5ad
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from hhca import celltyping
from hhca.log import get_logger

log = get_logger("2um")


def load_factor_order(path):
    """Level-3 type names in factor order."""
    with open(path) as handle:
        return json.load(handle)


def load_decode(path):
    """Coordinates (um), top factor and its posterior of every decode row."""
    return pd.read_csv(
        path,
        sep="\t",
        usecols=["X", "Y", "K1", "P1"],
        dtype={"X": np.float64, "Y": np.float64, "K1": np.int16, "P1": np.float64},
    )


def load_certainty(path, n_decode_rows):
    """certainty_score per decode row; the table must have one row per decode row."""
    scores = pd.read_csv(path, sep="\t", usecols=["certainty_score"])["certainty_score"].values
    if len(scores) != n_decode_rows:
        raise SystemExit(f"certainty table has {len(scores)} rows, decode {n_decode_rows}")
    return scores


def write_h5ad(adata, out):
    """Write to a partial file first, so that an interrupted job leaves no truncated h5ad under the final name."""
    partial = out + ".tmp"
    adata.write_h5ad(partial, compression="gzip")
    os.replace(partial, out)


def parse_args():
    """Command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slide", required=True, help="slide id (prefix of the obs names)")
    parser.add_argument("--decode", required=True, help="atlas_L3.decode.prj_12.r_4.0_5.0.pixel.tsv.gz")
    parser.add_argument("--square-002um", required=True, help="Space Ranger binned_outputs/square_002um folder")
    parser.add_argument("--factor-order", required=True, help="atlas_L3_factor_order.json")
    parser.add_argument(
        "--certainty", default=None, help="<slide>_certainty_columns.tsv.gz, row-aligned with the decode"
    )
    parser.add_argument("--region", required=True, help="epi, myo or endo (stored in uns)")
    parser.add_argument("--out", required=True, help="output h5ad")
    return parser.parse_args()


def main():
    """Join the decode of one slide to its square_002um counts and write the h5ad."""
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    decode = load_decode(args.decode)
    log(f"{args.slide}: decode rows {len(decode):,}")
    certainty = load_certainty(args.certainty, len(decode)) if args.certainty else None
    adata = celltyping.bins_2um(
        decode,
        celltyping.read_bins(args.square_002um),
        load_factor_order(args.factor_order),
        args.slide,
        args.region,
        certainty=certainty,
    )
    write_h5ad(adata, args.out)
    log(f"wrote {args.out}: {adata.shape}; build {adata.uns['build']}")


if __name__ == "__main__":
    main()
