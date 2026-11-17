#!/usr/bin/env python
"""8 um AnnData aggregated from the FICTURE 2 um Level-3 decode.

The aggregation is hhca.celltyping.bins_8um:
  * decode collapsed to one row per unique 2 um grid key round(X/2)*1e7 + round(Y/2) (max P1, first row on
    ties), as for the 2 um h5ad
  * key -> square_002um bin through the same grid key computed for ALL rows of tissue_positions.parquet
    (pxl_*_in_fullres * microns_per_pixel; first row wins), giving (array_row, array_col);
    8 um block r8 = array_row // 4, c8 = array_col // 4
  * 8 um universe: Berlin slides = square_008um filtered barcodes (h5 order), X = square_008um counts,
    x_um / y_um = square_008um pxl * microns_per_pixel. AH slides (no square_008um) = every 4x4 block with
    >= 1 square_002um bin, sorted by (r8, c8), X = 4x4 sum of square_002um counts, x_um / y_um = mean of the
    constituent 2 um bin coordinates (hhca.celltyping.aggregate_bins)
  * obs: L3_label = majority L3 over the block's 2 um bins (first type on ties; NaN without bins),
    L3_top1_frac, gcert_mean (mean float32 certainty_score; NaN for AH), n_2um_bins, x_um, y_um, slide;
    obsm L3_composition (n8 x 23 fractions) and spatial

Inputs: the pixel decode table of the slide, its square_002um folder (and square_008um folder for the Berlin
slides), the Level-3 factor order and, if present, the certainty columns of build_certainty_score.py.
Outputs: --out, one h5ad.

Usage: python build_8um_h5ad.py --slide SLIDE --decode PIXEL.tsv.gz --square-002um DIR [--square-008um DIR] \
           --factor-order JSON [--certainty TSV] --out SLIDE.h5ad
"""

import argparse
import os

from build_2um_h5ad import load_certainty, load_decode, load_factor_order, write_h5ad
from hhca import celltyping
from hhca.log import get_logger

log = get_logger("8um")


def parse_args():
    """Command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slide", required=True, help="slide id (prefix of the obs names)")
    parser.add_argument("--decode", required=True, help="atlas_L3.decode.prj_12.r_4.0_5.0.pixel.tsv.gz")
    parser.add_argument("--square-002um", required=True, help="Space Ranger binned_outputs/square_002um folder")
    parser.add_argument("--square-008um", default=None, help="Space Ranger square_008um folder (Berlin slides)")
    parser.add_argument("--factor-order", required=True, help="atlas_L3_factor_order.json")
    parser.add_argument(
        "--certainty", default=None, help="<slide>_certainty_columns.tsv.gz, row-aligned with the decode"
    )
    parser.add_argument(
        "--decode-label", default=None, help="decode path written to uns['build']['decode'] (default: --decode)"
    )
    parser.add_argument("--out", required=True, help="output h5ad")
    return parser.parse_args()


def main():
    """Aggregate the 2 um decode of one slide to 8 um bins and write the h5ad."""
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    decode = load_decode(args.decode)
    log(f"{args.slide}: decode rows {len(decode):,}")
    certainty = load_certainty(args.certainty, len(decode)) if args.certainty else None
    if args.square_008um:
        bins = celltyping.read_bins(args.square_008um)
    else:
        bins = celltyping.aggregate_bins(celltyping.read_bins(args.square_002um))
    adata = celltyping.bins_8um(
        decode,
        celltyping.read_bin_positions(args.square_002um),
        bins,
        load_factor_order(args.factor_order),
        args.slide,
        certainty=certainty,
        decode_label=args.decode_label or args.decode,
    )
    log(f"8um bins {adata.n_obs:,}; covered {(adata.obs['n_2um_bins'] > 0).sum():,}")
    write_h5ad(adata, args.out)
    log(f"wrote {args.out}: {adata.shape}")


if __name__ == "__main__":
    main()
