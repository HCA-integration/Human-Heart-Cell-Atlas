"""CMFLAG-clean 2 µm and 8 µm h5ads of one Visium HD slide.

Inputs:
  --h5ad-2um  <slide>.h5ad                FICTURE Level 3, one row per decoded 2 µm bin
  --h5ad-8um  <slide>.h5ad                the same decode aggregated to 8 µm
  --flags     <slide>_clean_bins.parquet  CMFLAG flags per decode row (or per 2 µm bin)
Outputs: <out-2um>/<slide>.h5ad, <out-8um>/<slide>.h5ad, <summary-dir>/<slide>_summary.json

Only the four spillover flags (cardiomyocyte, VSMC, arterial EC, venous EC) remove bins; the nucleus, lumen
and coverage columns of the flag table are not used. The flags are collapsed to the 2 um bins by
hhca.celltyping.bin_flags and the 8 um bins are rebuilt by hhca.celltyping.clean_bins_8um.
2 µm clean: the 2 µm h5ad without flagged bins.
8 µm clean: counts = 8 µm counts minus the counts of the flagged 2 µm bins of the block; label, composition,
certainty and bin count from the unflagged 2 µm bins. 8 µm bins whose decoded 2 µm bins are all flagged are
dropped.

Usage: python build_cmflag_clean_h5ads.py SLIDE --h5ad-2um H5AD --h5ad-8um H5AD --flags PARQUET \
           --out-2um DIR --out-8um DIR --summary-dir DIR
"""

import argparse
import json
import os
import time

import anndata as ad
import pandas as pd

from hhca import celltyping
from hhca.grid import SPILLOVER_FLAGS


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("slide", help="slide id")
    parser.add_argument("--h5ad-2um", required=True, help="2 um h5ad of the slide")
    parser.add_argument("--h5ad-8um", required=True, help="8 um h5ad of the slide")
    parser.add_argument("--flags", required=True, help="<slide>_clean_bins.parquet")
    parser.add_argument("--out-2um", required=True, help="folder for the clean 2 um h5ad")
    parser.add_argument("--out-8um", required=True, help="folder for the clean 8 um h5ad")
    parser.add_argument("--summary-dir", required=True, help="folder for <slide>_summary.json")
    return parser.parse_args()


def write_atomic(adata, path):
    """Write to a temporary file, then rename."""
    partial = path + ".part"
    adata.write_h5ad(partial, compression="gzip")
    os.replace(partial, path)


def main():
    """Remove the flagged bins of one slide from its 2 µm and 8 µm h5ads and write the summary."""
    args = parse_args()
    for folder in (args.out_2um, args.out_8um, args.summary_dir):
        os.makedirs(folder, exist_ok=True)
    slide = args.slide
    start = time.time()
    a2 = ad.read_h5ad(args.h5ad_2um)
    a8 = ad.read_h5ad(args.h5ad_8um)
    print(f"{slide}: loaded 2um {a2.shape} 8um {a8.shape} in {time.time() - start:.0f}s", flush=True)

    flags = celltyping.bin_flags(pd.read_parquet(args.flags, columns=["X", "Y"] + SPILLOVER_FLAGS), a2)
    flagged = flags.to_numpy().any(axis=1)
    c8, stats = celltyping.clean_bins_8um(a2, a8, flagged)

    summary = {
        "slide": slide,
        "n_2um": int(a2.n_obs),
        "n_2um_flagged": int(flagged.sum()),
        "pct_2um_flagged": round(100 * float(flagged.mean()), 3),
        **{f"n_{flag}": int(flags[flag].sum()) for flag in SPILLOVER_FLAGS},
        "n_2um_clean": int((~flagged).sum()),
        "n_8um": stats["n_8um"],
        "n_8um_dropped_all_flagged": stats["n_8um_dropped_all_flagged"],
        "n_8um_clean": stats["n_8um_clean"],
        "pct_8um_counts_removed": stats["pct_8um_counts_removed"],
        "frac_2um_in_flag_table": round(flags.attrs["frac_in_flag_table"], 5),
        "frac_2um_assigned_to_8um": stats["frac_2um_assigned_to_8um"],
        "frac_full_blocks_exact_sum": stats["frac_full_blocks_exact_sum"],
        "n_8um_negative_entries_clipped": stats["n_8um_negative_entries_clipped"],
        "pct_8um_counts_clipped": stats["pct_8um_counts_clipped"],
    }
    cmflag_info = {
        "flags_removed": SPILLOVER_FLAGS,
        "flag_source": args.flags,
        "flag_join": (
            "parquet rows collapsed to one row per 2 um bin by round(um/2) grid key, flags OR-ed within a bin"
        ),
        "built": time.strftime("%Y-%m-%d"),
        "script": "scripts/substrate/cmflag/build_cmflag_clean_h5ads.py",
        "summary": json.dumps(summary),
    }

    # 8 µm clean
    c8.uns["cmflag"] = {
        **cmflag_info,
        "x": "8 um counts minus counts of flagged 2 um bins in the block (negatives clipped to 0)",
        "labels": "L3_label / L3_top1_frac / L3_composition / gcert_mean / n_2um_bins from unflagged 2 um bins only",
        "dropped": "8 um bins whose decoded 2 um bins are all flagged",
        "block_assignment": "nearest 8 um bin centre in physical microns (<= 4.6 um)",
    }
    write_atomic(c8, f"{args.out_8um}/{slide}.h5ad")
    print(f"{slide}: wrote 8um clean {c8.shape} at {time.time() - start:.0f}s", flush=True)
    del c8

    # 2 µm clean
    c2 = a2[~flagged].copy()
    c2.uns["cmflag"] = cmflag_info
    write_atomic(c2, f"{args.out_2um}/{slide}.h5ad")
    print(f"{slide}: wrote 2um clean {c2.shape} at {time.time() - start:.0f}s", flush=True)

    with open(f"{args.summary_dir}/{slide}_summary.json", "w") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
