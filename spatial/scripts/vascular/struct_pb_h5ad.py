#!/usr/bin/env python
"""Fig 4 k: CMFLAG-clean per-vessel-structure pseudobulk from the 2 um FICTURE h5ad of one slide.

Per slide every 2 um bin is mapped to its 8 um lattice cell and hence to a vessel structure, and summed per
structure with and without the bins carrying any of the four spillover flags: total library, library over the
11-slide common gene set, bin count, and the counts of the RAAS and drug-target genes (columns *_clean and
*_raw; hhca.tl.pseudobulk_structures). Genes: `raas.genes` and `drug.classes` in config/vascular.yaml.

Inputs:
  * 2 um bin counts          <slide>.h5ad (bins x genes, obs x_um/y_um)
  * spillover flags          <slide>_clean_bins.parquet (CMFLAG)
  * structure membership     <slide>_cells.parquet of the vessel-structure step (vessel_structures.py)
  * common gene set          intersection of the var names of the two --common-from h5ads
                             (VS_BO_2024BH-008_myo and VS_AH9_LV_FW_epi; library `libc`)
Outputs: struct_pb_cmflag_clean_<slide>.parquet per slide (--out), and struct_pb_cmflag_clean.parquet with the
         structures of all slides (--out of the --pool call)

Usage: python struct_pb_h5ad.py --slide S --h5ad S.h5ad --flags S_clean_bins.parquet --cells S_cells.parquet
           --factor-order L3_ORDER.json --common-from A.h5ad B.h5ad --out OUT.parquet
       python struct_pb_h5ad.py --pool OUT_DIR --slides S ... --out struct_pb_cmflag_clean.parquet
"""

from __future__ import annotations

import argparse
import json
import os

from hhca import thread_env

os.environ.update(thread_env(1, numba=False, keep_existing=True))

import anndata as ad
import pandas as pd

import hhca
from hhca.config import load_config
from hhca.grid import spillover_keys
from hhca.log import get_logger

log = get_logger()


def pool(directory: str, slides, out: str) -> None:
    """Concatenate the per-slide tables in the order of `slides`."""
    pooled = pd.concat(
        [pd.read_parquet(f"{directory}/struct_pb_cmflag_clean_{slide}.parquet") for slide in slides],
        ignore_index=True,
    )
    pooled.to_parquet(out, index=False)
    log(f"wrote {len(pooled):,} structures -> {out}")


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slide", help="slide id")
    parser.add_argument("--h5ad", help="2 um h5ad of the slide")
    parser.add_argument("--flags", help="CMFLAG flags of the slide (<slide>_clean_bins.parquet)")
    parser.add_argument("--cells", help="<slide>_cells.parquet of vessel_structures.py")
    parser.add_argument("--factor-order", help="JSON list of the Level-3 types in factor order")
    parser.add_argument("--common-from", nargs=2, metavar="H5AD", help="two h5ads whose shared genes define `libc`")
    parser.add_argument(
        "--pool",
        default=None,
        metavar="DIR",
        help="concatenate DIR/struct_pb_cmflag_clean_<slide>.parquet of --slides instead",
    )
    parser.add_argument("--slides", nargs="*", help="slide ids to pool, in row order")
    parser.add_argument("--out", required=True, help="output parquet")
    return parser.parse_args()


def main():
    """Write the pseudobulk of one slide, or pool the per-slide tables with --pool."""
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.pool:
        pool(args.pool, args.slides, args.out)
        return
    if os.path.exists(args.out):
        log(f"{args.slide}: output exists, skipped")
        return
    cfg = load_config("vascular")
    structures, drug = cfg["structures"], cfg["drug"]
    targets = [gene for drug_class in drug["classes"].values() for gene in drug_class["genes"]]
    genes = list(dict.fromkeys(cfg["raas"]["genes"] + targets))
    with open(args.factor_order) as handle:
        factor_order = json.load(handle)
    first, second = (ad.read_h5ad(path, backed="r") for path in args.common_from)
    common = set(first.var_names) & set(second.var_names)
    log(f"genes shared by the two panels: {len(common)}")

    adata = ad.read_h5ad(args.h5ad, backed="r")
    vessels = hhca.io.read_vessels(args.cells, structures["vascular_types"], factor_order)
    per_struct = hhca.tl.pseudobulk_structures(
        adata, vessels, spillover_keys(args.flags), genes, common, structures["lattice_um"], drug["chunk_bins"]
    )
    per_struct.insert(0, "slide", args.slide)
    log(
        f"{args.slide}: {len(per_struct):,} structures; lib_clean>0 in {(per_struct.lib_clean > 0).mean():.4%}; "
        f"library kept after mask {per_struct.lib_clean.sum() / per_struct.lib_raw.sum():.3%}"
    )
    per_struct.to_parquet(args.out + ".part", index=False)
    os.replace(args.out + ".part", args.out)


if __name__ == "__main__":
    main()
