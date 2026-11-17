#!/usr/bin/env python
"""Fig 4 i, j: uniform SpaceRanger 2 um substrate per vessel structure, and the pooled table.

Per slide: the SpaceRanger square_002um filtered matrix, restricted to the 18,082-feature common panel
(raw feature names, resources/vascular/common_panel_symbols.json) and collapsed over duplicated names, is
summed per 8 um lattice cell (tot = common-panel library; hhca.tl.lattice_counts); per structure the counts of
the RAAS + TF genes and the library are summed over all 2 um bins of the structure's vascular lattice cells
(hhca.tl.structure_pseudobulk). Genes: `raas` in config/vascular.yaml.

With --flags (CMFLAG-clean), SpaceRanger 2 um bins carrying any of the four spillover flags are dropped before
the 8 um aggregation; the library is still summed over the common panel.

Inputs: the SpaceRanger square_002um matrix and tissue positions, the common panel, the tables of
        vessel_structures.py and, with --flags, the CMFLAG flags of the slide.
Outputs: build: uni_<slide>_bins8.npz (gx, gy, tot, gene_counts, genes), uni_<slide>_structpb.parquet;
         pool: the per-structure tables of --slides concatenated in that order (input of raas_model.py)

Usage:
  python raas_substrate.py build --slide S --spaceranger SQUARE_002UM_DIR --common-panel PANEL.json
                                  --structures-dir DIR --factor-order L3_ORDER.json --out-dir DIR
                                  [--flags FLAGS.parquet]
  python raas_substrate.py pool  --slides S ... --dir DIR --out pooled_struct_uniform.parquet
"""

import argparse
import json
import os

from hhca import thread_env

os.environ.update(thread_env(1, numba=False, keep_existing=True))

import numpy as np
import pandas as pd

import hhca
from hhca.config import load_config
from hhca.grid import KEYMUL, spillover_keys
from hhca.log import get_logger

log = get_logger()


def build_one(slide, sr_dir, common_panel_json, structures_dir, factor_order_json, out_dir, flags=None, force=False):
    """Write the 8 um count table and the per-structure pseudobulk of one slide."""
    bins_out = f"{out_dir}/uni_{slide}_bins8.npz"
    structs_out = f"{out_dir}/uni_{slide}_structpb.parquet"
    if os.path.exists(bins_out) and os.path.exists(structs_out) and not force:
        log(f"  {slide}: outputs exist, skipped")
        return
    cfg = load_config("vascular")
    structures, raas = cfg["structures"], cfg["raas"]
    genes = raas["genes"] + raas["tf_up_in_agt_pos"] + raas["tf_up_in_rest"]
    with open(factor_order_json) as handle:
        factor_order = json.load(handle)
    with open(common_panel_json) as handle:
        panel = json.load(handle)
    structs = pd.read_csv(f"{structures_dir}/{slide}_structs.csv")
    vessels = hhca.io.read_vessels(
        f"{structures_dir}/{slide}_cells.parquet", structures["vascular_types"], factor_order
    )
    adata = hhca.io.read_visium_hd(sr_dir)
    spillover = spillover_keys(flags) if flags else None
    lattice = hhca.tl.lattice_counts(adata, genes, panel, spillover, structures["lattice_um"])
    del adata
    keys = lattice.index.values
    np.savez_compressed(
        bins_out,
        gx=(keys // KEYMUL).astype(np.int64),
        gy=(keys % KEYMUL).astype(np.int64),
        tot=lattice["tot"].values.astype(np.float64),
        gene_counts=lattice[genes].values.astype(np.float32),
        genes=np.array(genes),
    )
    log(
        f"  {slide}: {len(keys):,} 8um bins; REN+bins={(lattice['REN'].values > 0).sum()}, "
        f"AGT+bins={(lattice['AGT'].values > 0).sum()}"
    )
    pseudobulk = hhca.tl.structure_pseudobulk(lattice, vessels, structs)
    pseudobulk["slide"] = slide
    pseudobulk.to_parquet(structs_out, index=False)
    log(f"  {slide}: struct pb {len(pseudobulk):,} structs (lib>0: {(pseudobulk['lib'] > 0).sum():,})")


def pool(slides, directory, out):
    """Concatenate the per-structure tables of the slides, in the order given."""
    parts = []
    for slide in slides:
        path = f"{directory}/uni_{slide}_structpb.parquet"
        if not os.path.exists(path):
            raise FileNotFoundError(f"no per-structure table for {slide}: run raas_substrate.py build first")
        parts.append(pd.read_parquet(path))
    pooled = pd.concat(parts, ignore_index=True)
    pooled.to_parquet(out, index=False)
    log(f"pooled {len(slides)} slides: {len(pooled):,} structures -> {out}")


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="cmd", required=True)
    build = commands.add_parser("build", help="per-slide tables")
    build.add_argument("--slide", required=True, help="slide id")
    build.add_argument("--spaceranger", required=True, help="binned_outputs/square_002um directory")
    build.add_argument("--common-panel", required=True, help="JSON list of the common feature names")
    build.add_argument("--structures-dir", required=True, help="directory with the vessel-structure tables")
    build.add_argument("--factor-order", required=True, help="JSON list of the Level-3 types in factor order")
    build.add_argument("--out-dir", required=True, help="output directory")
    build.add_argument("--flags", default=None, help="CMFLAG flags of the slide (<slide>_clean_bins.parquet)")
    build.add_argument("--force", action="store_true", help="rebuild even if both outputs exist")
    pooling = commands.add_parser("pool", help="concatenate the per-structure tables")
    pooling.add_argument("--slides", nargs="+", required=True, help="slide ids, in row order")
    pooling.add_argument("--dir", required=True, help="directory with the per-slide tables")
    pooling.add_argument("--out", required=True, help="output parquet")
    return parser.parse_args()


def main():
    """Build the tables of one slide, or pool the per-structure tables."""
    args = parse_args()
    if args.cmd == "build":
        os.makedirs(args.out_dir, exist_ok=True)
        log(f"building the common-panel tables of {args.slide}")
        build_one(
            args.slide,
            args.spaceranger,
            args.common_panel,
            args.structures_dir,
            args.factor_order,
            args.out_dir,
            args.flags,
            args.force,
        )
    else:
        pool(args.slides, args.dir, args.out)


if __name__ == "__main__":
    main()
