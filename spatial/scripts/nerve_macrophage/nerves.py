#!/usr/bin/env python
"""Call the nerves of one slide and sum their counts (nerve definition: config/nerve_macrophage.yaml `nerve`).

  unit      one 2 um bin = one obs of the CMFLAG-clean 2 um h5ad (label = Level 3 of its max-P1 decode row)
  nerve     connected component of neural bins at 6 um single linkage with >= 3 bins (hhca.tl.call_nerves:
            DBSCAN(eps=6, min_samples=3) on the unique bins; nerves.json states whether this partition equals the
            single-linkage components of >= 3 bins)
  counts    sum of the SpaceRanger 2 um counts (h5ad X) of the member bins; no halo (hhca.tl.nerve_pseudobulk)
  n_bins    number of member 2 um bins
  myelin    myelin_total = sum of MBP, PLP1, MOG, MPZ, PMP22 counts of the member bins
  vessel    d_vessel_um = smallest distance from a member bin to a vascular bin (the 5 vascular types of Fig 5 d)

Inputs: the CMFLAG-clean 2 um h5ad of the slide.
Outputs: <slide>_nerve_pseudobulk.h5ad (obs nerve, n_bins, myelin_total, total_counts, slide, donor, region, cx, cy,
d_vessel_um; var = the slide's feature panel; layers counts), <slide>_neural_bins.csv.gz (x_um, y_um, nerve,
comp_size) and <slide>_nerves.json.

Usage: python nerves.py --slide S --h5ad CLEAN.h5ad --outdir results/nerve_macrophage/nerves
"""

import argparse
import json
import os

from hhca import config_path, thread_env

os.environ.update(thread_env(1, numba=False))

import anndata as ad
import numpy as np

import hhca
from hhca.config import load_config


def region_of(slide):
    """Section region (myo, epi, endo) from the slide id."""
    region = slide.split("_")[-1]
    return region if region in ("myo", "epi", "endo") else "?"


def donor_of(slide):
    """Donor id from the slide id."""
    donor = slide.replace("VS_", "").replace("_henricke", "")
    return "_".join(donor.split("_")[:-1]) if region_of(slide) != "?" else donor


def nerves_are_components(neural_bins, min_bins):
    """Whether the nerves are exactly the groups of at least `min_bins` linked neural bins."""
    nerve, comp_size = neural_bins["nerve"].to_numpy(), neural_bins["comp_size"].to_numpy()
    in_nerve = nerve >= 0
    same = bool(np.array_equal(in_nerve, comp_size >= min_bins))
    return same and bool((np.bincount(nerve[in_nerve])[nerve[in_nerve]] == comp_size[in_nerve]).all())


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slide", required=True, help="slide id")
    parser.add_argument("--h5ad", required=True, help="CMFLAG-clean 2 um h5ad of the slide")
    parser.add_argument("--outdir", required=True, help="output folder")
    parser.add_argument(
        "--config", help="pipeline config with the nerve definition", default=config_path("nerve_macrophage")
    )
    return parser.parse_args()


def main():
    """Call the nerves of the slide and write the pseudobulk, the neural-bin table and the summary."""
    args = parse_args()
    cfg = load_config(args.config)
    nerve_cfg = cfg["nerve"]
    vascular_types = cfg["proximity"]["vascular_types"]
    os.makedirs(args.outdir, exist_ok=True)
    slide = args.slide

    adata = ad.read_h5ad(args.h5ad, backed="r")
    neural_bins = hhca.tl.call_nerves(adata, nerve_cfg["neural_type"], nerve_cfg["linkage_um"], nerve_cfg["min_bins"])
    same = nerves_are_components(neural_bins, nerve_cfg["min_bins"])
    nerves = hhca.tl.nerve_pseudobulk(adata, neural_bins, nerve_cfg["myelin_genes"], vascular_types)
    print(
        f"{slide}: {adata.n_obs:,} bins, {len(neural_bins):,} neural, {nerves.n_obs} nerves "
        f"(>= {nerve_cfg['min_bins']} bins), {int((neural_bins.nerve >= 0).sum()):,} neural bins in nerves; "
        f"DBSCAN == components: {same}",
        flush=True,
    )

    nerves.obs.insert(4, "slide", slide)
    nerves.obs.insert(5, "donor", donor_of(slide))
    nerves.obs.insert(6, "region", region_of(slide))
    nerves.obs_names = [f"{slide}__{i}" for i in range(nerves.n_obs)]
    nerves.uns["nerve_definition"] = {
        k: (v if not isinstance(v, list) else ",".join(map(str, v))) for k, v in nerve_cfg.items()
    }
    nerves.write_h5ad(f"{args.outdir}/{slide}_nerve_pseudobulk.h5ad")
    neural_bins.to_csv(f"{args.outdir}/{slide}_neural_bins.csv.gz", index=False)
    summary = {
        "slide": slide,
        "n_bins": int(adata.n_obs),
        "n_neural": int(len(neural_bins)),
        "n_vascular": int(adata.obs["L3_label"].isin(vascular_types).sum()),
        "n_nerves": int(nerves.n_obs),
        "n_neural_in_nerves": int((neural_bins.nerve >= 0).sum()),
        "dbscan_equals_components": same,
        "n_myelinated": int((nerves.obs.myelin_total >= nerve_cfg["myelinated_min"]).sum()),
        "n_genes": int(nerves.n_vars),
        "myelin_genes_on_panel": [g for g in nerve_cfg["myelin_genes"] if g in set(nerves.var_names)],
    }
    with open(f"{args.outdir}/{slide}_nerves.json", "w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
