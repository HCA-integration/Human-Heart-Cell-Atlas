#!/usr/bin/env python
"""hormone2cell producer and receiver strength per Level-4 cell type, stratified by donor subgroup.

For each subgroup (all, male, female, LV, LV by sex, LV female below / above the median age;
hhca.tl.donor_subgroups):
  * producing side: hormone2cell.hormone_strength(producing gene sets, adjustment=False, assay='cell',
    use_precomputed=None) on the Level-4 pseudobulk restricted to the FICTURE gene panel (the 14,016 genes of
    the L4 atlas profile), the same method as the all-donor table of hormone2cell_producer_L4.py
    (hhca.tl.producer_strength);
  * receiving side: hormone2cell.hormone_strength(receptor gene sets, adjustment=False,
    assay='nucleus_only', use_precomputed=None) on the Level-4 pseudobulk over all atlas genes (most receptor
    genes are outside the panel), with the measured share of expressing nuclei as detection rate
    (hhca.tl.receiver_strength).

Pseudobulk = mean per-cell CP10k per `Final Level 4_harmonized` type (Unassigned dropped, 100 types), from
layers/counts read in row blocks with the full library as denominator (hhca.tl.subgroup_pseudobulk). One pass
over the atlas accumulates all subgroups and is checkpointed, so an interrupted run resumes; scoring is then done
per subgroup. Read errors are retried. Subgroups pool cells, so duplicate donor ids do not matter here (only the
median-age split of the LV female donors counts donors). Settings: config/atlas_hormones.yaml (stratified).

Inputs: atlas h5ad, L4 atlas profile (gene panel and Level-4 order).
Outputs: producer_strength_L4_long__<subgroup>.csv, hrc_datadriven_hormones_level_4__<subgroup>.csv,
subgroup_counts.csv.

Usage: python hormone2cell_stratified.py --atlas ATLAS.h5ad --profile atlas_L4_pseudobulk.tsv \
           --outdir results/atlas/hormones/data_stratified --ckptdir results/atlas/hormones/_ckpt_stratified
"""

import argparse
import os

from hhca import thread_env

os.environ.update(thread_env(1, numba=False))

import hhca
from hhca.config import load_config
from hhca.log import get_logger
from hhca.tl._hormones import NOT_A_TYPE


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--atlas", required=True, help="HHCA atlas h5ad (layers/counts, obs)")
    parser.add_argument(
        "--profile", required=True, help="Level-4 pseudobulk table (genes x types): gene panel and Level-4 order"
    )
    parser.add_argument("--outdir", required=True, help="directory for the tables")
    parser.add_argument(
        "--ckptdir",
        required=True,
        help="directory for the streaming checkpoint and hormone2cell's max-expression files",
    )
    return parser.parse_args()


def main():
    """Accumulate the pseudobulk of every donor subgroup and score producers and receivers per subgroup."""
    args = parse_args()
    log = get_logger()
    params = load_config("atlas_hormones")["stratified"]
    for folder in (args.outdir, args.ckptdir):
        os.makedirs(folder, exist_ok=True)

    log("reading obs ...")
    obs = hhca.tl.read_obs(args.atlas, [params["level"], "donor_id", "sex", "development_stage", "tissue_general"])
    level4 = obs[params["level"]].astype(str).values
    level4_types, panel_genes = hhca.tl.read_profile_header(args.profile)
    assert set(level4_types) == (set(level4) - NOT_A_TYPE), "L4 types of the atlas differ from the L4 atlas profile"
    log(f"n_cells={obs.shape[0]:,}  n_L4={len(level4_types)}")

    subgroups, counts, cutoff = hhca.tl.donor_subgroups(obs)
    log(f"age cut-off (median age of the LV female donors) = {cutoff}")
    counts.to_csv(args.outdir + "/subgroup_counts.csv", index=False)
    log("subgroup counts:\n" + counts.to_string(index=False))

    # genes of the producing and receptor definitions
    producing, receptors = hhca.tl.hormone_definitions()
    producer_genes, receptor_genes = hhca.tl.hormone_genes(producing, receptors)
    pseudobulk = hhca.tl.subgroup_pseudobulk(
        args.atlas,
        subgroups,
        level4,
        level4_types,
        sorted(set(producer_genes) | set(receptor_genes)),
        layer=params["layer"],
        block=params["block"],
        checkpoint_dir=args.ckptdir,
        every=params["checkpoint_every"],
        log=log,
    )

    for name in subgroups:
        producer_out = args.outdir + f"/producer_strength_L4_long__{name}.csv"
        receiver_out = args.outdir + f"/hrc_datadriven_hormones_level_4__{name}.csv"
        if os.path.exists(producer_out) and os.path.exists(receiver_out):
            log(f"[{name}] already scored, skipping")
            continue
        producers = hhca.tl.producer_strength(
            pseudobulk[name],
            producing,
            genes=[g for g in producer_genes if g in panel_genes],
            max_expression_file=args.ckptdir + f"/gene_max_expression_producer__{name}.csv",
        )
        producers.to_csv(producer_out, index=False)
        log(f"[{name}] producer pairs={len(producers)} hormones={producers['Hormone'].nunique()} -> {producer_out}")

        receivers = hhca.tl.receiver_strength(
            pseudobulk[name],
            receptors,
            genes=receptor_genes,
            max_expression_file=args.ckptdir + f"/gene_max_expression_receptor__{name}.csv",
        )
        receivers.to_csv(receiver_out)  # with the index column, the layout of hormone2cell's hrc_* tables
        log(f"[{name}] receiver pairs={len(receivers)} hormones={receivers['Hormone'].nunique()} -> {receiver_out}")

    log("all subgroups done")


if __name__ == "__main__":
    main()
