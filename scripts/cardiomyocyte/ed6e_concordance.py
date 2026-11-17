"""ED6 e: concordance between the atlas-guided Visium HD FICTURE factors and the snRNA Level-3 atlas marker genes
(config ed6e; the panel draws the variant of config panels.ed6e_variant).

Definition:
  markers   for each atlas L3 type (Unassigned excluded), the top N genes by log2((mean_t + pc) / (max other type
            mean + pc)) among genes with mean_t >= marker_min_mean and present in the FICTURE gene set; union
            (hhca.tl.atlas_markers).
  profiles  each profile (atlas type, or FICTURE factor) normalised to 1e4 over the shared genes, log1p, then each
            marker gene z-scored across the 23 types/factors (specificity space).
  score     Pearson r(factor f, atlas type t) over the markers (hhca.tl.factor_concordance); self-match =
            argmax_t r(f, t) == f; diagonal mean r (hhca.tl.concordance_summary).
FICTURE factor profiles, three variants:
  posterior   decode posterior counts per factor (FICTURE `*.decode*.posterior.count.tsv.gz`), summed over the
              11 slides -- the factor loadings after decoding (closest to "FICTURE factors")
  clean_bins  summed raw counts of the CMFLAG-clean 2 um bins per decoded L3 label, 11 slides
  model       the fixed emission matrix given to FICTURE (atlas_L3_pseudobulk.tsv; built from the atlas, so
              this is the expected upper bound)

Inputs: mean_normcounts_L3.parquet of atlas_pseudobulk.py, <slide>_label_pseudobulk.csv.gz of label_pseudobulk.py,
and the FICTURE model and posterior tables named in the config (paths.model_pseudobulk, paths.posterior).
Outputs: in --out-dir, concordance_<variant>.csv (factors x atlas types) and concordance_summary.json.

Usage: ed6e_concordance.py --config config/cardiomyocyte.yaml --atlas-mean mean_normcounts_L3.parquet
         --labels-dir DIR --out-dir DIR
"""

import argparse
import glob
import json
import os

import pandas as pd

import cm_config
import hhca


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="FICTURE factor vs atlas marker concordance (ED6 e)")
    parser.add_argument("--config", required=True, help="config/cardiomyocyte.yaml")
    parser.add_argument("--atlas-mean", required=True, help="mean_normcounts_L3.parquet of atlas_pseudobulk.py")
    parser.add_argument("--labels-dir", required=True, help="folder with <slide>_label_pseudobulk.csv.gz")
    parser.add_argument("--out-dir", required=True, help="output folder")
    return parser.parse_args()


def main():
    """Select the atlas markers, correlate every factor variant with the atlas types and write the tables."""
    args = parse_args()
    cfg, _ = cm_config.load(args.config)
    params = cfg["ed6e"]
    os.makedirs(args.out_dir, exist_ok=True)

    atlas = pd.read_parquet(args.atlas_mean).drop(columns=["Unassigned"], errors="ignore")
    model = pd.read_csv(cm_config.path(cfg, "model_pseudobulk"), sep="\t", index_col=0)
    types = list(model.columns)
    assert sorted(types) == sorted(atlas.columns), (types, list(atlas.columns))
    atlas_means = atlas.loc[model.index.intersection(atlas.index), types]

    posterior = hhca.tl.pool_profiles(
        (
            pd.read_csv(cm_config.path(cfg, "posterior", slide), sep="\t", index_col=0)
            for slide in cm_config.slides(cfg, "all")
        ),
        by_name=False,
    )
    label_files = sorted(glob.glob(f"{args.labels_dir}/*_label_pseudobulk.csv.gz"))
    label_counts = hhca.tl.pool_profiles(pd.read_csv(file, index_col=0) for file in label_files)

    per_type = hhca.tl.atlas_markers(
        atlas_means, int(params["marker_top_n"]), float(params["marker_min_mean"]), float(params["pseudocount"])
    )
    markers = list(dict.fromkeys(gene for genes in per_type.values() for gene in genes))
    summary = {
        "n_markers": len(markers),
        "n_types": len(types),
        "markers_per_type": per_type,
        "n_label_files": len(label_files),
        "definition": {key: params[key] for key in params},
    }
    for name, profiles in (("posterior", posterior), ("clean_bins", label_counts), ("model", model)):
        if profiles is None:
            continue
        r = hhca.tl.factor_concordance(profiles, atlas_means, markers)
        r.to_csv(f"{args.out_dir}/concordance_{name}.csv")
        summary[name] = hhca.tl.concordance_summary(r)
        print(
            f"{name:10s} self-match {summary[name]['self_match']}/{len(types)}  "
            f"diagonal mean r {summary[name]['diag_mean_r']:.3f}",
            flush=True,
        )
    with open(f"{args.out_dir}/concordance_summary.json", "w") as handle:
        json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    main()
