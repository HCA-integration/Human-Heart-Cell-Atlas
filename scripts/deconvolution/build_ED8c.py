"""ED8 b-c: atrial-pericyte signature on the Kanemaru et al. 2023 Visium sections (LA, RA, LV, RV).

Three parts, drawn by hhca.pl.pericyte_signature: (a) the 30 signature genes by chamber (ED8 b), (b) pericyte-high
vs pericyte-low spots within the atria, (c) pericyte-high spots, atria vs ventricles (ED8 c).

Score = mean over the signature genes of the gene z-scored across all spots, so that no single highly
expressed gene dominates. Pericyte-high / -low = top / bottom within-region tertile of the total pericyte
abundance of the cell2location result stored in the objects (PC1_vent + PC2_atria + PC3_str + PC4_CMC-like).
The total is used because selecting spots on PC2_atria and then scoring an atrial-pericyte signature would be
circular.

Inputs:  <kanemaru>/visium-OCT_<region>_lognormalised.h5ad, the signature json
Outputs: <out>/ED8c_atrial_pericyte.{png,pdf}, ED8c_perspot.csv.gz, ED8c_gene_by_chamber_z.csv, ED8c_stats.json

The steps are hhca.tl.score_signature, hhca.tl.abundance_groups, hhca.tl.signature_by_chamber and
hhca.tl.compare_scores; chambers, pericyte types and tertiles: config slides and downstream.

Usage: build_ED8c.py --kanemaru DIR --signature atrial_pericyte_signature.json --out DIR [--config CONFIG]
"""

import argparse
import json
import os

import anndata as ad
import matplotlib
import pandas as pd
import scanpy as sc

matplotlib.use("Agg")

import deconv_config
import hhca


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kanemaru", required=True, help="directory with the Kanemaru Visium h5ad files")
    parser.add_argument("--signature", required=True, help="json with the atrial-pericyte marker genes")
    parser.add_argument("--out", required=True, help="output directory")
    return deconv_config.add_argument(parser).parse_args()


def load_spots(kanemaru_dir, markers, chamber_region, pericyte_columns):
    """The spots of all sections with the signature genes in X and chamber, region, donor, pericyte abundance."""
    sections, index_name = {}, None
    for chamber, region in chamber_region.items():
        adata = sc.read_h5ad(f"{kanemaru_dir}/visium-OCT_{chamber}_lognormalised.h5ad")
        genes = [gene for gene in markers if gene in adata.var_names]
        if not sections:
            missing = [gene for gene in markers if gene not in adata.var_names]
            print(f"  present in Visium: {len(genes)}/{len(markers)}; missing: {missing}")
        print(f"  {chamber}: {adata.n_obs:,} spots, {adata.obs['donor'].nunique()} donors")
        index_name = adata.obs_names.name
        obs = pd.DataFrame(index=adata.obs_names)
        obs["chamber"] = chamber
        obs["region"] = region
        obs["donor"] = adata.obs["donor"].astype(str).values
        obs["peri"] = adata.obs[pericyte_columns].sum(axis=1).values  # total pericyte abundance
        obs["peri_atrial"] = adata.obs["PC2_atria"].values
        sections[chamber] = ad.AnnData(X=adata[:, genes].X, obs=obs, var=pd.DataFrame(index=genes))
    return ad.concat(sections.values()), index_name


def main():
    """Score the signature per spot, draw the three parts and write the tables behind them."""
    args = parse_args()
    cfg = deconv_config.load(args.config)
    settings = cfg["downstream"]
    kanemaru_dir = args.kanemaru.rstrip("/")
    out_dir = args.out.rstrip("/")
    os.makedirs(out_dir + "/", exist_ok=True)
    run_stats = {"peri_props": None, "peri_rule": "tertile (Kanemaru c2l q05)"}

    with open(args.signature) as handle:
        markers = json.load(handle)["atrial_pericyte_markers"]
    print(f"signature: {len(markers)} genes")
    adata, index_name = load_spots(kanemaru_dir, markers, settings["ed8_regions"], settings["ed8_pericyte_kanemaru"])
    print(f"total spots {adata.n_obs:,} | genes used {adata.n_vars}")

    # per-spot signature score: z-score each gene across all spots, then average
    hhca.tl.score_signature(adata)
    adata = adata[adata.obs["score"].notna().values].copy()
    spots = adata.obs.rename_axis(index_name)

    # pericyte-high / -low by within-region tertiles of total pericyte abundance
    hhca.tl.abundance_groups(
        spots, "peri", "region", settings["ed8_tertile"], ["pericyte-low", "pericyte-high"], key_added="peri_grp"
    )
    adata.obs["peri_grp"] = spots["peri_grp"].values
    print("\nspots per group:")
    print(pd.crosstab(spots.region, spots.peri_grp).to_string())

    chambers = list(settings["ed8_regions"])
    atrial = [chamber for chamber, region in settings["ed8_regions"].items() if region == "atria"]
    chamber_means = hhca.tl.signature_by_chamber(adata, chambers, sort_by=atrial)

    # pericyte-high vs pericyte-low spots of the atria; pericyte-high spots, atria vs ventricles
    atria, high = spots[spots.region == "atria"], spots[spots.peri_grp == "pericyte-high"]
    test = hhca.tl.compare_scores(
        atria[atria.peri_grp == "pericyte-low"].score.values, atria[atria.peri_grp == "pericyte-high"].score.values
    )
    run_stats["b_atria_high_vs_low"] = {
        "n_high": test["n_group"],
        "n_low": test["n_reference"],
        "median_high": test["median_group"],
        "median_low": test["median_reference"],
        "delta_median": test["delta_median"],
        "mwu_p": test["mwu_p"],
        "mwu_U": test["mwu_U"],
    }
    print(
        f"\n(b) atria pericyte-high vs low: n={test['n_group']} vs {test['n_reference']}, "
        f"median {test['median_group']:.3f} vs {test['median_reference']:.3f}, p={test['mwu_p']:.3g}"
    )
    test = hhca.tl.compare_scores(
        high[high.region == "ventricle"].score.values, high[high.region == "atria"].score.values
    )
    run_stats["c_high_atria_vs_ventricle"] = {
        "n_atria": test["n_group"],
        "n_ventricle": test["n_reference"],
        "median_atria": test["median_group"],
        "median_ventricle": test["median_reference"],
        "delta_median": test["delta_median"],
        "mwu_p": test["mwu_p"],
        "mwu_U": test["mwu_U"],
    }
    print(
        f"(c) pericyte-high atria vs ventricle: n={test['n_group']} vs {test['n_reference']}, "
        f"median {test['median_group']:.3f} vs {test['median_reference']:.3f}, p={test['mwu_p']:.3g}"
    )

    fig = hhca.pl.pericyte_signature(chamber_means, spots, caption=True)
    for extension in ("png", "pdf"):
        fig.savefig(f"{out_dir}/ED8c_atrial_pericyte.{extension}", dpi=300)

    spots[["chamber", "region", "donor", "peri", "peri_atrial", "score", "peri_grp"]].to_csv(
        f"{out_dir}/ED8c_perspot.csv.gz", index=True, compression="gzip"
    )
    chamber_means.to_csv(f"{out_dir}/ED8c_gene_by_chamber_z.csv")
    print("\nwrote", f"{out_dir}/ED8c_atrial_pericyte.png/.pdf")

    donor_test = hhca.tl.paired_donor_test(spots[spots.peri_grp == "pericyte-high"])
    if donor_test is not None:
        print(
            f"\nper-donor test (pericyte-high spots): {donor_test['n_donors']} donors with both regions, "
            f"atria {donor_test['median_atria']:.3f} vs ventricle {donor_test['median_ventricle']:.3f}, "
            f"paired p={donor_test['wilcoxon_p']:.3g}"
        )
        run_stats["per_donor_paired"] = donor_test
    run_stats["n_spots_scored"] = int(len(spots))
    run_stats["groups"] = pd.crosstab(spots.region, spots.peri_grp).to_dict()
    with open(f"{out_dir}/ED8c_stats.json", "w") as handle:
        json.dump(run_stats, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
