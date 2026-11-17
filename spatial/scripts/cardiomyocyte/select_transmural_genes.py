"""Fig 3 i: select the transmural cardiomyocyte gene modules from the 45-bin profiles of three donors.

Per gene and donor, rho = Spearman correlation between bin index and the z-scored, smoothed 45-bin profile
(H0054, H0063: all 45 bins; AH10: its sampled bins only, epi 0-14 and endo 30-44, the mid-wall stays a gap);
the consensus is the mean rho over the donors in which the gene is measured (hhca.tl.transmural_rho).
Selection: among genes of the universe (pooled H0054 + H0063 detection) with the same sign of rho in all donors,
the genes with the highest consensus rho (rising towards the endocardium) and with the lowest (rising towards the
epicardium) (hhca.tl.select_modules). The 2-donor consensus (H0054 + H0063) is kept in the table and the JSON.
Parameters: `transmural` in config/cardiomyocyte.yaml.

Inputs: msearch_{H0054,H0063,AH10}.npz of compute_marker_search.py in --tab.
Outputs: written to --tab, module_marker_search_withAH10.csv (all genes, ranked),
markersearch_genesets_withAH10.json and traj_means_markersearch_withAH10_{H0054,H0063,AH10_cmflag}.csv (45-bin
means of the selected genes).

Usage: select_transmural_genes.py --tab DIR
"""

import os

from hhca import thread_env

os.environ.update(thread_env(1, numba=False))

import argparse
import json

import numpy as np

import hhca
from hhca.config import load_config


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Fig 3 i gene selection (3 donors)")
    parser.add_argument(
        "--tab", required=True, help="folder with msearch_{H0054,H0063,AH10}.npz; outputs are written here"
    )
    return parser.parse_args()


def main():
    """Rank the genes by transmural rho, select the modules and write the tables."""
    tab = parse_args().tab
    params = load_config("cardiomyocyte")["transmural"]
    donors = params["donors"]
    profiles = {donor: dict(np.load(f"{tab}/msearch_{donor}.npz", allow_pickle=True)) for donor in donors}

    ranking = hhca.tl.transmural_rho(profiles, params["min_detection"], params["sigma"])
    print(f"{len(ranking)} detected genes, {int(ranking.in_universe.sum())} in the selection universe")
    two = hhca.tl.select_modules(ranking, params["n_endo"], params["n_epi"], donors=2)
    three = hhca.tl.select_modules(ranking, params["n_endo"], params["n_epi"], donors=3)

    ranking["selected_module_3donor"] = ""
    ranking.loc[ranking.gene.isin(three["endo"]), "selected_module_3donor"] = "endo_rising"
    ranking.loc[ranking.gene.isin(three["epi"]), "selected_module_3donor"] = "epi_rising"
    ranking = ranking.sort_values("consensus_rho_3donor", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank_endo_to_epi_3donor", np.arange(1, len(ranking) + 1))
    ranking.to_csv(f"{tab}/module_marker_search_withAH10.csv", index=False)

    tables = hhca.tl.module_profiles(profiles, three["endo"] + three["epi"], mask_unsampled=False)
    for donor, table in tables.items():
        suffix = donor if len(profiles[donor]["layers"]) == len(params["layers"]) else f"{donor}_cmflag"
        table.to_csv(f"{tab}/traj_means_markersearch_withAH10_{suffix}.csv")
    with open(f"{tab}/markersearch_genesets_withAH10.json", "w") as handle:
        json.dump(
            {
                "endo_module": three["endo"],
                "epi_module": three["epi"],
                "endo_module_2donor": two["endo"],
                "epi_module_2donor": two["epi"],
                "DET_THRESH_SELECT": params["min_detection"],
                "SIGMA": params["sigma"],
                "N_ENDO": params["n_endo"],
                "N_EPI": params["n_epi"],
                "N_BINS": int(profiles[donors[0]]["bin_sum"].shape[0]),
                "N_PER_BLOCK": params["bins_per_section"],
                "donors": donors,
            },
            handle,
            indent=2,
        )
    print("3 donors, endo:", three["endo"], "epi:", three["epi"])
    print("2 donors, endo:", two["endo"], "epi:", two["epi"])
    print(ranking[ranking.selected_module_3donor != ""].drop(columns="abs_consensus_3donor").to_string(index=False))


if __name__ == "__main__":
    main()
