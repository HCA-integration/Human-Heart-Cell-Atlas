"""Fig 2 j, k, l: per-donor gene-set scores by sex on the 280 donors.

Endocardial serotonin reception (j) and its course over age (k), arterial beta-adrenergic drug target (l). Gene
sets, cell types, age bands and thresholds: config/config.yaml (donor_scores, donors).

Inputs: results/atlas/per_donor_celltype_scores_280donors.npz.
Outputs: results/figures/Fig2/panel_{j,k,l}.{png,pdf}.

Usage: python figures/Fig2/panel_jkl.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import hhca
from hhca.config import load_config
from hhca.donors import read_scores
from hhca.pl import save_panel


def main():
    """Draw panels j, k and l."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    cfg = load_config("config")
    params, adult_age = cfg["donor_scores"], cfg["donors"]["adult_age"]
    panels = params["panels"]
    donors, scores, _ = read_scores(
        hhca.results_dir("atlas", "per_donor_celltype_scores_280donors.npz"),
        levels=sorted({panel["level"] for panel in panels.values()}),
    )
    tables = {
        name: donors.assign(
            score=hhca.tl.donor_score(
                scores[panel["level"]], panel["gene_set"], panel["cell_type"], params["min_nuclei"]
            ).to_numpy()
        )
        for name, panel in panels.items()
    }

    for name, letter in (("serotonin", "j"), ("beta_adrenergic", "l")):
        female, male = hhca.tl.split_by_sex(tables[name], adult_age=adult_age)
        test = hhca.tl.compare_sexes(female, male)
        print(f"{letter}: nF={len(female)} nM={len(male)} p={test['p']:.3e} g={test['hedges_g']:+.3f}")
        fig = hhca.pl.sex_boxes(
            female, male, panels[name]["title"], test["p"], test["hedges_g"], panels[name]["jitter_seed"]
        )
        print(save_panel(fig, "Fig2", letter))

    bands = [tuple(band) for band in params["age_bands"]]
    band_scores = hhca.tl.split_by_sex(tables["serotonin"], adult_age=adult_age, bands=bands)
    p_values = [hhca.tl.compare_sexes(female, male, params["min_band_donors"])["p"] for female, male in band_scores]
    fig = hhca.pl.sex_by_age(band_scores, bands, p_values, panels["serotonin"]["title_age"])
    print(save_panel(fig, "Fig2", "k"))


if __name__ == "__main__":
    main()
