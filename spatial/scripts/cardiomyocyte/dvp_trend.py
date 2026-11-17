"""Fig 3 k: DVP transmural protein trend table (per cohort x cell type: slope, p, BH q).

Model (Methods): per protein x cohort x cell type, keep if >= 2/3 of the samples are valid; OLS log2 ~ region
(endo 0, myo 1, epi 2) + donor fixed effect; the region slope is sign-inverted (slope > 0 = higher toward
endocardium); BH q per cohort x cell type.

Inputs: DVP protein-group matrices of the two cohorts (UCLA6, 6 donors; UCLA7, 8 donors; cardiomyocytes and
macrophages; endo, myo, epi) as one long table (--matrix LONG.tsv) with the columns protein_group, gene, cohort,
cell_type, donor, region (endo|myo|epi), log2_intensity (replicates collapsed, missing = NaN). The matrices come
with the proteomics data and are not among the pipeline inputs, so this script is run by hand.
Outputs: long table protein_group, gene, cohort, cell_type, slope, p_value, q_value, n_samples
(input of figures/Fig3/panel_k.py).

Usage: dvp_trend.py --matrix M --out results/cardiomyocyte/fig3k/trend_table.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats

REGION_POSITION = {"endo": 0.0, "myo": 1.0, "epi": 2.0}


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="DVP transmural protein trend table (Fig 3 k)")
    parser.add_argument("--matrix", required=True, help="long table of log2 protein intensities (tsv)")
    parser.add_argument("--out", required=True, help="output csv")
    return parser.parse_args()


def bh(p):
    """Benjamini-Hochberg q values; NaN p values stay NaN."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = stats.false_discovery_control(p[ok])
    return out


def region_slope(samples):
    """OLS coefficient of the region position with donor fixed effects and its two-sided p value.

    Returns None when the model leaves no residual degree of freedom.
    """
    donors = pd.get_dummies(samples["donor"], drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(samples)), samples["x"].to_numpy(), donors.to_numpy()])
    y = samples["log2_intensity"].to_numpy(float)
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    dof = len(y) - rank
    if dof < 1:
        return None
    resid = y - X @ beta
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    t = beta[1] / np.sqrt(cov[1, 1]) if cov[1, 1] > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), dof) if np.isfinite(t) else np.nan
    return beta[1], p


def trend_table(intensities):
    """Slope, p and BH q of every protein group per cohort and cell type."""
    intensities["x"] = intensities["region"].map(REGION_POSITION)
    rows = []
    for (cohort, cell_type), group in intensities.groupby(["cohort", "cell_type"]):
        tested = []
        n_samples = group.groupby(["donor", "region"]).ngroups
        for (protein_group, gene), samples in group.groupby(["protein_group", "gene"]):
            samples = samples.dropna(subset=["log2_intensity"])
            if len(samples) < np.ceil(2 / 3 * n_samples):
                continue
            fit = region_slope(samples)
            if fit is None:
                continue
            slope, p = fit
            tested.append(
                {
                    "protein_group": protein_group,
                    "gene": gene,
                    "cohort": cohort,
                    "cell_type": cell_type,
                    "slope": -slope,
                    "p_value": p,
                    "n_samples": len(samples),
                }
            )
        tested = pd.DataFrame(tested)
        if len(tested):
            tested["q_value"] = bh(tested.p_value)
            rows += tested.to_dict("records")
    return pd.DataFrame(rows)


def main():
    """Fit the trends, write the table and print the number of proteins tested and significant."""
    args = parse_args()
    trends = trend_table(pd.read_csv(args.matrix, sep="\t"))
    trends.to_csv(args.out, index=False)
    print(
        trends.groupby(["cohort", "cell_type"])
        .agg(
            tested=("p_value", "size"),
            raw_p05=("p_value", lambda v: int((v < 0.05).sum())),
            q05=("q_value", lambda v: int((v < 0.05).sum())),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
