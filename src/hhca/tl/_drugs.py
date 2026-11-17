"""Calibre selectivity of drug-target classes on the per-structure pseudobulk."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats import t as tdist
from statsmodels.stats.multitest import multipletests

CALIBRE_ORDER = ["small_resistance", "intermediate", "large_conductance"]


def _ols_coef1(y, design):
    """OLS of y on the design matrix: coefficient of column 1 and its two-sided t-test p-value."""
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ (design.T @ y)
    resid = y - design @ beta
    dof = len(y) - design.shape[1]
    sigma2 = (resid**2).sum() / dof
    se = np.sqrt(xtx_inv[1, 1] * sigma2)
    tval = beta[1] / (se + 1e-12)
    return float(beta[1]), float(2 * tdist.sf(abs(tval), dof))


def _zscore(values):
    """Standardise with the population standard deviation (plus 1e-9)."""
    return (values - values.mean()) / (values.std() + 1e-9)


def drug_class_scores(pseudobulk, calibre, classes, variant="clean") -> pd.DataFrame:
    """Score every drug class per vessel structure.

    The pseudobulk is joined to the calibre table; per structure, each target gene gets
    ``log1p(CP10k)`` on the common-gene library, and a class scores the mean over its target genes.

    Parameters
    ----------
    pseudobulk
        Per-structure sums of :func:`hhca.tl.pseudobulk_structures`, all sections.
    calibre
        Calibre table with ``vessel_class``.
    classes
        Drug class -> ``{"genes": [...], "compartment": ...}``.
    variant
        ``"clean"`` uses the sums after the CMFLAG spillover mask, ``"raw"`` the sums of all bins.

    Returns
    -------
    One row per structure with ``caliber``, ``lib_common``, ``n_bins``, ``log1p_cp10k__<gene>``,
    ``score__<class>`` and ``det__<class>`` (1 if any target gene has a count).
    """
    targets = sorted({gene for drug_class in classes.values() for gene in drug_class["genes"]})
    structs = pseudobulk.merge(
        calibre[["slide", "struct", "vessel_class", "mural_VSMC_frac", "lumen_equiv_diam_um"]],
        on=["slide", "struct"],
        how="inner",
    ).rename(columns={"vessel_class": "caliber"})
    structs["lib_common"] = structs[f"libc_{variant}"].to_numpy()
    structs["n_bins"] = structs[f"nb_{variant}"].to_numpy()
    for gene in targets:
        structs[gene] = structs[f"{gene}_{variant}"].to_numpy()
    cp10k = structs[targets].values / np.maximum(structs["lib_common"].values, 1e-9)[:, None] * 1e4
    log_cp10k = pd.DataFrame(np.log1p(cp10k), columns=targets)
    for gene in targets:
        structs[f"log1p_cp10k__{gene}"] = log_cp10k[gene].values
    for name, drug_class in classes.items():
        structs[f"score__{name}"] = log_cp10k[drug_class["genes"]].mean(1).values
        structs[f"det__{name}"] = (structs[drug_class["genes"]].sum(1).values > 0).astype(int)
    return structs


def tested_structures(scores, min_library=50) -> pd.DataFrame:
    """Structures whose common-gene library reaches `min_library`."""
    return scores[scores["lib_common"] >= min_library].copy()


def calibre_selectivity(tested, classes) -> pd.DataFrame:
    """Test every drug class between large and small vessels.

    OLS ``score ~ is_large + z(log library) + C(slide)`` over the large and small structures, BH-FDR over the
    classes; in addition the same model with ``z(mural_VSMC_frac)`` and a Mann-Whitney test.

    Returns
    -------
    One row per class, sorted by the adjusted difference: mean, SEM, detection rate and n per calibre class,
    ``coef_large_minus_small_adj`` with ``p_LvsS`` and ``p_adj_LvsS``, the composition-adjusted coefficient and
    ``prefers``.
    """
    log_lib = np.log1p(tested["lib_common"].values)
    z_lib = _zscore(log_lib)
    z_vsmc = _zscore(tested["mural_VSMC_frac"].values)
    slide_dummies = pd.get_dummies(tested["slide"], drop_first=True).astype(float).values
    large_or_small = tested["caliber"].isin(["large_conductance", "small_resistance"]).values
    is_large = (tested["caliber"] == "large_conductance").astype(float).values
    intercept = np.ones(large_or_small.sum())

    records = []
    for name, drug_class in classes.items():
        genes, compartment = drug_class["genes"], drug_class["compartment"]
        score = tested[f"score__{name}"].values
        detected = tested[f"det__{name}"].values
        record = {"drug_class": name, "compartment": compartment, "target_genes": ",".join(genes)}
        for caliber in CALIBRE_ORDER:
            in_class = (tested["caliber"] == caliber).values
            record[f"mean_{caliber}"] = float(score[in_class].mean())
            record[f"sem_{caliber}"] = float(score[in_class].std() / np.sqrt(max(in_class.sum(), 1)))
            record[f"det_{caliber}"] = float(detected[in_class].mean())
            record[f"n_{caliber}"] = int(in_class.sum())
        y = score[large_or_small]
        design = np.column_stack(
            [intercept, is_large[large_or_small], z_lib[large_or_small], slide_dummies[large_or_small]]
        )
        record["coef_large_minus_small_adj"], record["p_LvsS"] = _ols_coef1(y, design)
        design_composition = np.column_stack(
            [
                intercept,
                is_large[large_or_small],
                z_lib[large_or_small],
                z_vsmc[large_or_small],
                slide_dummies[large_or_small],
            ]
        )
        record["coef_LvsS_composition_adj"], record["p_LvsS_composition_adj"] = _ols_coef1(y, design_composition)
        try:
            _, p_mwu = mannwhitneyu(
                score[(tested.caliber == "large_conductance").values],
                score[(tested.caliber == "small_resistance").values],
                alternative="two-sided",
            )
        except ValueError:  # an empty calibre class
            p_mwu = np.nan
        record["p_LvsS_mwu"] = float(p_mwu)
        records.append(record)
    summary = pd.DataFrame(records)
    summary["p_adj_LvsS"] = multipletests(summary["p_LvsS"], method="fdr_bh")[1]
    summary["prefers"] = np.where(summary["coef_large_minus_small_adj"] > 0, "large_conductance", "small_resistance")
    return summary.sort_values("coef_large_minus_small_adj", ascending=False)


def drug_scores_by_section(tested, classes) -> pd.DataFrame:
    """Mean score of every drug class per section and calibre class, and the large-minus-small difference."""
    rows = []
    for slide in sorted(tested.slide.unique()):
        on_slide = tested[tested.slide == slide]
        for name in classes:
            row = {"slide": slide, "drug_class": name}
            for caliber in CALIBRE_ORDER:
                in_class = (on_slide["caliber"] == caliber).values
                row[f"mean_{caliber}"] = (
                    float(on_slide[f"score__{name}"].values[in_class].mean()) if in_class.any() else np.nan
                )
                row[f"n_{caliber}"] = int(in_class.sum())
            row["L_minus_S"] = row["mean_large_conductance"] - row["mean_small_resistance"]
            rows.append(row)
    return pd.DataFrame(rows)


def calibre_context(tested) -> pd.DataFrame:
    """Structure count, median diameter and mean VSMC share per calibre class."""
    rows = []
    for caliber in CALIBRE_ORDER:
        in_class = (tested["caliber"] == caliber).values
        rows.append(
            {
                "caliber": caliber,
                "n": int(in_class.sum()),
                "median_diam_um": float(np.median(tested["lumen_equiv_diam_um"].values[in_class])),
                "mean_mural_VSMC_frac": float(tested["mural_VSMC_frac"].values[in_class].mean()),
            }
        )
    return pd.DataFrame(rows)
