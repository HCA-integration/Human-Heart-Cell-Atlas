"""Transmural slopes of cardiomyocyte transcripts and their agreement with protein slopes (Fig 3 l)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from hhca.tl._bins import VENTRICULAR_CM, _row_blocks
from hhca.tl._transmural import LAYERS


def section_pseudobulk(sections, cell_type=VENTRICULAR_CM, chunk=400_000, log=None) -> dict:
    """Sum the raw counts of each section over the bins of one cell type and over all bins.

    Sections with another gene space are matched by name to the genes of the first section (missing genes = 0).

    Parameters
    ----------
    sections
        Mapping of sample name (``<donor>_<layer>``) -> section (in memory or backed).
    cell_type
        Level-3 label of the cell-type pseudobulk.
    chunk
        Number of bins read at a time.

    Returns
    -------
    Dictionary with ``cell_type`` and ``all`` (genes x samples, summed counts), ``composition`` (bins per
    Level-3 label and sample) and ``meta`` (bins and total counts per sample).
    """
    genes = None
    composition, cell_type_counts, all_counts, meta = {}, {}, {}, []
    for sample, adata in sections.items():
        labels = adata.obs["L3_label"].astype(str).to_numpy()
        in_type = labels == cell_type
        composition[sample] = pd.Series(labels).value_counts()
        type_sum = np.zeros(adata.n_vars, dtype=np.float64)
        all_sum = np.zeros(adata.n_vars, dtype=np.float64)
        for start, stop, block in _row_blocks(adata, chunk):
            if in_type[start:stop].any():
                type_sum += np.asarray(block[in_type[start:stop]].sum(axis=0)).ravel()
            all_sum += np.asarray(block.sum(axis=0)).ravel()
        section_genes = adata.var_names.to_numpy()
        if genes is None:
            genes = section_genes
        if not np.array_equal(section_genes, genes):
            type_sum = pd.Series(type_sum, index=section_genes).groupby(level=0).sum().reindex(genes).fillna(0)
            all_sum = pd.Series(all_sum, index=section_genes).groupby(level=0).sum().reindex(genes).fillna(0)
            type_sum, all_sum = type_sum.to_numpy(), all_sum.to_numpy()
        cell_type_counts[sample], all_counts[sample] = type_sum, all_sum
        meta.append(
            {
                "sample": sample,
                "n_bins": int(adata.n_obs),
                "n_cell_type": int(in_type.sum()),
                "cell_type_counts": float(type_sum.sum()),
                "all_counts": float(all_sum.sum()),
            }
        )
        if log:
            log(f"{sample}: {adata.n_obs} bins, {int(in_type.sum())} {cell_type}")
    composition = pd.DataFrame(composition).fillna(0).astype(int)
    composition.index.name = "L3_label"
    return {
        "cell_type": pd.DataFrame(cell_type_counts, index=genes)[list(sections)],
        "all": pd.DataFrame(all_counts, index=genes)[list(sections)],
        "composition": composition,
        "meta": pd.DataFrame(meta),
    }


def log_cp10k_pseudobulk(counts, donors, layers=LAYERS) -> pd.DataFrame:
    """Pseudobulk counts of the donors' sections as log1p CP10k (genes x samples, ordered by donor and layer)."""
    samples = [f"{donor}_{layer}" for donor in donors for layer in layers if f"{donor}_{layer}" in counts.columns]
    counts = counts[samples]
    return np.log1p(counts.divide(counts.sum(axis=0), axis=1) * 1e4)


def transmural_slopes(expression, donors, layers=LAYERS) -> pd.DataFrame:
    """Slope of every gene across the wall, pooled over the sections and per donor.

    The pooled slope is the least-squares slope of log1p CP10k on the position of the section (0, 1, 2 for the
    three `layers`) over all samples, with its p value and r; q values are Benjamini-Hochberg adjusted. A gene is
    ``sign_consistent`` when the slopes of all donors are non-zero and share a sign. When every donor has all
    three sections, the table also holds the mean of each layer over the donors and whether these means are
    monotonic.

    Parameters
    ----------
    expression
        Genes x samples (``<donor>_<layer>``) log1p CP10k (:func:`log_cp10k_pseudobulk`).
    donors
        Donors to use.
    """
    samples = list(expression.columns)
    position = np.array([layers.index(sample.split("_")[1]) for sample in samples], float)
    donor_of = np.array([sample.split("_")[0] for sample in samples])
    complete = all((donor_of == donor).sum() == len(layers) for donor in donors)
    layer_position = np.arange(len(layers), dtype=float)
    values = expression[samples].to_numpy(float)
    rows = []
    for i, gene in enumerate(expression.index):
        y = values[i]
        slope, _, r, p, _ = stats.linregress(position, y)
        row = {"gene": gene, "slope": slope, "p": p, "r": r}
        donor_slopes = []
        for donor in donors:
            of_donor = donor_of == donor
            donor_slope = np.polyfit(position[of_donor], y[of_donor], 1)[0] if of_donor.sum() >= 2 else np.nan
            row[f"slope_{donor}"] = donor_slope
            donor_slopes.append(donor_slope)
        donor_slopes = np.array(donor_slopes)
        row["sign_consistent"] = bool(
            np.all(np.isfinite(donor_slopes)) and np.all(donor_slopes != 0) and len(set(np.sign(donor_slopes))) == 1
        )
        if complete:
            layer_mean = np.array(
                [
                    sum(y[samples.index(f"{donor}_{layer}")] * (1.0 / len(donors)) for donor in donors)
                    for layer in layers
                ]
            )
            rho = stats.spearmanr(layer_position, layer_mean).correlation if layer_mean.std() > 0 else 0.0
            row["monotonic"] = abs(rho) == 1.0
            row["spearman_layer"] = rho
            row.update(dict(zip(layers, layer_mean)))
        rows.append(row)
    slopes = pd.DataFrame(rows).set_index("gene")
    slopes["q"] = stats.false_discovery_control(slopes["p"].fillna(1.0).clip(1e-300, 1.0).to_numpy())
    return slopes


def protein_slopes(proteins) -> pd.DataFrame:
    """Consensus transmural slope of the cardiomyocyte proteins of the two DVP cohorts, one row per gene.

    Parameters
    ----------
    proteins
        DVP trend table with ``Genes`` and, per cohort, ``cohort<i>_CM_slope`` and ``cohort<i>_CM_q_value``.
    """
    proteins = proteins.copy()
    proteins["protein_CM_consensus"] = (proteins["cohort1_CM_slope"] + proteins["cohort2_CM_slope"]) / 2
    proteins["protein_CM_sign_consistent"] = np.sign(proteins["cohort1_CM_slope"]) == np.sign(
        proteins["cohort2_CM_slope"]
    )
    proteins["protein_CM_qmax"] = proteins[["cohort1_CM_q_value", "cohort2_CM_q_value"]].max(axis=1)
    return proteins.dropna(subset=["protein_CM_consensus"]).drop_duplicates("Genes").set_index("Genes")


def slope_concordance(transcript, protein) -> dict:
    """Spearman rho and sign agreement between transcript and protein slopes; zero or missing slopes dropped."""
    slopes = pd.DataFrame({"t": np.asarray(transcript, float), "p": np.asarray(protein, float)}).dropna()
    slopes = slopes[(slopes["t"] != 0) & (slopes["p"] != 0)]
    n = len(slopes)
    if n < 3:
        return {"n": n, "rho": np.nan, "p": np.nan, "sign_agree": np.nan}
    rho, p = stats.spearmanr(slopes["t"], slopes["p"])
    sign_agree = (np.sign(slopes["t"]) == np.sign(slopes["p"])).mean()
    return {"n": n, "rho": float(rho), "p": float(p), "sign_agree": float(sign_agree)}


def _first_gene(names):
    """First gene symbol of a `;`-separated protein-group entry, upper case."""
    return names.astype(str).str.split(";").str[0].str.upper()


def dvp_gene_sets(proteins, specificity, structural_axis, p_max=0.05) -> dict[str, list[str]]:
    """Gene sets for the comparison of transcript and protein slopes.

    ``curated17``: the hand-curated structural genes whose atlas ratio ``CM_over_topNonCM`` is at least 1.
    ``dvp_rule``: proteins with raw p < `p_max` in both cohorts and the same slope sign (cardiomyocytes).
    ``dvp_rule_cmexpr`` / ``_cmratio`` / ``_cmsafe``: the ``dvp_rule`` genes that are expressed in ventricular
    cardiomyocytes / cardiomyocyte-specific by the ratio / both (``cm_safe``).

    Parameters
    ----------
    proteins
        DVP trend table of all proteins.
    specificity
        Table of :func:`hhca.tl.cardiomyocyte_specificity`, indexed by gene.
    structural_axis
        Hand-curated structural cardiomyocyte genes.
    """
    ratio = specificity["CM_over_topNonCM"]
    gene = _first_gene(proteins["Genes"])
    p1, p2 = proteins["cohort1_CM_p_value"], proteins["cohort2_CM_p_value"]
    s1, s2 = proteins["cohort1_CM_slope"], proteins["cohort2_CM_slope"]
    rule = (p1 < p_max) & (p2 < p_max) & (np.sign(s1) == np.sign(s2))
    dvp_rule = sorted(set(gene[rule]))
    return {
        "curated17": [g for g in structural_axis if ratio.get(g, 0) >= 1.0],
        "dvp_rule": dvp_rule,
        "dvp_rule_cmexpr": [g for g in dvp_rule if specificity["cm_expressed"].get(g, False)],
        "dvp_rule_cmratio": [g for g in dvp_rule if ratio.get(g, 0) >= 1.0],
        "dvp_rule_cmsafe": [g for g in dvp_rule if specificity["cm_safe"].get(g, False)],
    }
