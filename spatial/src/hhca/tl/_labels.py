"""Expression profiles of the Level-3 labels and their agreement with the atlas (ED6 d, e)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from hhca.tl._bins import _label_codes, _row_blocks


def label_pseudobulk(adata, groupby="L3_label", chunk=500_000) -> pd.DataFrame:
    """Sum the raw counts of a section over the bins of each label.

    Parameters
    ----------
    adata
        One section (in memory or backed) with raw counts in ``X``.
    groupby
        Categorical ``obs`` column with the label of each bin.
    chunk
        Number of bins read at a time.

    Returns
    -------
    Genes x labels table of summed counts; ``attrs["n_bins"]`` holds the number of bins per label.
    """
    categories, codes = _label_codes(adata, groupby)
    sums = np.zeros((len(categories), adata.n_vars))
    for start, stop, block in _row_blocks(adata, chunk):
        membership = sp.csr_matrix(
            (np.ones(stop - start), (codes[start:stop], np.arange(stop - start))),
            shape=(len(categories), stop - start),
        )
        sums += (membership @ block.astype(np.float64)).toarray()
    counts = pd.DataFrame(sums.T, index=adata.var_names.to_numpy().astype(str), columns=categories)
    n_bins = np.bincount(codes, minlength=len(categories))
    counts.attrs["n_bins"] = {label: int(n) for label, n in zip(categories, n_bins)}
    return counts


def pool_profiles(tables, by_name=True) -> pd.DataFrame:
    """Add up genes x types tables of several sections; a gene missing from a section counts as 0.

    With `by_name`, rows that share a gene name are summed within each table first.
    """
    total = None
    for table in tables:
        if by_name:
            table = table.groupby(level=0).sum()
        total = table if total is None else total.add(table, fill_value=0)
    return total


def marker_zscores(profiles, genes) -> pd.DataFrame:
    """z-score of marker genes across labels: log1p CP10k of each label's profile, scaled per gene.

    Parameters
    ----------
    profiles
        Genes x labels table of summed counts.
    genes
        Marker genes, in the order of the rows of the result.
    """
    log_cp10k = np.log1p(profiles * 1e4 / profiles.sum(axis=0))
    expression = log_cp10k.reindex(genes)
    return expression.sub(expression.mean(axis=1), axis=0).div(expression.std(axis=1, ddof=0), axis=0)


def atlas_markers(atlas_means, top_n=10, min_mean=0.1, pseudocount=0.01) -> dict[str, list[str]]:
    """Marker genes of each atlas type: the genes with the highest ratio to the best other type.

    For every type (column of `atlas_means`), genes with a mean of at least `min_mean` are ranked by
    ``log2((mean + pseudocount) / (highest mean of another type + pseudocount))`` and the `top_n` are kept.
    """
    markers = {}
    for cell_type in atlas_means.columns:
        other = atlas_means.drop(columns=[cell_type]).max(axis=1)
        score = np.log2((atlas_means[cell_type] + pseudocount) / (other + pseudocount))
        score = score[atlas_means[cell_type] >= min_mean]
        markers[cell_type] = score.sort_values(ascending=False).head(top_n).index.tolist()
    return markers


def _specificity_profiles(profiles, genes, types, markers):
    """Profiles in marker-specificity space: CP10k over `genes`, log1p, z-score of each marker across types."""
    profiles = profiles.reindex(index=genes, columns=types).fillna(0.0)
    logged = np.log1p(profiles.div(profiles.sum(axis=0).replace(0, np.nan), axis=1) * 1e4).loc[markers]
    centred = logged.sub(logged.mean(axis=1), axis=0)
    return centred.div(logged.std(axis=1, ddof=0).replace(0, np.nan), axis=0).fillna(0.0)


def factor_concordance(profiles, atlas_means, markers) -> pd.DataFrame:
    """Correlate the profiles of the FICTURE factors with the atlas types over the atlas marker genes.

    Both sets of profiles are normalised to 10,000 over the genes of `atlas_means`, log1p-transformed and each
    marker gene is z-scored across the types, so a correlation compares which types a gene is specific for.

    Parameters
    ----------
    profiles
        Genes x factors table (posterior counts, summed counts per label or the model matrix).
    atlas_means
        Genes x atlas types table of mean expression; its genes and types define the comparison.
    markers
        Marker genes (the union of :func:`atlas_markers`).

    Returns
    -------
    Pearson r, factors (rows) x atlas types (columns).
    """
    genes, types = atlas_means.index, list(atlas_means.columns)
    atlas_z = _specificity_profiles(atlas_means, genes, types, markers)
    factor_z = _specificity_profiles(profiles, genes, types, markers)
    r = np.corrcoef(factor_z.T.values, atlas_z.T.values)[: len(types), len(types) :]
    r = pd.DataFrame(r, index=types, columns=types)
    r.index.name = "FICTURE factor"
    return r


def concordance_summary(r) -> dict:
    """Self-matches (the best atlas type of a factor is its own) and diagonal of a concordance matrix."""
    types = list(r.index)
    diagonal = np.diag(r.values)
    self_match = [t for t in types if r.loc[t].idxmax() == t]
    return {
        "self_match": len(self_match),
        "diag_mean_r": float(diagonal.mean()),
        "diag_median_r": float(np.median(diagonal)),
        "not_self_matched": {t: r.loc[t].idxmax() for t in types if t not in self_match},
    }
