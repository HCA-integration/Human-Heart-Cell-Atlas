"""Niches of gene-high cardiomyocytes on a 10 um grid of one section (Fig 3 f-h).

Do cardiomyocyte bins with a high density of a gene (NPPA, NPPB) lie closer to fibroblast-rich regions, the
endocardium and vessels than the other cardiomyocyte bins? Decode pixels and transcripts of a section are counted
on a regular grid; every bin gets its cell-type composition, the composition of its neighbourhood and the distance
to the nearest bin rich in a cell-type group; gene-high bins are then compared with the other cardiomyocyte bins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests

from hhca.grid import bin_key


def niche_bins(
    pixels,
    transcripts,
    factor_order,
    groups,
    genes=("NPPA", "NPPB"),
    spillover=None,
    grid_um=10.0,
    min_pixels=4,
    chunksize=3_000_000,
) -> pd.DataFrame:
    """Count the decode pixels and transcripts of one section on a regular grid.

    Parameters
    ----------
    pixels
        Decode pixels of the section (``X``, ``Y`` in um, ``K1``; :func:`hhca.io.read_decode`).
    transcripts
        Path of the FICTURE input transcripts of the section (``X``, ``Y``, ``gene``, ``Count``); the table is
        read in blocks of `chunksize` rows.
    factor_order
        Level-3 types in factor order.
    groups
        Mapping of cell-type group -> Level-3 types.
    genes
        Genes whose counts are kept per bin.
    spillover
        Keys of 2 um bins whose pixels and transcripts are left out (:func:`hhca.grid.spillover_keys`).
    grid_um
        Bin size.
    min_pixels
        Decode pixels a bin needs to be kept.

    Returns
    -------
    One row per bin: ``bin``, centre ``x_um`` / ``y_um``, ``npix``, ``total_counts``, the counts of each gene,
    the pixel fraction of each group (``frac_<group>``) and the gene densities per 1,000 counts (``<gene>_cp1k``).
    """
    xy = pixels[["X", "Y"]].values.astype(np.float64)
    top_factor = pixels["K1"].values.astype(np.int16)
    if spillover is not None:
        keep = ~np.isin(bin_key(xy[:, 0], xy[:, 1]), spillover)
        xy, top_factor = xy[keep], top_factor[keep]
    x, y = xy[:, 0], xy[:, 1]
    xmin, ymin = x.min(), y.min()
    nx = int((x.max() - xmin) // grid_um) + 1
    ny = int((y.max() - ymin) // grid_um) + 1
    column = np.clip(((x - xmin) / grid_um).astype(np.int64), 0, nx - 1)
    row = np.clip(((y - ymin) / grid_um).astype(np.int64), 0, ny - 1)
    bin_id = row * nx + column
    n_bins = nx * ny
    code_of = {name: code for code, name in enumerate(factor_order)}
    group_pixels = {}
    for group, types in groups.items():
        in_group = np.isin(top_factor, [code_of[name] for name in types])
        group_pixels[group] = np.bincount(bin_id[in_group], minlength=n_bins).astype(np.int32)
    npix = np.bincount(bin_id, minlength=n_bins).astype(np.int32)

    gene_counts = {gene: np.zeros(n_bins, np.float64) for gene in genes}
    total = np.zeros(n_bins, np.float64)
    reader = pd.read_csv(
        transcripts,
        sep="\t",
        usecols=["X", "Y", "gene", "Count"],
        dtype={"X": np.float32, "Y": np.float32, "gene": "category", "Count": np.float32},
        chunksize=chunksize,
    )
    for chunk in reader:
        if spillover is not None:
            chunk = chunk[~np.isin(bin_key(chunk["X"].values, chunk["Y"].values), spillover)]
        column = np.clip(((chunk["X"].values - xmin) / grid_um).astype(np.int64), 0, nx - 1)
        row = np.clip(((chunk["Y"].values - ymin) / grid_um).astype(np.int64), 0, ny - 1)
        chunk_bin = row * nx + column
        counts = chunk["Count"].values.astype(np.float64)
        np.add.at(total, chunk_bin, counts)
        names = chunk["gene"].values
        for gene in genes:
            is_gene = np.asarray(names == gene)
            if is_gene.any():
                np.add.at(gene_counts[gene], chunk_bin[is_gene], counts[is_gene])

    kept = np.where(npix >= min_pixels)[0]
    bins = pd.DataFrame(
        {
            "bin": kept,
            "x_um": xmin + (kept % nx + 0.5) * grid_um,
            "y_um": ymin + (kept // nx + 0.5) * grid_um,
            "npix": npix[kept],
            "total_counts": total[kept],
            **{gene: gene_counts[gene][kept] for gene in genes},
        }
    )
    for group in groups:
        bins[f"frac_{group}"] = group_pixels[group][kept] / npix[kept]
    safe_total = np.clip(bins["total_counts"].values, 1.0, None)
    for gene in genes:
        bins[f"{gene}_cp1k"] = 1000.0 * bins[gene].values / safe_total
    return bins


def niche_metrics(bins, neighbourhoods, rich, radius_um=30.0) -> pd.DataFrame:
    """Describe the surroundings of every bin.

    Adds, per group in `neighbourhoods`, the pixel-weighted fraction of the group within `radius_um`
    (``nbhd_<group>``) and, per group in `rich`, the distance to the nearest bin whose fraction of the group
    reaches the threshold (``dist_<group>_um``; NaN when the section has fewer than three such bins).

    Parameters
    ----------
    bins
        Bin table of :func:`niche_bins`; the columns are added in place and the table is returned.
    neighbourhoods
        Groups with a neighbourhood fraction.
    rich
        Mapping of group -> fraction above which a bin is rich in the group.
    """
    xy = bins[["x_um", "y_um"]].values
    npix = bins["npix"].values.astype(float)
    neighbours = cKDTree(xy).query_ball_point(xy, r=radius_um, workers=1)
    for group in neighbourhoods:
        group_pixels = bins[f"frac_{group}"].values * npix
        nbhd_group = np.empty(len(bins))
        nbhd_total = np.empty(len(bins))
        for i, members in enumerate(neighbours):
            nbhd_group[i] = group_pixels[members].sum()
            nbhd_total[i] = npix[members].sum()
        bins[f"nbhd_{group}"] = nbhd_group / np.clip(nbhd_total, 1, None)
    for group, threshold in rich.items():
        rich_xy = xy[bins[f"frac_{group}"].values >= threshold]
        bins[f"dist_{group}_um"] = cKDTree(rich_xy).query(xy, workers=1)[0] if len(rich_xy) >= 3 else np.nan
    return bins


def gene_high(bins, gene, cm_fraction=0.5, quantile=0.9):
    """Cardiomyocyte bins with a high density of a gene.

    A cardiomyocyte bin has at least `cm_fraction` cardiomyocyte pixels. Gene-high bins are the cardiomyocyte
    bins at or above the `quantile` of the non-zero densities (of all densities when 20 or fewer are non-zero).

    Returns
    -------
    Boolean mask over the rows of `bins`, and the density threshold (counts per 1,000).
    """
    cardiomyocyte = bins["frac_CM"].values >= cm_fraction
    density = bins[f"{gene}_cp1k"].values
    cm_density = density[cardiomyocyte]
    positive = cm_density[cm_density > 0]
    if positive.size > 20:
        threshold = np.quantile(positive, quantile)
    else:
        threshold = np.quantile(cm_density, quantile) if cm_density.size else np.inf
    return cardiomyocyte & (density >= threshold) & (density > 0), threshold


def compare_high_low(bins, gene, metrics, cm_fraction=0.5, quantile=0.9, min_cm_bins=50, min_group=10) -> dict:
    """Compare gene-high with the other cardiomyocyte bins of one section on each niche metric.

    One-sided Mann-Whitney U per metric, in the direction given in `metrics` (metric -> ``"greater"`` or
    ``"less"``). Returns the group sizes, the threshold and, per metric, medians, means, U, p and the log2 ratio
    of the means.
    """
    cardiomyocyte = bins["frac_CM"].values >= cm_fraction
    summary = {"gene": gene, "n_CM_bins": int(cardiomyocyte.sum())}
    if cardiomyocyte.sum() < min_cm_bins:
        summary["status"] = "too_few_CM_bins"
        return summary
    high, threshold = gene_high(bins, gene, cm_fraction, quantile)
    high = high[cardiomyocyte]
    cm_bins = bins[cardiomyocyte]
    summary["thr_cp1k"] = float(threshold)
    summary["n_high"] = int(high.sum())
    summary["n_low"] = int((~high).sum())
    results = {}
    for metric, side in metrics.items():
        if metric not in cm_bins.columns:
            continue
        a = cm_bins[metric].values[high]
        b = cm_bins[metric].values[~high]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < min_group or len(b) < min_group:
            continue
        try:
            u, p = stats.mannwhitneyu(a, b, alternative=side)
        except ValueError:
            u, p = np.nan, np.nan
        results[metric] = {
            "high_median": float(np.median(a)),
            "low_median": float(np.median(b)),
            "high_mean": float(np.mean(a)),
            "low_mean": float(np.mean(b)),
            "alt": side,
            "mw_U": float(u),
            "mw_p": float(p),
            "log2_ratio": float(np.log2((np.mean(a) + 1e-9) / (np.mean(b) + 1e-9))),
        }
    summary["metrics"] = results
    return summary


def _fdr_bh(pvals):
    """Benjamini-Hochberg q values; NaN p values stay NaN."""
    p = np.asarray(pvals, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def _permuted_mean_difference(values, high, n_perm, seed, side):
    """Label-permutation null of the difference in mean between `high` and the other values.

    Returns (observed difference, permutation p, z of the observed difference against the null).
    """
    n_high = int(high.sum())
    n = values.size
    if n_high < 5 or n - n_high < 5:
        return np.nan, np.nan, np.nan
    observed = values[high].mean() - values[~high].mean()
    rng = np.random.default_rng(seed)
    total = values.sum()
    null = np.empty(n_perm)
    index = np.arange(n)
    for i in range(n_perm):
        chosen = rng.choice(index, size=n_high, replace=False)
        null[i] = values[chosen].mean() - (total - values[chosen].sum()) / (n - n_high)
    z = (observed - null.mean()) / (null.std() + 1e-12)
    if side == "greater":
        p = (1 + np.sum(null >= observed)) / (n_perm + 1)
    else:
        p = (1 + np.sum(null <= observed)) / (n_perm + 1)
    return float(observed), float(p), float(z)


def niche_tests(
    bins,
    genes,
    metrics,
    endo_sections,
    labels=None,
    cm_fraction=0.5,
    quantile=0.9,
    n_perm=2000,
    seed=0,
    alpha=0.05,
    min_group=10,
    log=None,
) -> pd.DataFrame:
    """Test gene-high against the other cardiomyocyte bins on every section, gene and niche metric.

    Per test: one-sided Mann-Whitney U and a label-permutation null of the difference in means (`n_perm`
    shuffles of the high / low labels among the cardiomyocyte bins). Both sets of p values are adjusted over all
    tests (Benjamini-Hochberg).

    Parameters
    ----------
    bins
        Mapping of section -> bin table with the niche metrics (:func:`niche_metrics`).
    genes
        Genes that define the high bins.
    metrics
        Mapping of metric -> side of the test (``"greater"``: higher in gene-high bins; ``"less"``: lower).
    endo_sections
        Sections that contain the endocardial surface; the endocardial metrics are ``testable`` only there.
    labels
        Mapping of section -> ``{"short": ..., "donor": ...}`` written to the table.

    Returns
    -------
    One row per section, gene and metric with group sizes, means, medians, log2 ratio, p and q values.
    """
    rows = []
    for section, table in bins.items():
        cardiomyocyte = table["frac_CM"].values >= cm_fraction
        cm_bins = table[cardiomyocyte]
        for gene in genes:
            high, threshold = gene_high(table, gene, cm_fraction, quantile)
            high = high[cardiomyocyte]
            n_high, n_low = int(high.sum()), int((~high).sum())
            for metric, side in metrics.items():
                if metric not in cm_bins.columns:
                    continue
                values = cm_bins[metric].values.astype(float)
                ok = np.isfinite(values)
                a = values[ok & high]
                b = values[ok & (~high)]
                if len(a) < min_group or len(b) < min_group:
                    continue
                try:
                    _, p_mw = stats.mannwhitneyu(a, b, alternative=side)
                except ValueError:
                    p_mw = np.nan
                observed, p_perm, z = _permuted_mean_difference(values[ok], high[ok], n_perm, seed, side)
                rows.append(
                    {
                        "slide": section,
                        **(labels[section] if labels else {}),
                        "gene": gene,
                        "metric": metric,
                        "side": side,
                        "n_high": n_high,
                        "n_low": n_low,
                        "high_mean": float(a.mean()),
                        "low_mean": float(b.mean()),
                        "high_median": float(np.median(a)),
                        "low_median": float(np.median(b)),
                        "log2_ratio": float(np.log2((a.mean() + 1e-9) / (b.mean() + 1e-9))),
                        "mw_p": float(p_mw),
                        "perm_obs_meandiff": observed,
                        "perm_p": p_perm,
                        "perm_z": z,
                    }
                )
            if log:
                log(f"{section} {gene}: n_high={n_high} thr_cp1k={threshold:.3f}")
    pool = pd.DataFrame(rows)
    pool["mw_q"] = _fdr_bh(pool["mw_p"].values)
    pool["perm_q"] = _fdr_bh(pool["perm_p"].values)
    pool["sig_mw"] = pool["mw_q"] < alpha
    pool["sig_perm"] = pool["perm_q"] < alpha
    pool["on_endo_section"] = pool.slide.isin(endo_sections)
    pool["testable"] = ~(pool.metric.str.contains("endocardial") & ~pool.on_endo_section)
    return pool


def niche_consistency(pool, metric, sections, alpha=0.05) -> dict:
    """Per gene: sections tested for a metric, sections with q < `alpha` in the expected direction, median effect.

    The direction is judged on the means for neighbourhood fractions (side ``"greater"``) and on the medians for
    distances (side ``"less"``).
    """
    tests = pool[(pool.metric == metric) & (pool.slide.isin(sections))]
    summary = {}
    for gene in pd.unique(pool.gene):
        of_gene = tests[tests.gene == gene]
        n = len(of_gene)
        if n and of_gene.side.iloc[0] == "less":
            significant = (of_gene.mw_q < alpha) & (of_gene.high_median < of_gene.low_median)
        else:
            significant = (of_gene.mw_q < alpha) & (of_gene.high_mean > of_gene.low_mean)
        summary[gene] = {
            "n_slides": int(n),
            "n_sig": int(significant.sum()),
            "med_log2": float(np.nanmedian(of_gene.log2_ratio)) if n else np.nan,
            "med_perm_z": float(np.nanmedian(of_gene.perm_z)) if n else np.nan,
        }
    return summary
