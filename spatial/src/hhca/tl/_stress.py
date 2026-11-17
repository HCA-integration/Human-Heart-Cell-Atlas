"""Gene-set score of cardiomyocyte bins and the tiles and fields it is shown on (Fig 3 d)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_gene_sets(cm, gene_sets, groupby="section") -> None:
    """Score every bin as the mean z-score of the genes of a set.

    Each gene is z-scored against the mean and standard deviation over all bins of the object, so the scores of
    its sections are on one scale. Adds one ``obs`` column per gene set and the reference in
    ``cm.uns["score_reference"]``.

    Parameters
    ----------
    cm
        Bins x genes with log1p CP10k in ``X`` (:func:`hhca.tl.log_cp10k`); the bins of a section are contiguous.
    gene_sets
        Mapping of score name -> genes; genes that are not in `cm` are left out.
    groupby
        ``obs`` column with the section of each bin.
    """
    expression = cm.X
    sections = cm.obs[groupby].to_numpy()
    blocks = []
    for name in pd.unique(sections):
        rows = np.flatnonzero(sections == name)
        blocks.append(expression[rows[0] : rows[-1] + 1])
    n_bins = sum(block.shape[0] for block in blocks)
    mean = sum(block.sum(0, dtype=np.float64) for block in blocks) / n_bins
    sd = np.sqrt(sum((block.astype(np.float64) ** 2).sum(0) for block in blocks) / n_bins - mean**2)
    present = list(cm.var_names)
    for name, genes in gene_sets.items():
        columns = [present.index(gene) for gene in genes if gene in present]
        cm.obs[name] = ((expression[:, columns] - mean[columns]) / sd[columns]).mean(1).astype(np.float32)
    cm.uns["score_reference"] = {"mean": mean, "sd": sd}


def rank_tiles(cm, score, field_um=650.0, min_bins=4000) -> pd.DataFrame:
    """Cut one section into square tiles and rank them by their mean score.

    Parameters
    ----------
    cm
        Bins of one section with the score in ``obs`` (:func:`score_gene_sets`).
    score
        Name of the score.
    field_um
        Side of a tile.
    min_bins
        Bins a tile needs to be ranked.

    Returns
    -------
    One row per tile, sorted by ``mean``: ``mean``, ``count`` and the lower corner ``xlo``, ``ylo`` (um).
    """
    x, y = cm.obs["x_um"].to_numpy(), cm.obs["y_um"].to_numpy()
    column = np.floor((x - x.min()) / field_um).astype(np.int64)
    row = np.floor((y - y.min()) / field_um).astype(np.int64)
    n_rows = row.max() + 1
    tiles = pd.DataFrame({"k": column * n_rows + row, "s": cm.obs[score].to_numpy()})
    ranking = tiles.groupby("k")["s"].agg(["mean", "count"])
    ranking = ranking[ranking["count"] >= min_bins].sort_values("mean")
    ranking["xlo"] = x.min() + (ranking.index // n_rows) * field_um
    ranking["ylo"] = y.min() + (ranking.index % n_rows) * field_um
    return ranking


def tile_at(ranking, percentile, field_um=650.0) -> list[float]:
    """Field [xlo, xhi, ylo, yhi] of the tile at a percentile (0-1) of the ranking of :func:`rank_tiles`."""
    tile = ranking.iloc[int(np.clip(round(percentile * (len(ranking) - 1)), 0, len(ranking) - 1))]
    return [float(tile.xlo), float(tile.xlo + field_um), float(tile.ylo), float(tile.ylo + field_um)]


def field_grid(cm, score, field, field_um=650.0, grid_um=5.0) -> np.ndarray:
    """Mean score on a regular grid of one field, for display.

    Parameters
    ----------
    cm
        Bins of one section with the score in ``obs``.
    field
        [xlo, xhi, ylo, yhi] in um.

    Returns
    -------
    2D array (float32; rows = y ascending, columns = x); grid cells without a bin are NaN.
    """
    x, y, values = cm.obs["x_um"].to_numpy(), cm.obs["y_um"].to_numpy(), cm.obs[score].to_numpy()
    xlo, xhi, ylo, yhi = field
    inside = (x >= xlo) & (x < xhi) & (y >= ylo) & (y < yhi)
    n = int(field_um / grid_um)
    limits = [[xlo, xhi], [ylo, yhi]]
    score_sum, _, _ = np.histogram2d(x[inside], y[inside], [n, n], limits, weights=values[inside])
    n_bins, _, _ = np.histogram2d(x[inside], y[inside], [n, n], limits)
    return np.where(n_bins > 0, score_sum / np.maximum(n_bins, 1e-9), np.nan).T.astype(np.float32)
