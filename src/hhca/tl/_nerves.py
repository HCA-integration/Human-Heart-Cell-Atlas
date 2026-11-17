"""Nerves on the 2 um bins: calling them, their counts, their distance to vessels and their myelination."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from hhca.tl._bins import _as_sections, _coordinates, _label_codes


def _of_types(adata, types) -> np.ndarray:
    """Mask of the bins whose Level-3 label is one of `types`."""
    categories, codes = _label_codes(adata)
    types = [types] if isinstance(types, str) else list(types)
    return np.isin(codes, np.where(np.isin(categories, types))[0])


def call_nerves(adata, neural_type="Neural Cells", linkage_um=6.0, min_bins=3) -> pd.DataFrame:
    """Group the neural 2 um bins of one section into nerves.

    Neural bins closer than `linkage_um` are linked; a nerve is a group of at least `min_bins` linked bins
    (DBSCAN with ``eps=linkage_um`` and ``min_samples=min_bins`` on the bin centres).

    Parameters
    ----------
    adata
        One section (in memory or backed).
    neural_type
        Level-3 label of the neural bins.

    Returns
    -------
    One row per neural bin, indexed by its row in `adata`: ``x_um``, ``y_um``, ``nerve`` (id of its nerve, -1
    for a bin outside any nerve) and ``comp_size`` (number of bins it is linked with, itself included).
    """
    from sklearn.cluster import DBSCAN

    x, y = _coordinates(adata)
    rows = np.where(_of_types(adata, neural_type))[0]
    xy = np.c_[x[rows], y[rows]]
    nerve = DBSCAN(eps=linkage_um, min_samples=min_bins).fit(xy).labels_
    component = DBSCAN(eps=linkage_um, min_samples=1).fit_predict(xy)
    comp_size = np.bincount(component)[component]
    return pd.DataFrame({"x_um": xy[:, 0], "y_um": xy[:, 1], "nerve": nerve, "comp_size": comp_size}, index=rows)


def nerve_pseudobulk(adata, neural_bins, myelin_genes, vascular_types) -> ad.AnnData:
    """Sum the counts of the member bins of every nerve of one section.

    Parameters
    ----------
    adata
        One section (in memory or backed).
    neural_bins
        Neural bins of the section with their nerve (:func:`hhca.tl.call_nerves`).
    myelin_genes
        Genes whose counts are added up as ``myelin_total``.
    vascular_types
        Level-3 labels of the vascular bins for ``d_vessel_um``.

    Returns
    -------
    AnnData of nerves x genes with the summed counts in ``X`` and ``layers["counts"]``. ``obs``: ``nerve``,
    ``n_bins``, ``myelin_total``, ``total_counts``, the centroid ``cx``, ``cy`` and ``d_vessel_um``, the
    smallest distance from a member bin to a vascular bin.
    """
    x, y = _coordinates(adata)
    nerve = neural_bins["nerve"].to_numpy()
    neural_rows = neural_bins.index.to_numpy()
    neural_xy = neural_bins[["x_um", "y_um"]].to_numpy()
    in_nerve = nerve >= 0
    n_nerves = int(nerve.max() + 1)

    member = np.where(in_nerve)[0]
    order = np.argsort(neural_rows[member])
    nerve_of_row = nerve[member][order]
    counts = sp.csr_matrix(adata.X[neural_rows[member][order]])
    membership = sp.csr_matrix(
        (np.ones(len(nerve_of_row), np.float32), (nerve_of_row, np.arange(len(nerve_of_row)))),
        shape=(n_nerves, len(nerve_of_row)),
    )
    pseudobulk = (membership @ counts).tocsr().astype(np.float32)

    var_names = adata.var_names.to_numpy().astype(str)
    column = {gene: i for i, gene in enumerate(var_names)}
    myelin_columns = [column[gene] for gene in myelin_genes if gene in column]
    n_bins = np.bincount(nerve[in_nerve], minlength=n_nerves)
    d_vessel = np.full(n_nerves, np.nan)
    vascular = _of_types(adata, vascular_types)
    if vascular.any():
        d_bin = cKDTree(np.c_[x[vascular], y[vascular]]).query(neural_xy[in_nerve])[0]
        d_vessel = pd.Series(d_bin).groupby(nerve[in_nerve]).min().reindex(range(n_nerves)).to_numpy()
    obs = pd.DataFrame(
        {
            "nerve": np.arange(n_nerves),
            "n_bins": n_bins,
            "myelin_total": np.asarray(pseudobulk[:, myelin_columns].sum(1), dtype=np.float64).ravel(),
            "total_counts": np.asarray(pseudobulk.sum(1), dtype=np.float64).ravel(),
            "cx": np.bincount(nerve[in_nerve], weights=neural_xy[in_nerve, 0], minlength=n_nerves) / n_bins,
            "cy": np.bincount(nerve[in_nerve], weights=neural_xy[in_nerve, 1], minlength=n_nerves) / n_bins,
            "d_vessel_um": d_vessel,
        },
        index=[str(i) for i in range(n_nerves)],
    )
    nerves = ad.AnnData(pseudobulk, obs=obs, var=pd.DataFrame(index=var_names))
    nerves.layers["counts"] = nerves.X.copy()
    return nerves


def size_class_labels(edges) -> list[str]:
    """Labels of the nerve size classes with the lower edges `edges`: ``3–5``, ``6–9``, ..., ``≥40``."""
    labels = [f"{low}–{high - 1}" for low, high in zip(edges[:-2], edges[1:-1])]
    return [*labels, f"≥{edges[-2]}"]


def nerve_myelination(nerves, myelinated_min=3, size_edges=(3, 6, 10, 20, 40, 10000)) -> pd.DataFrame:
    """Myelination score and size class of every nerve.

    Parameters
    ----------
    nerves
        Per-nerve table with ``n_bins``, ``myelin_total`` and ``slide`` (the ``obs`` of the nerve pseudobulks).
    myelinated_min
        A nerve with at least this many myelin counts is myelinating.
    size_edges
        Edges of the size classes in bins (left-closed); nerves outside them are dropped.

    Returns
    -------
    One row per nerve: ``n_bins``, ``myelin_total``, ``slide``, ``myelinating``, ``szbin`` (size class) and
    ``logmy = log1p(myelin_total)``.
    """
    table = nerves[["n_bins", "myelin_total", "slide"]].reset_index(drop=True)
    table = table[table.n_bins >= size_edges[0]].copy()
    table["myelinating"] = table.myelin_total >= myelinated_min
    table["szbin"] = pd.cut(table.n_bins, bins=list(size_edges), labels=size_class_labels(size_edges), right=False)
    table = table.dropna(subset=["szbin"])
    table["logmy"] = np.log1p(table.myelin_total)
    return table


def nerve_vessel_distance(
    sections,
    neural_bins,
    vascular_types,
    neural_type="Neural Cells",
    placements=5,
    seed=0,
    max_tries=50,
    min_on_tissue=0.9,
    on_tissue_um=1.5,
) -> dict[str, pd.DataFrame]:
    """Distance of every nerve to the nearest vascular bin, and of the same nerve shapes at random positions.

    The distance of a nerve is the smallest distance from one of its bins to a vascular bin. For the background
    the bins of a nerve are moved together so that its medoid bin lands on a random bin of the section that is
    neither neural nor vascular; a placement is kept when at least `min_on_tissue` of the moved bins lie within
    `on_tissue_um` of a bin of the section, and up to `max_tries` positions are tried. The placements have the
    shape and size of the nerve, so the size dependence of a minimum distance cancels.

    Parameters
    ----------
    sections
        One section, or a mapping of name -> section; the random positions are drawn section after section from
        one generator.
    neural_bins
        Neural bins with their nerve (:func:`hhca.tl.call_nerves`), one table per section.
    vascular_types
        Level-3 labels of the vascular bins.
    placements
        Random placements per nerve.

    Returns
    -------
    ``nerves``: one row per nerve (``slide``, ``nerve``, ``n_bins``, ``d_vessel_um``); ``placements``: one row
    per kept placement (``nerve_index`` = row of its nerve in ``nerves``, ``d_vessel_um``); ``sections``: nerves
    and kept and failed placements per section.
    """
    sections = _as_sections(sections)
    neural_bins = neural_bins if isinstance(neural_bins, dict) else {"section": neural_bins}
    rng = np.random.default_rng(seed)
    nerve_rows, d_background, background_nerve, per_section = [], [], [], []
    for name, adata in sections.items():
        x, y = _coordinates(adata)
        xy = np.c_[x, y]
        vascular = _of_types(adata, vascular_types)
        neural = _of_types(adata, neural_type)
        vessel_tree = cKDTree(xy[vascular])
        tissue_tree = cKDTree(xy)
        candidates = np.where(~vascular & ~neural)[0]
        members = neural_bins[name]
        members = members[members.nerve >= 0]
        n_kept = n_failed = 0
        for nerve, bins in members.groupby("nerve", sort=True):
            points = bins[["x_um", "y_um"]].to_numpy(float)
            centroid = points.mean(0)
            medoid = points[np.argmin(((points - centroid) ** 2).sum(1))]
            nerve_rows.append((name, nerve, len(points), float(vessel_tree.query(points)[0].min())))
            for _ in range(placements):
                for _ in range(max_tries):
                    anchor = xy[candidates[rng.integers(len(candidates))]]
                    moved = points + (anchor - medoid)
                    on_tissue = tissue_tree.query(moved, distance_upper_bound=on_tissue_um)[0] < np.inf
                    if on_tissue.mean() >= min_on_tissue:
                        d_background.append(float(vessel_tree.query(moved)[0].min()))
                        background_nerve.append(len(nerve_rows) - 1)
                        n_kept += 1
                        break
                else:
                    n_failed += 1
        per_section.append(
            {
                "slide": name,
                "n_nerves": int(members.nerve.nunique()),
                "placements_kept": n_kept,
                "placements_failed": n_failed,
            }
        )
    return {
        "nerves": pd.DataFrame(nerve_rows, columns=["slide", "nerve", "n_bins", "d_vessel_um"]),
        "placements": pd.DataFrame({"nerve_index": background_nerve, "d_vessel_um": d_background}),
        "sections": pd.DataFrame(per_section),
    }


def proximity_by_size(nerves, placements, size_edges, near_um=8.0) -> pd.DataFrame:
    """Distance to vessels of the nerves and their random placements, per nerve size class.

    Parameters
    ----------
    nerves, placements
        Tables of :func:`hhca.tl.nerve_vessel_distance`.
    size_edges
        Edges of the size classes in bins (left-closed).
    near_um
        Distance below which a nerve counts as next to a vessel.

    Returns
    -------
    One row per size class: ``size_class``, ``n_nerves``, the median distance and the percentage within
    `near_um` (``pct8_*`` for 8 um) of the nerves and of the placements.
    """
    d_nerve, n_bins = nerves["d_vessel_um"].to_numpy(), nerves["n_bins"].to_numpy()
    d_background = placements["d_vessel_um"].to_numpy()
    n_bins_background = n_bins[placements["nerve_index"].to_numpy()]
    rows = []
    for low, high, label in zip(size_edges[:-1], size_edges[1:], size_class_labels(size_edges)):
        of_class = (n_bins >= low) & (n_bins < high)
        of_class_background = (n_bins_background >= low) & (n_bins_background < high)
        rows.append(
            {
                "size_class": label,
                "n_nerves": int(of_class.sum()),
                "median_nerve_um": float(np.median(d_nerve[of_class])),
                "median_background_um": float(np.median(d_background[of_class_background])),
                f"pct{near_um:g}_nerve": float(100 * (d_nerve[of_class] <= near_um).mean()),
                f"pct{near_um:g}_background": float(100 * (d_background[of_class_background] <= near_um).mean()),
            }
        )
    return pd.DataFrame(rows)


def myelination_groups(nerves, myelinated_min=3) -> None:
    """Label every nerve as myelinated, non-myelinated or intermediate.

    Adds ``nerves.obs["group"]``: ``myelinated`` with at least `myelinated_min` myelin counts,
    ``non_myelinated`` with none and ``intermediate`` in between.
    """
    myelin = nerves.obs["myelin_total"]
    nerves.obs["group"] = np.where(
        myelin >= myelinated_min, "myelinated", np.where(myelin == 0, "non_myelinated", "intermediate")
    )
