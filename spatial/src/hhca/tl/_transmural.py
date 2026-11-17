"""Expression across the ventricular wall: transmural coordinate, wall profiles and gene modules (Fig 3 i).

A donor is sampled by up to three sections (near the epicardium, mid-myocardium, near the endocardium). Within a
section, every cardiomyocyte bin gets a coordinate between the epicardial (0) and the endocardial (1) face; the
sections are then put one after the other, so the wall of a donor runs from 0 to 3 and is cut into equal bins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter1d, label
from scipy.sparse.linalg import cg, spsolve
from scipy.stats import spearmanr

from hhca.tl._bins import celltype_bins

LAYERS = ("epi", "myo", "endo")


def _zscore(values):
    """z-score ignoring NaN; zeros when the values are constant."""
    values = np.asarray(values, float)
    sd = np.nanstd(values)
    return (values - np.nanmean(values)) / sd if sd > 0 else values * 0.0


def _solve_laplace(tissue, epi, endo):
    """Solve the Laplace equation on the tissue mask with phi = 0 on the epi cells and 1 on the endo cells."""
    ny, nx = tissue.shape
    free = tissue & ~epi & ~endo
    idx = -np.ones((ny, nx), int)
    fy, fx = np.where(free)
    n_free = fy.size
    idx[fy, fx] = np.arange(n_free)
    rows = [np.arange(n_free)]
    cols = [np.arange(n_free)]
    b = np.zeros(n_free)
    deg = np.zeros(n_free)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        yy, xx = fy + dy, fx + dx
        inb = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
        yy2 = np.clip(yy, 0, ny - 1)
        xx2 = np.clip(xx, 0, nx - 1)
        nb_tis = inb & tissue[yy2, xx2]
        deg += nb_tis
        b[nb_tis & endo[yy2, xx2]] += 1.0
        nb_free = nb_tis & free[yy2, xx2]
        k = np.where(nb_free)[0]
        rows.append(k)
        cols.append(idx[yy2[k], xx2[k]])
    data = [deg] + [-np.ones(len(c)) for c in cols[1:]]
    A = sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n_free, n_free))
    preconditioner = sp.diags(1.0 / A.diagonal())
    sol, info = cg(A.tocsr(), b, rtol=1e-7, maxiter=20000, M=preconditioner)
    if info != 0:
        sol = spsolve(A.tocsc(), b)
    phi = np.full((ny, nx), np.nan)
    phi[fy, fx] = np.clip(sol, 0, 1)
    phi[epi] = 0.0
    phi[endo] = 1.0
    return phi


def marker_axis(xy, score):
    """Epi -> endo direction of a section from a marker score (low = epi, high = endo).

    The direction is the unit gradient of a plane fitted to the score, in image coordinates (+y down).
    Returns (ux, uy, R2); R2 is low on partial or patchy blocks.
    """
    ok = np.isfinite(score)
    X = xy[ok]
    Y = score[ok]
    Xc = X - X.mean(0)
    A = np.column_stack([Xc[:, 0], Xc[:, 1], np.ones(len(Xc))])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    pred = A @ coef
    r2 = 1 - ((Y - pred) ** 2).sum() / (((Y - Y.mean()) ** 2).sum() + 1e-12)
    g = coef[:2]
    n = np.linalg.norm(g)
    if n == 0:
        return 1.0, 0.0, 0.0
    return float(g[0] / n), float(g[1] / n), float(r2)


def laplace_coordinate(xy, direction, grid_um=40.0, face_fraction=0.08):
    """Coordinate of every bin between the two faces of a section, from a Laplace solution.

    The tissue is rasterised on a `grid_um` grid (closed, holes filled, largest piece). Its boundary cells at
    the low end of the projection onto `direction` are the epicardial face (0), those at the high end the
    endocardial face (1); `face_fraction` is the share of the projected span that counts as a face. The Laplace
    equation is solved in between.

    Parameters
    ----------
    xy
        Bin coordinates in um.
    direction
        Unit (dx, dy) in image coordinates pointing from epicardium to endocardium.

    Returns
    -------
    Coordinate per bin; NaN for bins outside the largest connected tissue piece.
    """
    x, y = xy[:, 0], xy[:, 1]
    xmin, ymin = x.min(), y.min()
    nx = int((x.max() - xmin) / grid_um) + 1
    ny = int((y.max() - ymin) / grid_um) + 1
    ix = np.clip(((x - xmin) / grid_um).astype(int), 0, nx - 1)
    iy = np.clip(((y - ymin) / grid_um).astype(int), 0, ny - 1)

    # tissue mask: occupied grid cells, closed, holes filled, largest connected piece
    mask = (np.bincount(iy * nx + ix, minlength=ny * nx) > 0).reshape(ny, nx)
    mask = binary_closing(mask, iterations=3)
    mask = binary_fill_holes(mask)
    pieces, n_pieces = label(mask)
    if n_pieces > 1:
        sizes = np.bincount(pieces.ravel())
        sizes[0] = 0
        mask = pieces == sizes.argmax()
    mask = binary_fill_holes(mask)

    yy_c, xx_c = np.mgrid[0:ny, 0:nx].astype(float)
    dx, dy = direction
    # projection (in grid-cell units) onto the epi -> endo axis
    s = xx_c * dx + yy_c * dy
    s_t = s[mask]
    smin, smax = s_t.min(), s_t.max()
    span = smax - smin + 1e-9

    inner = np.zeros_like(mask)
    inner[1:-1, 1:-1] = mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    boundary = mask & ~inner
    epi = boundary & (s <= smin + face_fraction * span)  # low projection = epi
    endo = boundary & (s >= smax - face_fraction * span)  # high projection = endo

    phi = _solve_laplace(mask, epi, endo)
    return phi[iy, ix]


def _marker_score(cm, markers):
    """Mean z-score over the marker genes that are present, per bin."""
    column_of = {gene: i for i, gene in enumerate(cm.var_names)}
    scores = [
        _zscore(np.asarray(cm.X.getcol(column_of[gene]).todense()).ravel().astype(np.float64))
        for gene in markers
        if gene in column_of
    ]
    return np.nanmean(np.vstack(scores), axis=0) if scores else np.zeros(cm.n_obs)


def transmural_coordinate(cm, endo_markers, epi_markers, grid_um=40.0, face_fraction=0.08) -> None:
    """Give every bin of a section its position between the epicardial and the endocardial face.

    The direction from epicardium to endocardium is the gradient of a marker score (mean z-score of the
    endocardial markers minus that of the epicardial markers; :func:`marker_axis`); the coordinate is the
    Laplace solution between the two faces along it (:func:`laplace_coordinate`).

    Adds ``cm.obs["phi"]`` (0 = epicardial face, 1 = endocardial face, NaN outside the main tissue piece) and
    ``cm.uns["transmural"]`` with the direction and the R2 of the marker plane.

    Parameters
    ----------
    cm
        Cardiomyocyte bins of one section with log1p CP10k in ``X`` (:func:`hhca.tl.celltype_bins`).
    """
    xy = cm.obsm["spatial"]
    score = _marker_score(cm, endo_markers) - _marker_score(cm, epi_markers)
    ux, uy, r2 = marker_axis(xy, score)
    cm.obs["phi"] = laplace_coordinate(xy, (ux, uy), grid_um, face_fraction).astype(float)
    cm.uns["transmural"] = {"direction": [ux, uy], "r2": r2}


def wall_bins(cm, layer, layers=LAYERS, bins_per_section=15) -> np.ndarray:
    """Bin of the wall (0 = epicardial end) of every bin of a section; bins without a coordinate get -1.

    The section at position `layer` covers ``bins_per_section`` of the ``len(layers) * bins_per_section`` bins.
    """
    n_bins = len(layers) * bins_per_section
    edges = np.linspace(0, len(layers), n_bins + 1)
    phi = cm.obs["phi"].to_numpy()
    wall_bin = np.clip(np.digitize(list(layers).index(layer) + phi, edges) - 1, 0, n_bins - 1)
    return np.where(np.isfinite(phi), wall_bin, -1).astype(np.int64)


def transmural_profiles(
    sections,
    endo_markers,
    epi_markers,
    layers=LAYERS,
    bins_per_section=15,
    cell_type="Ventricular Cardiomyocytes",
    grid_um=40.0,
    face_fraction=0.08,
    log=None,
) -> dict:
    """Expression of every gene along the wall of one donor.

    For each section: the bins of `cell_type` are log-normalised (:func:`hhca.tl.celltype_bins`), placed between
    the two faces (:func:`transmural_coordinate`) and assigned to wall bins (:func:`wall_bins`); log1p CP10k is
    summed per wall bin.

    Parameters
    ----------
    sections
        Mapping of layer (``"epi"``, ``"myo"``, ``"endo"``) -> section of the donor; a missing layer leaves its
        wall bins empty.

    Returns
    -------
    Dictionary with ``genes``, ``bin_sum`` (wall bins x genes), ``bin_count`` (bins per wall bin), ``nnz``
    (cardiomyocyte bins with a count, per gene), ``n_cm``, ``layers`` and ``diagnostics`` (one row per section).
    """
    n_bins = len(layers) * bins_per_section
    genes = bin_sum = nnz = None
    bin_count = np.zeros(n_bins, np.float64)
    n_cm_total = 0
    diagnostics = []
    for layer, adata in sections.items():
        cm = celltype_bins(adata, cell_type)
        if genes is None:
            genes = cm.var_names.to_numpy().astype(str)
            bin_sum = np.zeros((n_bins, genes.size), np.float64)
            nnz = np.zeros(genes.size, np.int64)
        nnz += cm.var["n_bins"].to_numpy()
        n_cm_total += cm.n_obs
        transmural_coordinate(cm, endo_markers, epi_markers, grid_um, face_fraction)
        wall_bin = wall_bins(cm, layer, layers, bins_per_section)
        wall_bin = np.where(wall_bin < 0, n_bins, wall_bin)
        membership = sp.csr_matrix(
            (np.ones(cm.n_obs, np.float64), (wall_bin, np.arange(cm.n_obs))), shape=(n_bins + 1, cm.n_obs)
        )
        bin_sum += np.asarray((membership @ cm.X)[:n_bins].todense())
        bin_count += np.bincount(wall_bin, minlength=n_bins + 1)[:n_bins]
        diagnostics.append(
            {
                "layer": layer,
                "n_cm": cm.n_obs,
                "marker_axis_R2": float(cm.uns["transmural"]["r2"]),
                "phi_ok": int(np.isfinite(cm.obs["phi"]).sum()),
            }
        )
        if log:
            log(f"{layer}: n_cm={cm.n_obs} R2={cm.uns['transmural']['r2']:.2e} phi_ok={diagnostics[-1]['phi_ok']}")
    return {
        "genes": genes,
        "bin_sum": bin_sum,
        "bin_count": bin_count,
        "nnz": nnz,
        "n_cm": np.int64(n_cm_total),
        "layers": np.array(list(sections)),
        "diagnostics": diagnostics,
    }


def profile_means(profile) -> pd.DataFrame:
    """Mean log1p CP10k per wall bin (rows) and gene (columns) of a donor; empty wall bins are NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = profile["bin_sum"] / profile["bin_count"][:, None]
    return pd.DataFrame(mean, columns=np.asarray(profile["genes"]).astype(str))


def _smooth(values, sigma):
    """Gaussian smoothing within each run of finite values (gaps stay open)."""
    out = np.full_like(values, np.nan, float)
    finite = np.isfinite(values)
    i, n = 0, len(values)
    while i < n:
        if finite[i]:
            j = i
            while j < n and finite[j]:
                j += 1
            out[i:j] = values[i:j] if j - i == 1 else gaussian_filter1d(values[i:j], sigma, mode="nearest")
            i = j
        else:
            i += 1
    return out


def _interpolate(values):
    """Fill empty bins by linear interpolation."""
    position = np.arange(len(values))
    good = np.isfinite(values)
    return np.interp(position, position[good], values[good]) if good.sum() >= 2 else values


def transmural_rho(profiles, min_detection=0.002, sigma=1.3) -> pd.DataFrame:
    """Rank genes by how consistently their expression changes across the wall.

    Per gene and donor, rho is the Spearman correlation between the wall-bin index and the z-scored, smoothed
    profile. The first two donors are sampled across the whole wall; a third donor may lack the mid-myocardial
    section, its rho is then taken over the sampled bins only. The consensus is the mean rho over the donors in
    which the gene is measured, and a gene is sign-consistent when rho has the same sign in all of them.

    Parameters
    ----------
    profiles
        Mapping of donor -> profile (:func:`transmural_profiles`); the first two donors share one gene space.
    min_detection
        Fraction of the cardiomyocyte bins of the first two donors with a count that a gene needs to be
        ``in_universe``.
    sigma
        Width of the Gaussian smoothing, in wall bins.

    Returns
    -------
    One row per detected gene with the rho of each donor, the 2- and 3-donor consensus and the sign consistency.
    """
    first, second, third = list(profiles)
    means = {donor: profile_means(profile) for donor, profile in profiles.items()}
    genes = np.asarray(profiles[first]["genes"]).astype(str)
    detection = (profiles[first]["nnz"] + profiles[second]["nnz"]) / (
        profiles[first]["n_cm"] + profiles[second]["n_cm"]
    )
    columns = {donor: {gene: i for i, gene in enumerate(means[donor].columns)} for donor in profiles}
    values = {donor: means[donor].to_numpy() for donor in profiles}
    n_bins = values[first].shape[0]
    n_per = n_bins // 3
    wall_bin = np.arange(n_bins)
    rows = []
    for position, gene in enumerate(genes):
        if detection[position] <= 0:
            continue
        rho = {}
        for donor in (first, second):
            profile = _smooth(_zscore(_interpolate(values[donor][:, columns[donor][gene]])), sigma)
            rho[donor] = spearmanr(wall_bin, profile).correlation
        # third donor: rho over its sampled bins only (the mid-wall gap stays open)
        if gene in columns[third]:
            profile = _smooth(_zscore(values[third][:, columns[third][gene]]), sigma)
            finite = np.isfinite(profile)
            rho_third = spearmanr(wall_bin[finite], profile[finite]).correlation if finite.sum() >= 3 else np.nan
            epi_block, endo_block = profile[0:n_per], profile[2 * n_per : 3 * n_per]
            epi_mean = np.nanmean(epi_block) if np.isfinite(epi_block).any() else np.nan
            endo_mean = np.nanmean(endo_block) if np.isfinite(endo_block).any() else np.nan
            difference = endo_mean - epi_mean
        else:
            rho_third = difference = np.nan
        rho1, rho2 = rho[first], rho[second]
        consensus2 = np.nanmean([rho1, rho2])
        sign2 = bool((rho1 > 0 and rho2 > 0) or (rho1 < 0 and rho2 < 0))
        measured = [rho1, rho2] + ([rho_third] if np.isfinite(rho_third) else [])
        consensus3 = float(np.nanmean(measured))
        # a gene that is not in the gene space of the third donor is judged on the other two
        sign3 = bool(all(r > 0 for r in measured) or all(r < 0 for r in measured)) if len(measured) == 3 else sign2
        rows.append(
            {
                "gene": gene,
                "det_pooled": float(detection[position]),
                "in_universe": bool(detection[position] >= min_detection),
                f"rho_{first}": float(rho1),
                f"rho_{second}": float(rho2),
                f"rho_{third}": float(rho_third) if np.isfinite(rho_third) else np.nan,
                f"{third.lower()}_in_genespace": len(measured) == 3,
                f"{third.lower()}_endo_minus_epi": float(difference) if np.isfinite(difference) else np.nan,
                "consensus_rho_2donor": float(consensus2),
                "sign_consistent_2donor": sign2,
                "consensus_rho_3donor": consensus3,
                "abs_consensus_3donor": abs(consensus3),
                "sign_consistent_3donor": sign3,
            }
        )
    return pd.DataFrame(rows)


def select_modules(ranking, n_endo=6, n_epi=4, donors=3, only=None) -> dict[str, list[str]]:
    """The genes that rise most consistently towards the endocardium and towards the epicardium.

    Among the sign-consistent genes of the universe, the `n_endo` genes with the highest and the `n_epi` genes
    with the lowest consensus rho.

    Parameters
    ----------
    ranking
        Gene table of :func:`transmural_rho`.
    donors
        Use the 2- or the 3-donor consensus.
    only
        Boolean column of `ranking` that a gene must also fulfil (e.g. ``"cm_safe"``).
    """
    consensus, consistent = f"consensus_rho_{donors}donor", f"sign_consistent_{donors}donor"
    eligible = ranking[ranking.in_universe & ranking[consistent]]
    if only is not None:
        eligible = eligible[eligible[only]]
    rising = eligible[eligible[consensus] > 0].sort_values(consensus, ascending=False)
    falling = eligible[eligible[consensus] < 0].sort_values(consensus, ascending=True)
    return {"endo": rising.head(n_endo)["gene"].tolist(), "epi": falling.head(n_epi)["gene"].tolist()}


def module_profiles(profiles, genes, mask_unsampled=True) -> dict[str, pd.DataFrame]:
    """Mean profile across the wall of selected genes, per donor (genes x wall bins).

    Empty wall bins of a donor with all three sections are filled by linear interpolation. The profile of a donor
    that lacks a section is not interpolated, and genes outside its gene space are NaN.

    Parameters
    ----------
    profiles
        Mapping of donor -> profile (:func:`transmural_profiles`).
    genes
        Genes to return.
    mask_unsampled
        Set the wall bins of a missing section to NaN. (A few bins at the face of a neighbouring section fall on
        the first of them.)
    """
    tables = {}
    for donor, profile in profiles.items():
        means = profile_means(profile)
        n_bins = len(means)
        n_per = n_bins // len(LAYERS)
        sampled = [str(layer) for layer in profile["layers"]]
        rows = []
        for gene in genes:
            if gene not in means.columns:
                rows.append(np.full(n_bins, np.nan))
                continue
            values = means[gene].to_numpy().astype(float)
            if len(sampled) == len(LAYERS):
                values = _interpolate(values)
            elif mask_unsampled:
                for position, layer in enumerate(LAYERS):
                    if layer not in sampled:
                        values[position * n_per : (position + 1) * n_per] = np.nan
            rows.append(values)
        tables[donor] = pd.DataFrame(np.vstack(rows), index=list(genes), columns=[f"bin{k}" for k in range(n_bins)])
    return tables


def consensus_profile(tables, gene, sigma=1.3):
    """Smoothed mean and spread across donors of the z-scored wall profile of a gene.

    Parameters
    ----------
    tables
        Per-donor profiles of :func:`module_profiles`.

    Returns
    -------
    Two arrays over the wall bins: the mean and the standard deviation across donors, both smoothed.
    """
    z = np.vstack([_zscore(table.loc[gene].to_numpy(float)) for table in tables.values()])
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(z, axis=0)
        spread = np.nanstd(z, axis=0)
    return _smooth(mean, sigma), _smooth(spread, sigma)


def module_rules(ranking, selection, hand_drop=()) -> dict:
    """Gene modules under different cardiomyocyte-specificity rules.

    ``hand_drop``: the selected modules without the genes in `hand_drop`; ``cm_safe_post`` / ``cm_ratio_post``:
    the selected genes that are ``cm_safe`` / have ``CM_over_topNonCM`` >= 1; ``cm_safe_pre``: the top genes
    among the ``cm_safe`` genes only (filter before the selection); ``selected_unfiltered``: the modules as
    selected.

    Parameters
    ----------
    ranking
        Gene table of :func:`transmural_rho` with the specificity columns of
        :func:`hhca.tl.cardiomyocyte_specificity` joined.
    selection
        The selected modules and their sizes (``endo_module``, ``epi_module``, ``N_ENDO``, ``N_EPI``).
    """
    ratio_ok = ranking.set_index("gene")["CM_over_topNonCM"].fillna(0) >= 1.0
    safe = ranking.set_index("gene")["cm_safe"]
    endo, epi = selection["endo_module"], selection["epi_module"]
    return {
        "hand_drop": {"endo": [gene for gene in endo if gene not in hand_drop], "epi": list(epi)},
        "cm_safe_post": {
            "endo": [gene for gene in endo if safe.get(gene, False)],
            "epi": [gene for gene in epi if safe.get(gene, False)],
        },
        "cm_ratio_post": {
            "endo": [gene for gene in endo if ratio_ok.get(gene, False)],
            "epi": [gene for gene in epi if ratio_ok.get(gene, False)],
        },
        "cm_safe_pre": select_modules(ranking, selection["N_ENDO"], selection["N_EPI"], only="cm_safe"),
        "selected_unfiltered": {"endo": list(endo), "epi": list(epi)},
    }
