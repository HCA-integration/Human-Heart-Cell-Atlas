"""Fields of a section: windows of 2 um bins, label rasters and crops of the registered H&E."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from hhca.grid import KEYMUL, bin_key
from hhca.tl._bins import CARDIOMYOCYTES, _coordinates, _label_codes


def _he_bounds_um(um_to_px, size_px):
    """Bounding box (x0, x1, y0, y1, um) of an H&E image of size_px = (width, height)."""
    px_to_um = np.linalg.inv(np.asarray(um_to_px, float))
    width, height = size_px
    corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]], float) @ px_to_um.T
    return corners[:, 0].min(), corners[:, 0].max(), corners[:, 1].min(), corners[:, 1].max()


def _top_fields(ranked, side, per_class):
    """The first `per_class` structures of a ranked table whose centres are more than one field side apart."""
    kept = []
    for row in ranked.itertuples():
        if all(np.hypot(row.x_um - other.x_um, row.y_um - other.y_um) > side for other in kept):
            kept.append(row)
        if len(kept) == per_class:
            break
    return kept


def field_candidates(structures, calibre, vessels, um_to_px, he_size_px, side_um=120.0, per_class=4) -> pd.DataFrame:
    """Rank arteriole and venule structures of one section as candidate fields.

    Candidates lie within the registered H&E and away from the capture edge and are ranked by vascular votes:
    arteriole = ``Arterial(muscular)`` with an arterial-EC lining, 20-60 um; venule = ``Venous`` or
    ``Venous(muscular)``, 15-60 um. The top `per_class` per class are kept, at least one field apart.

    Parameters
    ----------
    structures, calibre, vessels
        Structure table, calibre table and lattice cells of the section.
    um_to_px
        3 x 3 matrix from um to H&E pixels.
    he_size_px
        (width, height) of the H&E image.
    """
    he_x0, he_x1, he_y0, he_y1 = _he_bounds_um(um_to_px, he_size_px)
    cells = vessels.obs
    cells_x0, cells_x1, cells_y0, cells_y1 = cells.x_um.min(), cells.x_um.max(), cells.y_um.min(), cells.y_um.max()
    structs = structures.merge(calibre[["struct", "lumen_equiv_diam_um", "vessel_class"]], on="struct")
    margin = side_um / 2 + 10
    inside = (
        (structs.x_um - margin > max(he_x0, cells_x0))
        & (structs.x_um + margin < min(he_x1, cells_x1))
        & (structs.y_um - margin > max(he_y0, cells_y0))
        & (structs.y_um + margin < min(he_y1, cells_y1))
    )
    structs = structs[inside]
    arterioles = structs[
        (structs.vessel_type == "Arterial(muscular)")
        & (structs.ec_identity == "Arterial")
        & structs.lumen_equiv_diam_um.between(20, 60)
    ].assign(klass="arteriole", score=lambda t: t.vote_VSMC + t.vote_ArterialEC)
    venules = structs[
        structs.vessel_type.isin(["Venous", "Venous(muscular)"]) & structs.lumen_equiv_diam_um.between(15, 60)
    ].assign(klass="venule", score=lambda t: t.vote_VenousEC + t.vote_VSMC)
    rows = []
    for klass, table in (("arteriole", arterioles), ("venule", venules)):
        kept = _top_fields(table.sort_values("score", ascending=False), side_um, per_class)
        for rank, row in enumerate(kept, 1):
            rows.append(
                {
                    "rank": rank,
                    "klass": klass,
                    "struct": int(row.struct),
                    "x_um": round(float(row.x_um), 1),
                    "y_um": round(float(row.y_um), 1),
                    "diam_um": round(float(row.lumen_equiv_diam_um), 1),
                    "n_bins": int(row.n_bins),
                    "vessel_type": row.vessel_type,
                    "frac_VSMC": round(float(row.frac_VSMC), 3),
                    "frac_ArterialEC": round(float(row.frac_ArterialEC), 3),
                    "frac_VenousEC": round(float(row.frac_VenousEC), 3),
                    "score": int(row.score),
                }
            )
    return pd.DataFrame(rows)


def _crop_he(image, um_to_px, x0, x1, y0, y1, max_px=1600):
    """Axis-aligned (um) window sampled from the registered H&E at about native resolution.

    Only the block of `image` around the window is read, so a memory-mapped image can be passed.
    """
    um_to_px = np.asarray(um_to_px, float)
    native = float(
        np.mean([np.hypot(um_to_px[0, 0], um_to_px[1, 0]), np.hypot(um_to_px[0, 1], um_to_px[1, 1])])
    )  # px / um
    n_x = min(max_px, int(round((x1 - x0) * native)))
    n_y = int(round(n_x * (y1 - y0) / (x1 - x0)))
    grid_x = x0 + (np.arange(n_x) + 0.5) / n_x * (x1 - x0)
    grid_y = y0 + (np.arange(n_y) + 0.5) / n_y * (y1 - y0)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    px = np.column_stack([mesh_x.ravel(), mesh_y.ravel(), np.ones(mesh_x.size)]) @ um_to_px.T
    px_x, px_y = px[:, 0], px[:, 1]
    height, width = image.shape[:2]
    block_x0 = max(0, int(np.floor(px_x.min())) - 2)
    block_x1 = min(width, int(np.ceil(px_x.max())) + 2)
    block_y0 = max(0, int(np.floor(px_y.min())) - 2)
    block_y1 = min(height, int(np.ceil(px_y.max())) + 2)
    block = np.asarray(image[block_y0:block_y1, block_x0:block_x1, :3])
    crop = np.zeros((n_y * n_x, 3), np.float32)
    for channel in range(3):
        crop[:, channel] = map_coordinates(
            block[:, :, channel], [px_y - block_y0, px_x - block_x0], order=1, mode="nearest"
        )
    return crop.reshape(n_y, n_x, 3).clip(0, 255).astype(np.uint8)


def extract_field(pixels, factor_order, tissue_xy, image, um_to_px, centre, side_um=120.0, spillover=None) -> dict:
    """Cut one square field out of a section: the H&E crop and the 2 um bins with and without a cell type.

    A 2 um bin is assigned the Level-3 type of its highest-probability decode pixel. In-tissue bins without a
    decode pixel are unassigned; with `spillover`, the bins that carry a CMFLAG spillover flag count as
    unassigned too.

    Parameters
    ----------
    pixels
        Decode pixels of the section (``X``, ``Y``, ``K1``, ``P1``).
    factor_order
        Level-3 types in factor order.
    tissue_xy
        (x, y) in um of the in-tissue Space Ranger 2 um bins.
    image
        Registered H&E image (rows x columns x channels).
    um_to_px
        3 x 3 matrix from um to H&E pixels.
    centre
        (x, y) of the field centre in um.

    Returns
    -------
    Dictionary with ``he`` (uint8 crop), ``extent`` (x0, x1, y1, y0), ``ax`` / ``ay`` / ``atype`` (assigned bins
    and their type), ``ux`` / ``uy`` (unassigned bins) and ``fx`` / ``fy`` (bins removed by the spillover flags).
    """
    x0, x1 = centre[0] - side_um / 2, centre[0] + side_um / 2
    y0, y1 = centre[1] - side_um / 2, centre[1] + side_um / 2
    inside = (pixels.X >= x0 - 2) & (pixels.X < x1 + 2) & (pixels.Y >= y0 - 2) & (pixels.Y < y1 + 2)
    window = pixels.loc[inside, ["X", "Y", "K1", "P1"]].reset_index(drop=True)
    window["key"] = bin_key(window.X.to_numpy(), window.Y.to_numpy())
    window = window.sort_values("P1", ascending=False).drop_duplicates("key", keep="first")
    assigned = window.key.to_numpy()
    assigned_type = np.array([factor_order[int(code)] for code in window.K1.to_numpy()], dtype=object)
    tissue_x, tissue_y = tissue_xy
    in_window = (tissue_x >= x0 - 1) & (tissue_x < x1 + 1) & (tissue_y >= y0 - 1) & (tissue_y < y1 + 1)
    tissue = np.unique(bin_key(tissue_x[in_window], tissue_y[in_window]))
    unassigned = np.setdiff1d(tissue, assigned)
    flagged = np.array([], np.int64)
    if spillover is not None:
        flagged = np.intersect1d(np.union1d(assigned, tissue), spillover)
        keep = ~np.isin(assigned, flagged)
        assigned, assigned_type = assigned[keep], assigned_type[keep]
        unassigned = np.union1d(unassigned, flagged)
    return {
        "he": _crop_he(image, um_to_px, x0, x1, y0, y1),
        "extent": np.array([x0, x1, y1, y0]),
        "ax": (assigned // KEYMUL).astype(float) * 2.0,
        "ay": (assigned % KEYMUL).astype(float) * 2.0,
        "atype": assigned_type.astype(str),
        "ux": (unassigned // KEYMUL).astype(float) * 2.0,
        "uy": (unassigned % KEYMUL).astype(float) * 2.0,
        "fx": (flagged // KEYMUL).astype(float) * 2.0,
        "fy": (flagged % KEYMUL).astype(float) * 2.0,
    }


def extract_window(adata, centre, size_um, genes=(), chunk=500_000) -> dict:
    """The 2 um bins of a square window of one section, with their label and the counts of a few genes.

    Parameters
    ----------
    adata
        One section (in memory or backed).
    centre
        (x, y) of the window centre in um.
    size_um
        Side of the window.
    genes
        Genes whose raw counts are returned; a gene that is not in the section gives zeros.

    Returns
    -------
    Dictionary with ``x``, ``y`` (um), ``code`` and ``cats`` (label of a bin = ``cats[code]``), one array of
    counts per gene and ``ext`` (x0, x1, y0, y1).
    """
    x, y = _coordinates(adata)
    categories, codes = _label_codes(adata)
    (cx, cy), half = centre, size_um / 2
    rows = np.where((x >= cx - half) & (x < cx + half) & (y >= cy - half) & (y < cy + half))[0]
    var_names = adata.var_names.to_numpy().astype(str)
    columns = {gene: int(np.where(var_names == gene)[0][0]) for gene in genes if gene in set(var_names)}
    counts = {gene: np.zeros(len(rows), np.float32) for gene in genes}
    if columns and len(rows):
        for start in range(int(rows.min()), int(rows.max()) + 1, chunk):
            stop = min(int(rows.max()) + 1, start + chunk)
            in_block = rows[(rows >= start) & (rows < stop)]
            if not len(in_block):
                continue
            block = adata.X[start:stop].tocsr()[in_block - start]
            position = np.searchsorted(rows, in_block)
            for gene, column in columns.items():
                counts[gene][position] = block[:, column].toarray().ravel()
    return {
        "x": x[rows],
        "y": y[rows],
        "code": codes[rows],
        "cats": categories,
        **counts,
        "ext": np.array([cx - half, cx + half, cy - half, cy + half]),
    }


def label_raster(adata, raster_um=4.0, background=CARDIOMYOCYTES) -> dict:
    """Raster of the Level-3 labels of a whole section, for display.

    A raster pixel takes the most frequent label of its bins that is not in `background`; pixels with background
    bins only take the background label, so the other cell types are drawn on top.

    Returns
    -------
    Dictionary with ``img`` (label code per pixel, -1 = no bin), ``cats`` and ``ext`` (x0, x1, y0, y1 in um).
    """
    x, y = _coordinates(adata)
    categories, codes = _label_codes(adata)
    x0, y0 = x.min(), y.min()
    column = ((x - x0) // raster_um).astype(np.int64)
    row = ((y - y0) // raster_um).astype(np.int64)
    n_columns, n_rows = column.max() + 1, row.max() + 1
    is_background = np.isin(categories[codes], background)
    image = np.full(n_rows * n_columns, -1, np.int16)
    pixel = row * n_columns + column
    image[pixel[is_background]] = codes[is_background]
    votes = pd.DataFrame({"p": pixel[~is_background], "c": codes[~is_background]}).value_counts()
    votes = votes.reset_index(name="n")
    votes = votes.sort_values(["p", "n", "c"], ascending=[True, False, True]).drop_duplicates("p")
    image[votes.p.to_numpy()] = votes.c.to_numpy()
    return {
        "img": image.reshape(n_rows, n_columns),
        "cats": categories,
        "ext": np.array([x0, x0 + n_columns * raster_um, y0, y0 + n_rows * raster_um]),
    }


def he_window(adata, image, um_to_px, centre, size_um, um_per_px=0.25) -> dict:
    """A square field of one section: the registered H&E resampled on the um grid, and its 2 um bins.

    Parameters
    ----------
    adata
        One section (in memory or backed).
    image
        Registered H&E image (rows x columns x channels); a memory-mapped image can be passed.
    um_to_px
        (matrix, offset) of the affine map from um to H&E pixels (``M0``, ``t0`` of the CMFLAG registration).
    centre
        (x, y) of the field centre in um.
    um_per_px
        Resolution of the returned crop.

    Returns
    -------
    Dictionary with ``he`` (uint8 crop) and the bins of :func:`extract_window` (``x``, ``y``, ``code``, ``cats``,
    ``ext``).
    """
    matrix, offset = um_to_px
    (cx, cy), half = centre, size_um / 2
    n = int(round(size_um / um_per_px))
    grid_x = cx - half + (np.arange(n) + 0.5) * um_per_px
    grid_y = cy - half + (np.arange(n) + 0.5) * um_per_px
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    pixels = np.column_stack([mesh_x.ravel(), mesh_y.ravel()]) @ matrix.T + offset
    px, py = pixels[:, 0], pixels[:, 1]
    x0, x1 = max(0, int(px.min()) - 3), min(image.shape[1], int(px.max()) + 4)
    y0, y1 = max(0, int(py.min()) - 3), min(image.shape[0], int(py.max()) + 4)
    block = np.asarray(image[y0:y1, x0:x1, :3]).astype(np.float32)
    crop = np.stack([map_coordinates(block[:, :, c], [py - y0, px - x0], order=1, mode="nearest") for c in range(3)], 1)
    crop = np.clip(crop, 0, 255).astype(np.uint8).reshape(n, n, 3)
    return {"he": crop, **extract_window(adata, centre, size_um)}


def _um_to_px(x_um, y_um, um_to_px, shift=(0.0, 0.0)):
    """H&E pixel coordinates of positions in um, moved by `shift` um before the affine map (matrix, offset)."""
    matrix, offset = um_to_px
    px = (np.column_stack([np.asarray(x_um, float), np.asarray(y_um, float)]) + np.array(shift)) @ matrix.T + offset
    return px[:, 0], px[:, 1]


def _sample_rgb(image, px, py):
    """Bilinear RGB of an image at pixel coordinates; white outside the image."""
    height, width = image.shape[:2]
    rgb = np.full((len(px), 3), 255.0, np.float32)
    covered = (px >= 0) & (px <= width - 1) & (py >= 0) & (py <= height - 1)
    if covered.any():
        x0, x1 = max(0, int(px[covered].min()) - 2), min(width, int(px[covered].max()) + 2)
        y0, y1 = max(0, int(py[covered].min()) - 2), min(height, int(py[covered].max()) + 2)
        block = np.asarray(image[y0:y1, x0:x1, :3])
        for channel in range(3):
            rgb[covered, channel] = map_coordinates(
                block[:, :, channel], [py[covered] - y0, px[covered] - x0], order=1, mode="nearest"
            )
    return rgb


def _grey(rgb):
    """Luma of an RGB array."""
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def registration_offset(
    image, um_to_px, reference, reference_um_per_px, centre, half_um=160.0, res_um=2.0, max_error=0.45, max_um=110.0
) -> dict:
    """Local correction of the H&E registration around one point, by phase correlation with a reference image.

    The H&E and the reference (the CytAssist image, which shares the coordinates of the bins) are sampled in grey
    on a grid of `res_um` in a tile of `half_um` around `centre`; the shift between the two is the correction.

    Parameters
    ----------
    image
        Registered H&E image (rows x columns x channels); a memory-mapped image can be passed.
    um_to_px
        (matrix, offset) of the affine map from um to H&E pixels (``M0``, ``t0`` of the CMFLAG registration).
    reference
        Reference image as RGB array.
    reference_um_per_px
        Pixel size of the reference.
    max_error, max_um
        The correction is applied when the phase-correlation error and both shift components stay below these.

    Returns
    -------
    Dictionary with ``offset_um`` (dx, dy), ``error`` and ``applied``; ``shift`` is the offset to use: the found
    one if it is applied, otherwise (0, 0).
    """
    from skimage.registration import phase_cross_correlation

    grid_x = np.arange(centre[0] - half_um, centre[0] + half_um, res_um)
    grid_y = np.arange(centre[1] - half_um, centre[1] + half_um, res_um)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    shape = (len(grid_y), len(grid_x))
    he_grey = _grey(_sample_rgb(image, *_um_to_px(mesh_x.ravel(), mesh_y.ravel(), um_to_px)).reshape(*shape, 3))
    reference_grey = _grey(np.asarray(reference)).astype(np.float32)
    reference_grey = map_coordinates(
        reference_grey,
        [mesh_y.ravel() / reference_um_per_px, mesh_x.ravel() / reference_um_per_px],
        order=1,
        mode="nearest",
    ).reshape(shape)
    shift, error, _ = phase_cross_correlation(reference_grey, he_grey, upsample_factor=10, normalization=None)
    dx, dy, error = -float(shift[1] * res_um), -float(shift[0] * res_um), float(error)
    applied = bool(error <= max_error and abs(dx) <= max_um and abs(dy) <= max_um)
    return {"offset_um": [dx, dy], "error": error, "applied": applied, "shift": (dx, dy) if applied else (0.0, 0.0)}


def square_window(centre, side_um) -> tuple[float, float, float, float]:
    """Window (x0, x1, y0, y1) of a square field of `side_um` around `centre`."""
    return (centre[0] - side_um / 2, centre[0] + side_um / 2, centre[1] - side_um / 2, centre[1] + side_um / 2)


def in_window(x, y, centre_um, half_size_um) -> np.ndarray:
    """Which positions lie inside a square window: less than `half_size_um` from `centre_um` along x and along y.

    Parameters
    ----------
    x, y
        Positions in um (arrays or Series of the same length).
    centre_um
        (x, y) of the centre of the window.
    half_size_um
        Half the side of the window.

    Returns
    -------
    Boolean array, True for the positions inside the window (the border itself is outside).
    """
    centre_x, centre_y = centre_um
    inside_x = np.abs(np.asarray(x) - centre_x) < half_size_um
    inside_y = np.abs(np.asarray(y) - centre_y) < half_size_um
    return inside_x & inside_y


def _window_rows(x, y, window, margin):
    """Rows of the positions inside a window (x0, x1, y0, y1) widened by `margin`."""
    x0, x1, y0, y1 = window
    return np.where((x >= x0 - margin) & (x < x1 + margin) & (y >= y0 - margin) & (y < y1 + margin))[0]


def registered_field(adata, image, um_to_px, window, shift=(0.0, 0.0), margin_um=4.0, tissue_xy=None) -> dict:
    """A rectangular field of one section: the H&E at its native resolution and the 2 um bins with their label.

    Parameters
    ----------
    adata
        One section (in memory or backed).
    image
        Registered H&E image (rows x columns x channels); a memory-mapped image can be passed.
    um_to_px
        (matrix, offset) of the affine map from um to H&E pixels (``M0``, ``t0`` of the CMFLAG registration).
    window
        (x0, x1, y0, y1) of the field in um.
    shift
        Local correction of the registration in um (:func:`hhca.tl.registration_offset`).
    margin_um
        Bins up to this distance outside the window are included.
    tissue_xy
        (x, y) in um of the in-tissue Space Ranger 2 um bins; with them the in-tissue bins of the field that are
        not in `adata` are returned as unassigned.

    Returns
    -------
    Dictionary with ``he`` (uint8 crop), ``extent`` (x0, x1, y1, y0), ``bx`` / ``by`` / ``blab`` (bins and their
    Level-3 label) and, with `tissue_xy`, ``ux`` / ``uy`` (unassigned bins) and ``n_tissue`` (in-tissue bins).
    """
    x0, x1, y0, y1 = window
    matrix = um_to_px[0]
    native = float(np.mean([np.hypot(matrix[0, 0], matrix[1, 0]), np.hypot(matrix[0, 1], matrix[1, 1])]))  # px / um
    n_x, n_y = int(round((x1 - x0) * native)), int(round((y1 - y0) * native))
    grid_x = x0 + (np.arange(n_x) + 0.5) / n_x * (x1 - x0)
    grid_y = y0 + (np.arange(n_y) + 0.5) / n_y * (y1 - y0)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    he = _sample_rgb(image, *_um_to_px(mesh_x.ravel(), mesh_y.ravel(), um_to_px, shift))
    x, y = _coordinates(adata)
    categories, codes = _label_codes(adata)
    rows = _window_rows(x, y, window, margin_um)
    field = {
        "he": he.reshape(n_y, n_x, 3).clip(0, 255).astype(np.uint8),
        "extent": np.array([x0, x1, y1, y0]),
        "bx": x[rows],
        "by": y[rows],
        "blab": categories[codes[rows]].astype(str),
    }
    if tissue_xy is not None:
        tissue_x, tissue_y = tissue_xy
        in_field = _window_rows(tissue_x, tissue_y, window, 2.0)
        tissue = np.unique(bin_key(tissue_x[in_field], tissue_y[in_field]))
        unassigned = np.setdiff1d(tissue, bin_key(x, y))
        field["ux"] = (unassigned // KEYMUL).astype(float) * 2.0
        field["uy"] = (unassigned % KEYMUL).astype(float) * 2.0
        field["n_tissue"] = int(len(tissue))
    return field


def local_myelin(adata, window, myelin_genes, radius_um=4.0, neural_type="Neural Cells", neural_bins=None) -> dict:
    """Myelin counts around every neural bin of a field.

    Parameters
    ----------
    adata
        One section (in memory or backed).
    window
        (x0, x1, y0, y1) of the field in um.
    myelin_genes
        Genes whose counts are added up.
    radius_um
        The score of a neural bin sums the myelin counts of all bins within this distance.
    neural_bins
        Neural bins of the section with their nerve (:func:`hhca.tl.call_nerves`).

    Returns
    -------
    Dictionary with ``nx``, ``ny`` (neural bins of the field), ``nscore`` (myelin counts within `radius_um`),
    ``nown`` (myelin counts of the bin itself) and, with `neural_bins`, ``nnerve`` (nerve of the bin, -1 = none).
    """
    x, y = _coordinates(adata)
    categories, codes = _label_codes(adata)
    var_names = adata.var_names.to_numpy().astype(str)
    column = {gene: i for i, gene in enumerate(var_names)}
    around = _window_rows(x, y, window, radius_um + 4)
    counts = sp.csr_matrix(adata.X[around])[:, [column[gene] for gene in myelin_genes if gene in column]]
    myelin = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    rows = _window_rows(x, y, window, 4.0)
    neural = rows[categories[codes[rows]] == neural_type]
    neighbours = cKDTree(np.c_[x[around], y[around]]).query_ball_point(np.c_[x[neural], y[neural]], r=radius_um)
    field = {
        "nx": x[neural],
        "ny": y[neural],
        "nscore": np.array([myelin[near].sum() for near in neighbours], float),
        "nown": myelin[np.searchsorted(around, neural)],
    }
    if neural_bins is not None:
        nerve_of_key = pd.Series(neural_bins.nerve.values, index=bin_key(neural_bins.x_um, neural_bins.y_um))
        nerve_of_key = nerve_of_key[~nerve_of_key.index.duplicated()]
        field["nnerve"] = nerve_of_key.reindex(bin_key(x[neural], y[neural])).fillna(-1).astype(int).to_numpy()
    return field
