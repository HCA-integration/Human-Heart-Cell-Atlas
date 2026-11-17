"""Certainty score of the FICTURE decode: four components per decode row, their pooled scaling and the threshold."""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from hhca.stats import robust_norm

GRID_UM = 2.0
BYTE_BITS = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
SCORE_COMPONENTS = ("margin", "depth_raw", "marker_raw", "spatial_raw")
# per-row arrays of the pooled sample: the components and the labels of the precision / coverage curve
SAMPLE_FIELDS = (*SCORE_COMPONENTS, "marker_support", "cross_stable_bin", "refit_recall_flag", "cross_stability")
# share of log1p UMI and of log1p gene richness in the depth component
DEPTH_UMI, DEPTH_GENES = 0.60, 0.40


def model_markers(model, top_n=24, min_enrichment=1.5, expression_quantile=0.70) -> dict[str, list[str]]:
    """Marker genes of each cell type of the FICTURE model.

    A gene is a candidate for a type when its profile value is at least `min_enrichment` times the mean of the
    other types (after adding 1 to both) and lies above the `expression_quantile` of the type's profile.
    Candidates are ranked by ``log2(enrichment) * log1p(expression)`` and the `top_n` are kept; a type with
    fewer candidates takes the `top_n` genes of the ranking over all genes.

    Parameters
    ----------
    model
        Genes x types table of mean expression (the model matrix, columns in factor order).

    Returns
    -------
    Type -> marker genes with the highest rank first, in the order of the columns of `model`.
    """
    genes = model.index.astype(str).to_numpy()
    profiles = model.to_numpy(np.float64)
    total = profiles.sum(axis=1)
    n_types = profiles.shape[1]
    markers = {}
    for i, cell_type in enumerate(model.columns):
        expression = profiles[:, i]
        other = (total - expression) / max(1, n_types - 1)
        enrichment = (expression + 1.0) / (other + 1.0)
        score = np.log2(enrichment) * np.log1p(expression)
        keep = (enrichment >= min_enrichment) & (expression >= np.quantile(expression, expression_quantile))
        candidates = np.where(keep)[0]
        if candidates.size < top_n:
            candidates = np.argsort(score)[-top_n:]
        else:
            candidates = candidates[np.argsort(score[candidates])[-top_n:]]
        markers[cell_type] = list(dict.fromkeys(genes[candidates][::-1].tolist()))
    return markers


def _marker_masks(markers) -> dict[str, int]:
    """For each marker gene, a bit mask of the types (by position in `markers`) it marks."""
    masks: dict[str, int] = {}
    for i, genes in enumerate(markers.values()):
        for gene in genes:
            masks[gene] = masks.get(gene, 0) | (1 << i)
    return masks


def read_bounds(path) -> tuple[float, float, float, float]:
    """xmin, xmax, ymin, ymax (um) of a section from the ``coordinate_minmax.tsv`` of its FICTURE inputs."""
    bounds = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            name, value = line.rstrip("\n").split("\t")
            bounds[name] = float(value)
    return bounds["xmin"], bounds["xmax"], bounds["ymin"], bounds["ymax"]


def _grid(bounds):
    """Origin (um), width and height in cells of the 2 um grid that covers the coordinate range of a section."""
    x0, x1, y0, y1 = bounds
    width = int(math.ceil((x1 - x0) / GRID_UM)) + 1
    height = int(math.ceil((y1 - y0) / GRID_UM)) + 1
    return x0, y0, width, height


def _cell_index(x, y, grid):
    """Flat index of the 2 um grid cell of each coordinate."""
    x0, y0, width, height = grid
    ix = np.clip(((x - x0) / GRID_UM).astype(np.int64), 0, width - 1)
    iy = np.clip(((y - y0) / GRID_UM).astype(np.int64), 0, height - 1)
    return iy * width + ix


def _local_sum_3x3(values, height, width):
    """Sum over the 3 x 3 neighbourhood of every grid cell."""
    padded = np.pad(values.reshape(height, width), 1, mode="constant", constant_values=0)
    out = np.zeros((height, width), dtype=np.float32)
    for dy in (-1, 0, 1):
        ys = 1 + dy
        for dx in (-1, 0, 1):
            xs = 1 + dx
            out += padded[ys : ys + height, xs : xs + width]
    return out.reshape(-1)


def _spatial_agreement(call_by_cell, height, width):
    """Share of the called 8 neighbours that carry the same call as the cell (NaN without called neighbours)."""
    call2d = call_by_cell.reshape(height, width)
    valid = call2d >= 0
    padded = np.pad(call2d, 1, mode="constant", constant_values=-1)
    same = np.zeros((height, width), dtype=np.float32)
    denom = np.zeros((height, width), dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbour = padded[(1 + dy) : (1 + dy + height), (1 + dx) : (1 + dx + width)]
            pair_valid = valid & (neighbour >= 0)
            denom += pair_valid
            same += pair_valid & (neighbour == call2d)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 0, same / denom, np.nan).astype(np.float32)
    return out.reshape(-1)


def _transcript_features(transcripts, grid, call_by_cell, marker_masks, chunksize):
    """Per grid cell: UMI, estimated number of genes (64-bit hash sketch) and UMI of called-type markers."""
    n_cells = grid[2] * grid[3]
    umi = np.zeros(n_cells, dtype=np.float32)
    marker_umi = np.zeros(n_cells, dtype=np.float32)
    gene_bitmap = np.zeros(n_cells, dtype=np.uint64)
    for chunk in pd.read_csv(
        transcripts,
        sep="\t",
        usecols=["X", "Y", "gene", "Count"],
        dtype={"X": np.float32, "Y": np.float32, "gene": str, "Count": np.int16},
        chunksize=chunksize,
    ):
        x = chunk["X"].to_numpy(np.float32, copy=False)
        y = chunk["Y"].to_numpy(np.float32, copy=False)
        count = chunk["Count"].to_numpy(np.int16, copy=False).astype(np.float32, copy=False)
        cell = _cell_index(x, y, grid)

        np.add.at(umi, cell, count)

        gene_hash = pd.util.hash_pandas_object(chunk["gene"], index=False).to_numpy(np.uint64)
        bit = (gene_hash & np.uint64(63)).astype(np.int64)
        packed = np.unique((cell.astype(np.int64) << 6) | bit)
        np.bitwise_or.at(
            gene_bitmap, (packed >> 6).astype(np.int64), np.left_shift(np.uint64(1), (packed & 63).astype(np.uint64))
        )

        masks = chunk["gene"].map(marker_masks).fillna(0).to_numpy(np.uint64, copy=False)
        has_mask = masks > 0
        if np.any(has_mask):
            cell_m = cell[has_mask]
            call = call_by_cell[cell_m]
            called = call >= 0
            if np.any(called):
                is_marker = (
                    (masks[has_mask][called] >> call[called].astype(np.uint64, copy=False)) & np.uint64(1)
                ).astype(bool)
                if np.any(is_marker):
                    np.add.at(marker_umi, cell_m[called][is_marker], count[has_mask][called][is_marker])

    bits_per_cell = BYTE_BITS[gene_bitmap.view(np.uint8)].reshape(-1, 8).sum(axis=1).astype(np.float32)
    frac_zero = np.clip((64.0 - bits_per_cell) / 64.0, 1e-6, 1.0)
    gene_estimate = (-64.0 * np.log(frac_zero)).astype(np.float32)
    return umi, gene_estimate, marker_umi


def certainty_components(decode, inputs, markers, chunksize=1_200_000) -> pd.DataFrame:
    """The four components of the certainty score for every row of a FICTURE pixel decode.

    The decode rows and the transcripts that went into FICTURE are placed on the 2 um grid of the section.

    ``margin``
        ``P1 - P2``, the posterior of the top factor minus that of the second.
    ``depth_raw``
        Local depth: 0.6 log1p(UMI) + 0.4 log1p(estimated number of genes) in the 3 x 3 neighbourhood.
    ``marker_raw``
        Share of the UMI of the 3 x 3 neighbourhood that are markers of the type called in their grid cell.
    ``spatial_raw``
        Share of the 8 neighbouring grid cells with a call that carry the same call.

    Parameters
    ----------
    decode
        Pixel decode of one section (``X``, ``Y``, ``K1``, ``P1``, ``P2``).
    inputs
        FICTURE input folder of the section (``transcripts.tsv.gz`` and ``coordinate_minmax.tsv``).
    markers
        Type -> marker genes in factor order (:func:`model_markers`).
    chunksize
        Transcript rows read at a time.

    Returns
    -------
    One row per decode row: the four components, ``marker_support`` (1 if the neighbourhood holds a marker UMI of
    the called type), ``local_umi`` and ``local_gene_est``.
    """
    x = decode["X"].to_numpy(np.float32)
    y = decode["Y"].to_numpy(np.float32)
    k1 = decode["K1"].to_numpy(np.int16)
    p1 = decode["P1"].to_numpy(np.float32)
    p2 = decode["P2"].to_numpy(np.float32)
    grid = _grid(read_bounds(os.path.join(inputs, "coordinate_minmax.tsv")))
    width, height = grid[2], grid[3]
    cell = _cell_index(x, y, grid)
    call_by_cell = np.full(width * height, -1, dtype=np.int16)
    call_by_cell[cell] = k1

    umi, gene_estimate, marker_umi = _transcript_features(
        os.path.join(inputs, "transcripts.tsv.gz"), grid, call_by_cell, _marker_masks(markers), chunksize
    )
    local_umi = _local_sum_3x3(umi, height, width)
    local_gene = _local_sum_3x3(gene_estimate, height, width)
    local_marker = _local_sum_3x3(marker_umi, height, width)
    del umi, gene_estimate, marker_umi
    local_marker_frac = (local_marker / np.clip(local_umi, 1.0, None)).astype(np.float32)
    spatial = _spatial_agreement(call_by_cell, height, width)
    return pd.DataFrame(
        {
            "margin": np.clip(p1 - p2, 0.0, 1.0).astype(np.float32),
            "depth_raw": (DEPTH_UMI * np.log1p(local_umi[cell]) + DEPTH_GENES * np.log1p(local_gene[cell])).astype(
                np.float32
            ),
            "marker_raw": local_marker_frac[cell].astype(np.float32),
            "spatial_raw": np.nan_to_num(spatial[cell], nan=0.0).astype(np.float32),
            "marker_support": (local_marker[cell] > 0).astype(np.uint8),
            "local_umi": local_umi[cell].astype(np.float32),
            "local_gene_est": local_gene[cell].astype(np.float32),
        },
        index=decode.index,
    )


def decode_keys(x, y) -> np.ndarray:
    """One integer key per decode coordinate (0.01 um resolution), to match rows between decodes."""
    x100 = np.rint(np.asarray(x) * 100.0).astype(np.int64)
    y100 = np.rint(np.asarray(y) * 100.0).astype(np.int64)
    return (x100 << 32) | y100


def match_calls(keys, other_keys, other_k1) -> np.ndarray:
    """K1 of another decode for every row of this decode (-1 where the other decode has no such row)."""
    order = np.argsort(other_keys)
    sorted_keys = other_keys[order]
    sorted_k1 = other_k1[order]
    position = np.searchsorted(sorted_keys, keys)
    out = np.full(keys.shape[0], -1, dtype=np.int16)
    # searchsorted can return the length of the array: clip before indexing, then test the bound
    clipped = np.clip(position, 0, sorted_keys.shape[0] - 1)
    good = (position < sorted_keys.shape[0]) & (sorted_keys[clipped] == keys)
    out[good] = sorted_k1[clipped[good]]
    return out


def unmatched_rows(keys, other_keys) -> np.ndarray:
    """True for the rows of `keys` without a row of the same key in `other_keys`."""
    target = np.sort(other_keys)
    position = np.searchsorted(target, keys)
    clipped = np.clip(position, 0, target.shape[0] - 1)
    return ~((position < target.shape[0]) & (target[clipped] == keys))


def call_stability(k1, alternative_calls):
    """Agreement of the call of each decode row with its call in alternative decodes.

    Parameters
    ----------
    k1
        Top factor per decode row.
    alternative_calls
        Top factor per row in each alternative decode (-1 where the decode has no such row).

    Returns
    -------
    Share of the available alternative decodes with the same call (NaN without any), and whether all of them
    agree (uint8).
    """
    available = np.zeros(k1.shape[0], dtype=np.float32)
    matching = np.zeros(k1.shape[0], dtype=np.float32)
    for alternative in alternative_calls:
        present = alternative >= 0
        available[present] += 1.0
        matching[present] += (alternative[present] == k1[present]).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        stability = np.where(available > 0, matching / available, np.nan).astype(np.float32)
    stable = ((available > 0) & (matching == available)).astype(np.uint8)
    return stability, stable


def certainty_score(components, quantiles, weights) -> np.ndarray:
    """Weighted sum of the four components, each scaled to [0, 1] between its pooled quantiles.

    Parameters
    ----------
    components
        Mapping with ``margin``, ``depth_raw``, ``marker_raw`` and ``spatial_raw`` (table of
        :func:`certainty_components` or a checkpoint of the pipeline).
    quantiles
        Component -> (low, high), the values scaled to 0 and 1.
    weights
        Weights of ``posterior`` (the margin), ``depth``, ``marker`` and ``spatial``.
    """
    margin, depth, marker, spatial = (np.asarray(components[name]) for name in SCORE_COMPONENTS)
    return (
        weights["posterior"] * robust_norm(margin, *quantiles["margin"])
        + weights["depth"] * robust_norm(depth, *quantiles["depth_raw"])
        + weights["marker"] * robust_norm(marker, *quantiles["marker_raw"])
        + weights["spatial"] * robust_norm(spatial, *quantiles["spatial_raw"])
    ).astype(np.float32)


def sample_rows(checkpoints, n_per_section=180_000, seed=7, fields=SAMPLE_FIELDS) -> dict[str, np.ndarray]:
    """Up to `n_per_section` random decode rows of every section, pooled in the order of `checkpoints`.

    Parameters
    ----------
    checkpoints
        Section -> per-row arrays (the feature checkpoints of the pipeline); the order fixes the random draw.
    fields
        Arrays to sample.
    """
    rng = np.random.default_rng(seed)
    sample: dict[str, list] = {field: [] for field in fields}
    for values in checkpoints.values():
        n_rows = values["margin"].shape[0]
        if n_rows <= n_per_section:
            rows = np.arange(n_rows)
        else:
            rows = rng.choice(n_rows, size=n_per_section, replace=False)
            rows.sort()
        for field in fields:
            sample[field].append(values[field][rows])
    return {field: np.concatenate(parts) for field, parts in sample.items()}


def component_quantiles(sample, low=0.05, high=0.95) -> dict[str, list[float]]:
    """The `low` and `high` quantile of each score component in a pooled sample."""
    return {
        name: [float(np.nanquantile(sample[name], low)), float(np.nanquantile(sample[name], high))]
        for name in SCORE_COMPONENTS
    }


def precision_coverage(sample, quantiles, weights, n_thresholds=101) -> pd.DataFrame:
    """Precision and coverage of the certainty score at `n_thresholds` thresholds between 0 and 1.

    A sampled row counts as positive when it has marker support and the same call in the alternative decodes;
    rows without an alternative decode are left out. Coverage is the share of rows at or above a threshold,
    precision the share of positives among them.

    Returns
    -------
    One row per threshold: ``coverage``, ``precision``, ``marker_rate``, ``stability_rate`` and ``n_selected``;
    ``attrs["n_rows"]`` holds the number of evaluated rows.
    """
    evaluated = np.isfinite(sample["cross_stability"])
    score = certainty_score(sample, quantiles, weights)[evaluated].astype(np.float32)
    marker_supported = sample["marker_support"][evaluated].astype(bool)
    stable = sample["cross_stable_bin"][evaluated].astype(bool)
    positive = marker_supported & stable
    rows = []
    for threshold in np.linspace(0.0, 1.0, n_thresholds):
        selected = score >= threshold
        if selected.sum() > 0:
            precision = float(positive[selected].mean())
            marker_rate = float(marker_supported[selected].mean())
            stability_rate = float(stable[selected].mean())
        else:
            precision = marker_rate = stability_rate = np.nan
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(selected.mean()),
                "precision": precision,
                "marker_rate": marker_rate,
                "stability_rate": stability_rate,
                "n_selected": int(selected.sum()),
            }
        )
    curve = pd.DataFrame(rows)
    curve.attrs["n_rows"] = int(score.shape[0])
    return curve


def operating_point(curve, target_precision, min_coverage=0.02) -> dict:
    """Lowest threshold (largest coverage) of a precision / coverage curve whose precision reaches the target."""
    reached = curve[(curve["precision"] >= target_precision) & (curve["coverage"] >= min_coverage)]
    if len(reached):
        row = reached.sort_values("threshold").iloc[0]
    else:
        # target not reached: the point of highest precision
        row = curve.sort_values("precision", ascending=False).iloc[0]
    return {
        "target_precision": float(target_precision),
        "threshold": float(row["threshold"]),
        "precision": float(row["precision"]),
        "coverage": float(row["coverage"]),
        "marker_rate": float(row["marker_rate"]),
        "stability_rate": float(row["stability_rate"]),
    }


def select_threshold(curve, target_precision=0.99, min_coverage=0.02) -> float:
    """Threshold on the certainty score: the lowest one that reaches the target precision.

    Precision is already high when every row is kept, so the target must lie above that baseline for the
    threshold to remove the least certain rows: if the baseline reaches the target, the target moves half-way
    from the baseline to 1 (at most 0.999).
    """
    baseline = float(curve.sort_values("threshold").iloc[0]["precision"])
    target = target_precision
    if baseline >= target:
        target = min(0.999, baseline + 0.5 * (1.0 - baseline))
    return operating_point(curve, target, min_coverage)["threshold"]
